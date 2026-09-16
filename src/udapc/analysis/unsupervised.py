import numpy as np
import pandas as pd
import itertools

from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import adjusted_rand_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

from udapc.analysis.base import DiscriminantAnalysis
from udapc.analysis.solvers.unrtlda import UnRTLDA
from udapc.analysis.solvers.untrlda import UnTRLDA
from udapc.analysis.solvers.ldakm import LDAKm
from udapc.utils.pca_tools import (choose_npcs_by_variance,
                                   n_significant_pcs_mp)
from udapc.utils.alignment import multiway_align, membership_consistency
from snpio import GenotypeEncoder


class UnsupervisedDiscriminantAnalysis(DiscriminantAnalysis):
    def __init__(
        self,
        genotype_data,
        solver="unrtlda",
        n_pca=None,
        n_pca_min=10,
        n_pca_max=300,
        n_pca_step=10,
        reps=10,
        n_folds=5,
        subset=1.0,
        fixed_k=None,
        max_k=8,
        prop_pc_var=None,
        solver_kwargs=None,
        selection_rule="max",
        scoring="membership",
        linked_npc=True,
        null_correction=False,
        n_null=1,
        n_pca_mp=False,
        n_pca_var=None,
        impute="mode",
        random_state=42,
    ):
        super().__init__(genotype_data)
        self.genotype_data = genotype_data
        self.selection_rule = selection_rule
        self.scoring = scoring
        self.linked_npc = linked_npc
        self.n_folds = n_folds
        # Null correction (gap statistic, after Tibshirani et al. 2001).
        # Raw CV stability is not comparable across (K, nPC): some reproducibility
        # comes free from the geometry, and how much depends on the cell. With few
        # PCs, k-means on a structure-free cloud is already highly reproducible
        # (measured: 0.87 at nPC=2), so raw stability is maximised at the smallest
        # nPC and the smallest K regardless of the data. Subtracting the score
        # obtained on structure-free genotypes removes that baseline, leaving only
        # reproducibility the structure actually earned.
        # Three ways to fix nPC instead of searching it: an explicit n_pca, the
        # Marchenko-Pastur noise edge, or a proportion of variance. All three are
        # computed from the genotype matrix before any clustering, so they yield a
        # single nPC for the whole run -- there is no K for them to vary over, and
        # per-K (unlinked) nPC has nothing to do. K is still searched normally.
        if n_pca_mp and n_pca_var is not None:
            raise ValueError("n_pca_mp and n_pca_var are mutually exclusive")
        if (n_pca_mp or n_pca_var is not None) and isinstance(n_pca, int):
            raise ValueError("n_pca is already fixed; drop n_pca_mp / n_pca_var")
        if n_pca_var is not None and not 0 < n_pca_var < 1:
            raise ValueError(f"n_pca_var must be in (0, 1), got {n_pca_var}")
        # Missing-genotype imputation. "mode" and "mean" use the genotype matrix
        # alone. "pop_mode" imputes within the popmap groups, which is common
        # practice but means the labels influence the clustering -- the one place
        # anything outside the genotype matrix can reach the model. Where the
        # popmap is wrong, it makes individuals resemble their mislabelled group
        # before clustering begins, so prefer "mode" unless missingness is high
        # and the groups are trusted.
        if impute not in ("mode", "mean", "pop_mode"):
            raise ValueError(f"impute must be mode, mean or pop_mode, got {impute!r}")
        self.impute = impute
        self.n_pca_mp = n_pca_mp
        self.n_pca_var = n_pca_var
        self.null_correction = null_correction
        self.n_null = n_null
        self.random_state = random_state
        self._null_coords = None

        solver_registry = {
            "unrtlda": UnRTLDA,
            "untrlda": UnTRLDA,
            "ldakm": LDAKm,
        }
        self.solver_cls = (
            solver_registry[solver.lower()] if isinstance(solver, str) else solver
        )
        self.solver_kwargs = solver_kwargs or {}
        self.n_pca = n_pca
        self.n_pca_min = n_pca_min
        self.n_pca_max = n_pca_max
        self.n_pca_step = n_pca_step
        self.reps = reps
        self.subset = subset
        self.fixed_k = fixed_k
        self.max_k = max_k
        self.prop_pc_var = prop_pc_var

        self.results = {}
        self.metrics = None
        # Stores resampling label matrices keyed by (k, npc) or k
        self._resample_labels = {}
        # Stores full-data fits keyed by k
        self._fits_by_k = {}

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _score_name(self):
        """What the Stability column actually holds, for logs and plots."""
        return f"gap({self.scoring})" if self.null_correction else self.scoring

    def _make_null_coords(self, X_raw, n_components):
        """PC coordinates for structure-free data matching X_raw.

        Genotypes are permuted independently within each locus. That preserves
        every locus's allele frequency while destroying covariance *between*
        loci -- which is the only place population structure lives. The result is
        the same data with the structure removed, which is the reference the gap
        statistic needs.

        Uses no labels: the null is built from the genotype matrix alone.
        """
        rng = np.random.RandomState(self.random_state)
        mats = []
        for _ in range(max(1, self.n_null)):
            Xp = np.column_stack(
                [rng.permutation(X_raw[:, j]) for j in range(X_raw.shape[1])]
            )
            mats.append(PCA(n_components=n_components).fit_transform(Xp))
        return mats

    def _score_cell(self, coords, k, null_coords=None):
        """Stability for one (K, nPC) cell, optionally null-corrected.

        Returns (score, se, raw, null, labels). When null correction is off the
        score is the raw stability and `null` is 0.0, so downstream selection is
        unchanged.
        """
        raw, se, labels = self._stability_score(coords, k)
        if not self.null_correction or not null_coords:
            return raw, se, raw, 0.0, labels

        p = coords.shape[1]
        null_scores, null_ses = [], []
        for nc in null_coords:
            ns, nse, _ = self._stability_score(nc[:, :p], k)
            null_scores.append(ns)
            null_ses.append(nse)
        null = float(np.mean(null_scores))
        # difference of two independent estimates
        se_gap = float(np.sqrt(se ** 2 + np.mean(null_ses) ** 2))
        return raw - null, se_gap, raw, null, labels

    def _get_numeric_matrix(self):
        """Genotypes as 0/1/2 with missing values imputed.

        Imputation matters more than it looks on data with substantial
        missingness: replacing a large fraction of genotypes with a single value
        per locus manufactures similarity between individuals whose missingness
        overlaps, and missingness in reduced-representation data tracks coverage
        and library batch rather than ancestry.
        """
        encoder = GenotypeEncoder(self.genotype_data)
        X = encoder.genotypes_012.copy().astype(float)
        miss = X == -9
        if not miss.any():
            return X

        def fill(vals, mask):
            """Imputed value for one locus within one group of individuals."""
            obs = vals[~mask]
            if obs.size == 0:
                return None
            if self.impute == "mean":
                return float(obs.mean())
            return float(np.bincount(obs.astype(int)).argmax())

        if self.impute == "pop_mode":
            pops = np.asarray(self.genotype_data.populations)
            for col in np.flatnonzero(miss.any(axis=0)):
                vals, mask = X[:, col], miss[:, col]
                glob = fill(vals, mask)
                for p in np.unique(pops):
                    sel = pops == p
                    if not mask[sel].any():
                        continue
                    v = fill(vals[sel], mask[sel])
                    # fall back to the global value where a group is wholly missing
                    X[np.flatnonzero(sel)[mask[sel]], col] = (
                        v if v is not None else (glob if glob is not None else 0.0))
        else:
            for col in np.flatnonzero(miss.any(axis=0)):
                v = fill(X[:, col], miss[:, col])
                X[miss[:, col], col] = 0.0 if v is None else v
        return X

    def _select_parsimonious(self, metrics_df):
        """Select the most parsimonious (K, nPC) from the stability grid."""
        if self.selection_rule == "max":
            return metrics_df.sort_values(
                ["Stability", "K", "n_PCs"], ascending=[False, True, True]
            ).iloc[0]

        best_idx = metrics_df["Stability"].idxmax()
        best_stab = metrics_df.loc[best_idx, "Stability"]

        # 1se: everything within one standard error of the best.
        # (A multiplicative "within 95% of best" rule was removed: it assumes a
        # positive ratio scale, which null-corrected scores violate -- for a
        # negative best, multiplying by 0.95 raises the threshold above the max.)
        best_se = metrics_df.loc[best_idx, "SE"]
        threshold = best_stab - best_se

        plateau = metrics_df[metrics_df["Stability"] >= threshold]
        selected = plateau.sort_values(["K", "n_PCs"]).iloc[0]

        print(f"\n{'='*60}")
        print(f"Model Selection ({self.selection_rule} rule, "
              f"scoring={self._score_name()}):")
        print(
            f"  Best stability: {best_stab:.4f} at K={int(metrics_df.loc[best_idx, 'K'])}, "
            f"nPC={int(metrics_df.loc[best_idx, 'n_PCs'])}"
        )
        print(f"  Threshold: {threshold:.4f}")
        print(f"  Plateau size: {len(plateau)} cells")
        print(
            f"  Selected: K={int(selected.K)}, nPC={int(selected.n_PCs)}, "
            f"Stability={selected.Stability:.4f}"
        )
        print(f"{'='*60}")

        return selected

    def _stability_score(self, coords, k):
        """Resampling stability for one (K, nPC) cell.

        Repeatedly refit on a random subset of individuals and measure how much
        the resulting clusterings agree with each other. This is resampling
        stability (Ben-Hur et al. 2002; Monti et al. 2003; Lange et al. 2004),
        NOT cross-validation: nothing is held out and nothing is predicted. The
        PCA was fitted on every individual, the model labels every individual in
        every draw, and predict() re-clusters from scratch rather than applying
        the centroids just learned. The training subset only ever determines the
        projection.

        The question answered is "would a slightly different sample of
        individuals give the same clustering", not "does this model predict
        unseen data". Both are useful; only the first is what this computes.

        Returns:
            (score, se, label_matrix)
        """
        rkf = RepeatedKFold(
            n_splits=self.n_folds, n_repeats=self.reps, random_state=42
        )
        n_reps = self.n_folds * self.reps
        n_samples = coords.shape[0]

        labels = np.full((n_samples, n_reps), -1, dtype=int)
        for idx, (train_idx, _) in enumerate(rkf.split(coords)):
            if self.subset < 1.0:
                # thin the draw further; seeded per draw so a run is reproducible
                sub_n = int(self.subset * len(train_idx))
                rng = np.random.RandomState(self.random_state + idx)
                train_idx = rng.choice(train_idx, sub_n, replace=False)
            model = self.solver_cls(
                c=k, Npc=coords.shape[1], no_pca=True, **self.solver_kwargs
            )
            model.fit(coords[train_idx])
            labels[:, idx] = model.predict(coords)

        if self.scoring == "membership":
            aligned, Q = multiway_align(labels, k)
            mc = membership_consistency(Q)
            score = float(np.mean(mc))
            # Bootstrap SE estimate for membership consistency
            rng = np.random.RandomState(42)
            boot_scores = []
            for _ in range(200):
                idx = rng.choice(n_samples, n_samples, replace=True)
                boot_scores.append(float(np.mean(np.max(Q[idx], axis=1))))
            se = float(np.std(boot_scores, ddof=1))
        else:
            # Pairwise ARI (permutation-invariant, no alignment needed)
            aris = [
                adjusted_rand_score(labels[:, i], labels[:, j])
                for i, j in itertools.combinations(range(n_reps), 2)
            ]
            score = float(np.mean(aris))
            se = float(np.std(aris, ddof=1) / np.sqrt(len(aris)))

        return score, se, labels

    def _fit_all_k(self, coords_max, k_values, npc_values):
        """Run full-data fits for all (K, nPC) and store in _fits_by_k."""
        for k in k_values:
            if k in self._fits_by_k:
                continue
            # Use the best nPC for this K from metrics
            if self.metrics is not None:
                k_rows = self.metrics[self.metrics["K"] == k]
                if not k_rows.empty:
                    best_npc = int(
                        k_rows.sort_values("Stability", ascending=False).iloc[0][
                            "n_PCs"
                        ]
                    )
                else:
                    best_npc = npc_values[0]
            else:
                best_npc = npc_values[0]

            coords = coords_max[:, :best_npc]
            model = self.solver_cls(
                c=k, Npc=best_npc, no_pca=True, **self.solver_kwargs
            )
            T, G, W, log = model.fit(coords)
            self._fits_by_k[k] = {
                "T": T, "G": G, "W": W, "n_pcs": best_npc,
                "coords": coords, "obj_log": log,
            }

    # ------------------------------------------------------------------ #
    #  Main entry point                                                    #
    # ------------------------------------------------------------------ #

    def fit_predict(self):
        X_raw = self._get_numeric_matrix()
        n_s, n_f = X_raw.shape

        # Resolve a data-derived nPC before anything else, so every route ends up
        # in the same fixed-nPC branch below (which still searches K).
        if self.n_pca_mp or self.n_pca_var is not None:
            if self.n_pca_mp:
                n_fix = n_significant_pcs_mp(X_raw)
                how = "Marchenko-Pastur"
            else:
                _p = PCA(n_components=min(n_s, n_f)).fit(X_raw)
                n_fix = choose_npcs_by_variance(_p, self.n_pca_var)
                how = f"{self.n_pca_var:.0%} of variance"
            self.n_pca = int(min(max(n_fix, 2), n_s - 1, n_f))
            print(f"[nPC] {how}: using nPC={self.n_pca}; searching K only")
            if not self.linked_npc:
                print("[nPC] note: nPC is fixed, so --unlinked_npc has no effect")

        # --- Fixed nPC ---
        if isinstance(self.n_pca, int):
            pca = PCA(n_components=min(self.n_pca, n_s, n_f))
            coords_fixed = pca.fit_transform(X_raw)[:, : self.n_pca]

            if self.fixed_k is not None:
                # Both fixed: single run, no CV
                model = self.solver_cls(
                    c=self.fixed_k, Npc=self.n_pca, no_pca=True,
                    **self.solver_kwargs,
                )
                T, G, W, log = model.fit(coords_fixed)
                self.results = {
                    "fixed_k": self.fixed_k, "n_pcs": self.n_pca,
                    "G": G, "T": T, "coords": coords_fixed, "obj_log": log,
                }
                self.metrics = pd.DataFrame(
                    [{"K": self.fixed_k, "n_PCs": self.n_pca, "Stability": 1.0, "SE": 0.0}]
                )
                self._fits_by_k[self.fixed_k] = {
                    "T": T, "G": G, "W": W, "n_pcs": self.n_pca,
                    "coords": coords_fixed, "obj_log": log,
                }
                return

            # Fixed nPC, search K
            records = []
            for k in range(2, self.max_k + 1):
                stab, se, lbl = self._stability_score(coords_fixed, k)
                print(f"[stability] K={k}, PCs={self.n_pca} -> "
                      f"{self._score_name()}={stab:.4f} +/- {se:.4f}")
                records.append({"K": k, "n_PCs": self.n_pca, "Stability": stab, "SE": se})
                self._resample_labels[(k, self.n_pca)] = lbl

            self.metrics = pd.DataFrame(records)
            best = self._select_parsimonious(self.metrics)
            bk = int(best.K)

            model = self.solver_cls(
                c=bk, Npc=self.n_pca, no_pca=True, **self.solver_kwargs
            )
            T, G, W, log = model.fit(coords_fixed)
            self.results = {
                "fixed_k": bk, "n_pcs": self.n_pca,
                "G": G, "T": T, "coords": coords_fixed, "obj_log": log,
                "resample_labels": self._resample_labels.get((bk, self.n_pca)),
            }

            # Full-data fits for all K (for multi-K plots)
            for k in range(2, self.max_k + 1):
                m = self.solver_cls(
                    c=k, Npc=self.n_pca, no_pca=True, **self.solver_kwargs
                )
                T_k, G_k, W_k, log_k = m.fit(coords_fixed)
                self._fits_by_k[k] = {
                    "T": T_k, "G": G_k, "W": W_k, "n_pcs": self.n_pca,
                    "coords": coords_fixed, "obj_log": log_k,
                }
            return

        # (A float n_pca formerly selected nPC by variance here, but that path
        # hard-coded k = fixed_k or 2 and never searched K. Variance-based nPC is
        # now resolved above via n_pca_var and falls through to the K search.)

        # --- Range run: search over nPC and K ---
        p_min, p_max, p_step = (
            self.n_pca
            if isinstance(self.n_pca, (tuple, list)) and len(self.n_pca) == 3
            else (self.n_pca_min, min(self.n_pca_max, n_s, n_f), self.n_pca_step)
        )
        pca_full = PCA(n_components=p_max)
        coords_max = pca_full.fit_transform(X_raw)
        npc_values = list(range(p_min, p_max + 1, p_step))

        # One permutation + PCA, reused across the whole (K, nPC) grid: the null
        # depends only on the data, not on the cell being scored.
        self._null_coords = (
            self._make_null_coords(X_raw, p_max) if self.null_correction else None
        )
        if self.null_correction:
            print(f"[null] built {len(self._null_coords)} structure-free "
                  f"reference(s) for gap correction")

        if self.linked_npc:
            # Joint grid search: single (K, nPC) selection
            records = []
            for p in npc_values:
                coords_p = coords_max[:, :p]
                for k in range(2, self.max_k + 1):
                    stab, se, raw, null, lbl = self._score_cell(
                        coords_p, k, self._null_coords
                    )
                    print(f"[stability] K={k}, PCs={p} -> {self._score_name()}={stab:.4f} "
                          f"+/- {se:.4f}" +
                          (f"  (raw={raw:.4f} null={null:.4f})"
                           if self.null_correction else ""))
                    records.append({"K": k, "n_PCs": p, "Stability": stab, "SE": se,
                                    "Raw": raw, "Null": null})
                    self._resample_labels[(k, p)] = lbl

            self.metrics = pd.DataFrame(records)
            best = self._select_parsimonious(self.metrics)
            bk, bp = int(best.K), int(best.n_PCs)

        else:
            # Independent nPC per K: for each K, find best nPC
            records = []
            best_by_k = {}
            for k in range(2, self.max_k + 1):
                k_records = []
                for p in npc_values:
                    coords_p = coords_max[:, :p]
                    stab, se, raw, null, lbl = self._score_cell(
                        coords_p, k, self._null_coords
                    )
                    print(f"[stability] K={k}, PCs={p} -> {self._score_name()}={stab:.4f} "
                          f"+/- {se:.4f}" +
                          (f"  (raw={raw:.4f} null={null:.4f})"
                           if self.null_correction else ""))
                    row = {"K": k, "n_PCs": p, "Stability": stab, "SE": se,
                           "Raw": raw, "Null": null}
                    records.append(row)
                    k_records.append(row)
                    self._resample_labels[(k, p)] = lbl

                # Best nPC for this K
                k_df = pd.DataFrame(k_records)
                best_p = int(
                    k_df.sort_values("Stability", ascending=False).iloc[0]["n_PCs"]
                )
                best_stab = k_df["Stability"].max()
                best_by_k[k] = {"n_PCs": best_p, "Stability": best_stab}

            self.metrics = pd.DataFrame(records)

            # Select K using each K's best nPC
            marginal = pd.DataFrame(
                [{"K": k, "n_PCs": v["n_PCs"], "Stability": v["Stability"],
                  "SE": self.metrics.loc[
                      (self.metrics["K"] == k) & (self.metrics["n_PCs"] == v["n_PCs"]),
                      "SE"].values[0]}
                 for k, v in best_by_k.items()]
            )
            best = self._select_parsimonious(marginal)
            bk = int(best.K)
            bp = int(best.n_PCs)

        coords = coords_max[:, :bp]
        model = self.solver_cls(
            c=bk, Npc=bp, no_pca=True, **self.solver_kwargs
        )
        T, G, W, log = model.fit(coords)
        self.results = {
            "fixed_k": bk, "n_pcs": bp,
            "G": G, "T": T, "coords": coords, "obj_log": log,
            "resample_labels": self._resample_labels.get((bk, bp)),
        }

        # Full-data fits for all K at their respective best nPC
        for k in range(2, self.max_k + 1):
            if not self.linked_npc and k in best_by_k:
                k_npc = best_by_k[k]["n_PCs"]
            else:
                k_npc = bp
            c = coords_max[:, :k_npc]
            m = self.solver_cls(c=k, Npc=k_npc, no_pca=True, **self.solver_kwargs)
            T_k, G_k, W_k, log_k = m.fit(c)
            self._fits_by_k[k] = {
                "T": T_k, "G": G_k, "W": W_k, "n_pcs": k_npc,
                "coords": c, "obj_log": log_k,
            }

    # ------------------------------------------------------------------ #
    #  Accessors                                                           #
    # ------------------------------------------------------------------ #

    def score(self):
        return self.metrics

    def _get_sample_order(self):
        """Stable sample ordering: by population, then sample ID. Cached."""
        if not hasattr(self, "_cached_sample_order"):
            pops = np.array(self.genotype_data.populations)
            samples = np.array(self.genotype_data.samples)
            self._cached_sample_order = np.lexsort((samples, pops))
        return self._cached_sample_order

    def _pop_dividers(self, pops_sorted):
        """Compute population divider positions and label centers."""
        dividers, centers = [], []
        prev, start = None, 0
        for idx, p in enumerate(pops_sorted):
            if p != prev:
                if prev is not None:
                    dividers.append(idx - 0.5)
                    centers.append(((start + idx - 1) / 2, prev))
                start = idx
                prev = p
        centers.append(((start + len(pops_sorted) - 1) / 2, prev))
        return dividers, centers

    # ------------------------------------------------------------------ #
    #  Plot: PCA + DA embeddings for each K                                #
    # ------------------------------------------------------------------ #

    def plot_all_k(self, filename=None):
        """PCA + DA scatter for each K tested (full-data fits)."""
        if not self._fits_by_k:
            print("No per-K fits available.")
            return

        k_values = sorted(self._fits_by_k.keys())
        pops = np.array(self.genotype_data.populations)
        samples = np.array(self.genotype_data.samples)
        n_k = len(k_values)
        sel_k = self.results.get("fixed_k")

        fig, axes = plt.subplots(n_k, 2, figsize=(10, 3.2 * n_k))
        if n_k == 1:
            axes = axes[np.newaxis, :]

        cmap = plt.colormaps.get_cmap("tab10")

        all_rows = []
        for ri, k in enumerate(k_values):
            fit = self._fits_by_k[k]
            T, G, coords = fit["T"], fit["G"], fit["coords"]
            n_pcs = fit["n_pcs"]

            # PCA plot
            ax_pca = axes[ri, 0]
            for c in range(k):
                mask = G == c
                ax_pca.scatter(coords[mask, 0], coords[mask, 1], s=10,
                              alpha=0.7, color=cmap(c), label=f"C{c}")
            ax_pca.set_xlabel("PC1"); ax_pca.set_ylabel("PC2")
            title_prefix = "* " if k == sel_k else ""
            ax_pca.set_title(f"{title_prefix}K={k} (nPC={n_pcs}) — PCA", fontsize=9)
            ax_pca.set_xticks([]); ax_pca.set_yticks([])

            # DA plot
            ax_da = axes[ri, 1]
            if self._n_da(k, T) >= 2:
                for c in range(k):
                    mask = G == c
                    ax_da.scatter(T[mask, 0], T[mask, 1], s=10,
                                alpha=0.7, color=cmap(c))
                ax_da.set_xlabel("DA1"); ax_da.set_ylabel("DA2")
            else:
                # Only one discriminant axis exists (K=2): show its
                # distribution per cluster rather than inventing a y-axis.
                for c in range(k):
                    mask = G == c
                    ax_da.hist(T[mask, 0], bins=25, alpha=0.55,
                               color=cmap(c), density=True)
                ax_da.set_xlabel("DA1"); ax_da.set_ylabel("density")
            ax_da.set_title(f"{title_prefix}K={k} — DA Embedding", fontsize=9)
            ax_da.set_xticks([]); ax_da.set_yticks([])

            # Collect assignment table
            for i in range(len(G)):
                all_rows.append({
                    "Sample": samples[i], "Population": pops[i],
                    "K": k, "Cluster": int(G[i]),
                })

        fig.suptitle(f"{self.solver_cls.__name__} — All K", fontweight="bold")
        plt.tight_layout()
        if filename:
            fig.savefig(filename, dpi=150, bbox_inches="tight")
            plt.close(fig)
            tsv = filename.rsplit(".", 1)[0] + ".tsv"
            pd.DataFrame(all_rows).to_csv(tsv, sep="\t", index=False)
            print(f"Saved: {filename}, {tsv}")
        else:
            plt.show()

    # ------------------------------------------------------------------ #
    #  Plot: assignment proportion plots for all K                          #
    # ------------------------------------------------------------------ #

    def plot_all_assignments(self, filename=None):
        """Stack of assignment proportion plots, one per K, consistent ordering."""
        if not self._resample_labels:
            print("No CV label data available.")
            return

        # Get unique K values that have CV labels
        k_values = sorted(set(k for k, _ in self._resample_labels.keys()))
        if not k_values:
            print("No CV label data available.")
            return

        pops = np.array(self.genotype_data.populations)
        samples = np.array(self.genotype_data.samples)
        n = len(pops)
        order = self._get_sample_order()
        pops_sorted = pops[order]
        dividers, centers = self._pop_dividers(pops_sorted)
        sel_k = self.results.get("fixed_k")
        cmap = plt.colormaps.get_cmap("tab10")

        n_k = len(k_values)
        fig, axes = plt.subplots(n_k, 1, figsize=(14, 2.5 * n_k + 1))
        if n_k == 1:
            axes = [axes]

        for ki, k in enumerate(k_values):
            ax = axes[ki]

            # Find the best nPC for this K from metrics
            if self.metrics is not None:
                k_rows = self.metrics[self.metrics["K"] == k]
                if not k_rows.empty:
                    best_npc = int(
                        k_rows.sort_values("Stability", ascending=False).iloc[0]["n_PCs"]
                    )
                    lbl = self._resample_labels.get((k, best_npc))
                    stab = k_rows["Stability"].max()
                else:
                    lbl = None; best_npc = 0; stab = 0
            else:
                lbl = None; best_npc = 0; stab = 0

            if lbl is None:
                ax.text(0.5, 0.5, f"K={k}: no CV data", ha="center", va="center",
                       transform=ax.transAxes)
                continue

            aligned, Q = multiway_align(lbl, k)
            mc = membership_consistency(Q)
            Q_sorted = Q[order]

            bottom = np.zeros(n)
            for j in range(k):
                ax.bar(range(n), Q_sorted[:, j], bottom=bottom, width=1.0,
                       color=cmap(j), edgecolor='none')
                bottom += Q_sorted[:, j]

            for xpos in dividers:
                ax.axvline(xpos, color="black", linewidth=0.5)

            ax.set_xlim(-0.5, n - 0.5)
            ax.set_ylim(0, 1)
            ax.set_ylabel("Prop.")
            prefix = "* " if k == sel_k else ""
            ax.set_title(
                f"{prefix}K={k}  nPC={best_npc}  "
                f"{self.scoring}={stab:.3f}  membership={np.mean(mc):.3f}",
                fontsize=9,
            )
            ax.set_xticks([])

            # Population labels on bottom panel only
            if ki == n_k - 1:
                for center, label in centers:
                    ax.text(center, -0.08, label, ha="center", va="top",
                           fontsize=6, rotation=45,
                           transform=ax.get_xaxis_transform())

            # Save per-K TSV
            if filename:
                tsv_k = filename.rsplit(".", 1)[0] + f"_k{k}.tsv"
                q_df = pd.DataFrame(Q, columns=[f"Cluster_{j}" for j in range(k)])
                q_df.insert(0, "Sample", samples)
                q_df.insert(1, "Population", pops)
                q_df["Membership_Consistency"] = mc
                q_df.to_csv(tsv_k, sep="\t", index=False)

        fig.suptitle("Assignment Proportions Across K", fontweight="bold", y=1.01)
        plt.tight_layout()
        if filename:
            fig.savefig(filename, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {filename} (+ per-K TSVs)")
        else:
            plt.show()

    # ------------------------------------------------------------------ #
    #  Plot: assignment proportions for selected K                          #
    # ------------------------------------------------------------------ #

    def plot_assignment(self, filename=None):
        """Assignment proportion plot for the selected K only."""
        resample_labels = self.results.get("resample_labels")
        if resample_labels is None:
            print("No CV label data available for assignment plot.")
            return

        k = self.results["fixed_k"]
        aligned, Q = multiway_align(resample_labels, k)
        mc = membership_consistency(Q)

        pops = np.array(self.genotype_data.populations)
        samples = np.array(self.genotype_data.samples)
        n = Q.shape[0]
        order = self._get_sample_order()
        Q_sorted = Q[order]
        pops_sorted = pops[order]
        dividers, centers = self._pop_dividers(pops_sorted)

        fig, ax = plt.subplots(figsize=(14, 4))
        cmap = plt.colormaps.get_cmap("tab10")
        bottom = np.zeros(n)
        for j in range(k):
            ax.bar(range(n), Q_sorted[:, j], bottom=bottom, width=1.0,
                   color=cmap(j), edgecolor='none', label=f"Cluster {j}")
            bottom += Q_sorted[:, j]

        for xpos in dividers:
            ax.axvline(xpos, color="black", linewidth=0.5)
        for center, label in centers:
            ax.text(center, -0.08, label, ha="center", va="top",
                    fontsize=6, rotation=45, transform=ax.get_xaxis_transform())

        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Assignment Proportion")
        ax.set_title(
            f"K={k}, nPC={self.results['n_pcs']}  "
            f"(mean membership consistency={np.mean(mc):.3f})"
        )
        ax.set_xticks([])
        ax.legend(loc="upper right", fontsize=7, ncol=k)
        plt.tight_layout()
        if filename:
            fig.savefig(filename, dpi=150, bbox_inches="tight")
            plt.close(fig)
            tsv_path = filename.rsplit(".", 1)[0] + ".tsv"
            q_df = pd.DataFrame(Q, columns=[f"Cluster_{j}" for j in range(k)])
            q_df.insert(0, "Sample", samples)
            q_df.insert(1, "Population", pops)
            q_df["Membership_Consistency"] = mc
            q_df.to_csv(tsv_path, sep="\t", index=False)
            print(f"Assignment table saved: {tsv_path}")
        else:
            plt.show()

    # ------------------------------------------------------------------ #
    #  Plot: PCA + DA for selected K only                                  #
    # ------------------------------------------------------------------ #

    def plot_embeddings(self, filename=None):
        """PCA + DA scatter for the selected K."""
        if not self.results or "fixed_k" not in self.results:
            print("No results available for plotting.")
            return
        T = self.results["T"]
        G = self.results["G"]
        coords = self.results["coords"]
        pops = self.genotype_data.populations
        name = self.solver_cls.__name__

        df_pca = pd.DataFrame(coords[:, :2], columns=["PC1", "PC2"])
        df_pca["Cluster"] = G
        df_pca["Population"] = pops
        n_da = self._n_da(self.results["fixed_k"], T)
        df_da = pd.DataFrame(
            T[:, :n_da], columns=[f"DA{i+1}" for i in range(n_da)]
        )
        df_da["Cluster"] = G
        df_da["Population"] = pops

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="Cluster",
                       style="Population", palette="deep", ax=axes[0], legend=False)
        axes[0].set_title(f"{name} K={self.results['fixed_k']} — PCA")
        if n_da >= 2:
            sns.scatterplot(data=df_da, x="DA1", y="DA2", hue="Cluster",
                           style="Population", palette="deep", ax=axes[1], legend=False)
        else:
            sns.kdeplot(x="DA1", data=df_da, hue="Cluster",
                       fill=None, common_norm=False, palette="deep", ax=axes[1])
            df_da["y"] = 0.1
            sns.scatterplot(data=df_da, x="DA1", y="y", hue="Cluster",
                           style="Population", palette="deep", ax=axes[1], legend=False)
        axes[1].set_title(f"{name} K={self.results['fixed_k']} — DA")
        plt.subplots_adjust(wspace=0.3)
        if filename:
            fig.savefig(filename, dpi=150, bbox_inches="tight")
            plt.close(fig)
            # TSV
            tsv = filename.rsplit(".", 1)[0] + ".tsv"
            out = pd.DataFrame({
                "Sample": self.genotype_data.samples,
                "Population": pops, "Cluster": G,
            })
            for i in range(min(n_da, 5)):
                out[f"DA{i+1}"] = T[:, i]
            out.to_csv(tsv, sep="\t", index=False)
        else:
            plt.show()

    # ------------------------------------------------------------------ #
    #  Discriminant rank                                                   #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _n_da(k, T):
        """Number of *meaningful* discriminant axes for a K-cluster fit.

        S_b is a sum of K centred cluster-mean outer products, so its rank is
        at most K-1. Solvers return a full-width W regardless, and the extra
        columns are arbitrary directions from the null space of S_b -- they
        carry no between-cluster signal. Plotting one as "DA2" invites the
        reader to interpret noise as structure, which is exactly wrong at
        K=2, where only a single discriminant axis exists.
        """
        return int(min(max(k - 1, 1), T.shape[1]))

    # ------------------------------------------------------------------ #
    #  Plot: stability heatmap                                                    #
    # ------------------------------------------------------------------ #

    def plot_stability_heatmap(self, filename=None, cmap="viridis", annot=True, fmt=".2f"):
        """Heatmap of stability across nPC x K grid."""
        if self.metrics is None:
            print("No metrics available."); return

        pivot = self.metrics.pivot(
            index="n_PCs", columns="K", values="Stability"
        ).sort_index()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(pivot, ax=ax, cmap=cmap, annot=annot, fmt=fmt,
                   cbar_kws={"label": f"Stability ({self.scoring})"})

        # Mark selected cell
        sel_k = self.results.get("fixed_k")
        sel_p = self.results.get("n_pcs")
        if sel_k in pivot.columns and sel_p in pivot.index:
            k_pos = list(pivot.columns).index(sel_k) + 0.5
            p_pos = list(pivot.index).index(sel_p) + 0.5
            ax.plot(k_pos, p_pos, marker="*", markersize=15, color="red",
                    markeredgecolor="white", markeredgewidth=1.5)

        title = f"Clustering Stability ({self.scoring})"
        if "SE" in self.metrics.columns:
            best_idx = self.metrics["Stability"].idxmax()
            best_se = self.metrics.loc[best_idx, "SE"]
            threshold = self.metrics.loc[best_idx, "Stability"] - best_se
            title += f"\n{self.selection_rule} threshold={threshold:.3f}"
        ax.set_xlabel("Clusters (K)")
        ax.set_ylabel("PCs")
        ax.set_title(title)
        plt.tight_layout()
        if filename:
            fig.savefig(filename, dpi=150, bbox_inches="tight")
            plt.close(fig)
            tsv = filename.rsplit(".", 1)[0] + ".tsv"
            self.metrics.to_csv(tsv, sep="\t", index=False)
            print(f"Saved: {filename}, {tsv}")
        else:
            plt.show()

    # ------------------------------------------------------------------ #
    #  Plot: stability curve                                               #
    # ------------------------------------------------------------------ #

    def plot_stability_curve(self, filename=None):
        """Stability vs K with error bars, threshold, and selected K."""
        if self.metrics is None or "SE" not in self.metrics.columns:
            print("No CV metrics available."); return

        fig, ax = plt.subplots(figsize=(8, 5))

        for npc, grp in self.metrics.groupby("n_PCs"):
            grp = grp.sort_values("K")
            ax.errorbar(grp["K"], grp["Stability"], yerr=grp["SE"],
                       marker="o", label=f"nPC={npc}", capsize=3)

        # Threshold
        best_idx = self.metrics["Stability"].idxmax()
        best_stab = self.metrics.loc[best_idx, "Stability"]
        best_se = self.metrics.loc[best_idx, "SE"]
        threshold = best_stab - best_se
        ax.axhline(threshold, color="red", linestyle="--", alpha=0.7,
                  label=f"1-SE threshold ({threshold:.3f})")

        # Selected
        sel_k = self.results.get("fixed_k")
        sel_p = self.results.get("n_pcs")
        if sel_k and sel_p:
            sel_row = self.metrics[
                (self.metrics["K"] == sel_k) & (self.metrics["n_PCs"] == sel_p)
            ]
            if not sel_row.empty:
                ax.plot(sel_k, sel_row["Stability"].values[0], marker="*",
                       markersize=15, color="red", zorder=5,
                       label=f"Selected K={sel_k}")

        ax.set_xlabel("K (number of clusters)")
        ax.set_ylabel(f"Stability ({self.scoring})")
        ax.set_title("Cluster Stability vs K")
        ax.legend()
        plt.tight_layout()
        if filename:
            fig.savefig(filename, dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

    # ------------------------------------------------------------------ #
    #  Plot: scree                                                         #
    # ------------------------------------------------------------------ #

    def plot_scree(self, max_n=None, filename=None):
        X = self._get_numeric_matrix()
        n_s, n_f = X.shape
        n = min(n_s, n_f) if max_n is None else min(max_n, n_s, n_f)
        pca = PCA(n_components=n)
        pca.fit(X)
        var = pca.explained_variance_ratio_
        cum = np.cumsum(var)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(1, n + 1), var, marker="o", label="Individual")
        ax.plot(range(1, n + 1), cum, marker="s", label="Cumulative")
        ax.set_xlabel("Number of PCs")
        ax.set_ylabel("Explained Variance")
        ax.set_title("Scree Plot")
        ax.legend()
        plt.tight_layout()
        if filename:
            fig.savefig(filename, dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()
