#!/usr/bin/env python3

import click
import snpio
from udapc.analysis.unsupervised import UnsupervisedDiscriminantAnalysis


@click.group()
def cli():
    """Command-line interface for UDAPC."""
    pass


@cli.command()
@click.option("--vcf", type=click.Path(exists=True), required=True,
              help="Path to the VCF file")
@click.option("--popmap", type=click.Path(exists=True), required=True,
              help="Path to the population map file")
@click.option("--prefix", type=str, default="udapc_output",
              help="Output prefix for all files")
@click.option("--max_k", type=int, default=8,
              help="Maximum number of clusters to test")
@click.option("--fixed_k", type=int, default=None,
              help="Fixed number of clusters (skip K search)")
@click.option("--n_pca", type=int, default=None,
              help="Fixed number of PCs (skip nPC search)")
@click.option("--impute", type=click.Choice(["mode", "mean", "pop_mode"]),
              default="mode",
              help="Missing-genotype imputation. mode/mean use the genotype "
                   "matrix alone; pop_mode imputes within popmap groups, which "
                   "lets the labels influence the clustering.")
@click.option("--n_pca_mp", is_flag=True, default=False,
              help="Fix nPC at the number of PCs exceeding the Marchenko-Pastur "
                   "noise edge (Patterson et al. 2006). K is still searched.")
@click.option("--n_pca_var", type=float, default=None,
              help="Fix nPC at the number of PCs capturing this proportion of "
                   "variance, e.g. 0.5. K is still searched.")
@click.option("--n_pca_min", type=int, default=10,
              help="Minimum PCs for range search")
@click.option("--n_pca_max", type=int, default=200,
              help="Maximum PCs for range search")
@click.option("--n_pca_step", type=int, default=10,
              help="Step size for PC range search")
@click.option("--subset", type=float, default=1.0,
              help="Thin each training draw to this fraction of its individuals "
                   "(1.0 = use the whole draw). Lowering it widens the resampling.")
@click.option("--n_folds", type=int, default=5,
              help="Folds the sample is partitioned into per repeat; each draw trains on (n_folds-1)/n_folds of individuals")
@click.option("--n_reps", type=int, default=10,
              help="Times the partition is redrawn. Total draws = n_folds x n_reps")
@click.option("--solver", type=click.Choice(["unrtlda", "untrlda", "ldakm"]),
              default="unrtlda", help="Solver algorithm")
@click.option("--scoring", type=click.Choice(["ari", "membership"]),
              default="membership",
              help="Scoring metric: ari (pairwise ARI) or membership (mean consistency)")
@click.option("--selection_rule", type=click.Choice(["1se", "max"]),
              default="max",
              help="Model selection rule: 1se (most parsimonious within one "
                   "standard error of the best) or max (highest scoring)")
@click.option("--unlinked_npc", is_flag=True, default=False,
              help="Choose nPC separately for each K. Default is one nPC shared "
                   "across all K, which benchmarked marginally better. Has no "
                   "effect when nPC is fixed (--n_pca, --n_pca_mp, --n_pca_var): "
                   "those derive a single nPC from the data before any "
                   "clustering, so there is no K for it to vary over.")
@click.option("--null_correction/--no_null_correction", default=False,
              help="Score cells by the gap statistic: CV stability minus the "
                   "stability obtained on structure-free data (genotypes permuted "
                   "within locus). Raw stability is maximised by the smallest K and "
                   "nPC regardless of the data, so without this the search returns "
                   "whatever grid floor it is given. Roughly doubles runtime.")
@click.option("--n_null", type=int, default=1,
              help="Permutation references averaged for the null (higher is a "
                   "steadier null, at proportional cost)")
def run(vcf, popmap, prefix, max_k, fixed_k, n_pca, n_pca_mp, n_pca_var,
        n_pca_min, n_pca_max, impute, subset,
        n_pca_step, n_folds, n_reps, solver, scoring, selection_rule,
        unlinked_npc, null_correction, n_null):
    """
    Run unsupervised discriminant analysis of principal components (UDAPC).

    Selects K and the number of PCs by resampling stability, then produces
    diagnostic plots and tables.
    """
    gd = snpio.VCFReader(
        filename=vcf,
        popmapfile=popmap,
        force_popmap=True,
        verbose=True,
        prefix="snpio_udapc",
    )

    udapc = UnsupervisedDiscriminantAnalysis(
        genotype_data=gd,
        reps=n_reps,
        n_folds=n_folds,
        subset=subset,
        n_pca=n_pca,
        impute=impute,
        n_pca_mp=n_pca_mp,
        n_pca_var=n_pca_var,
        n_pca_min=n_pca_min,
        n_pca_max=n_pca_max,
        n_pca_step=n_pca_step,
        max_k=max_k,
        fixed_k=fixed_k,
        solver=solver,
        selection_rule=selection_rule,
        scoring=scoring,
        linked_npc=not unlinked_npc,
        null_correction=null_correction,
        n_null=n_null,
        solver_kwargs={
            "Ninit": 10,
            "max_iter": 300,
            "gamma": 1e-6,
            "tol": 1e-6,
        },
    )

    udapc.fit_predict()

    # Always output
    udapc.plot_embeddings(f"{prefix}_embeddings.pdf")
    udapc.plot_all_k(f"{prefix}_all_k.pdf")

    # Assignment proportion plots (need CV data)
    udapc.plot_assignment(f"{prefix}_assignment.pdf")
    udapc.plot_all_assignments(f"{prefix}_assignments_all_k.pdf")

    # Diagnostics (need grid search)
    if not (fixed_k and n_pca):
        udapc.plot_stability_heatmap(f"{prefix}_heatmap.pdf")
        udapc.plot_stability_curve(f"{prefix}_stability.pdf")

    # Summary
    print("\n" + "=" * 60)
    print("Stability summary:")
    print("=" * 60)
    print(udapc.score().to_string(index=False))
    print(f"\nSelected: K={udapc.results['fixed_k']}, nPC={udapc.results['n_pcs']}")
    if null_correction:
        m = udapc.metrics
        sel = m[(m.K == udapc.results["fixed_k"]) &
                (m.n_PCs == udapc.results["n_pcs"])]
        if len(sel):
            r = sel.iloc[0]
            print(f"  gap={r['Stability']:.4f}  (raw stability={r['Raw']:.4f}, "
                  f"null={r['Null']:.4f})")


if __name__ == "__main__":
    cli()
