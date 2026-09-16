import numpy as np

def choose_npcs_by_variance(pca, prop_pc_var):
    """Return number of PCs required to capture at least `prop_pc_var` variance."""
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    return np.searchsorted(cum_var, prop_pc_var) + 1

def n_significant_pcs_mp(X, center=True, scale=True):
    """Number of PCs whose eigenvalue exceeds the Marchenko-Pastur noise edge.

    For an n x p matrix of independent noise, the eigenvalues of the sample
    covariance fall below (1 + sqrt(n/p))^2. Eigenvalues above that edge carry
    signal rather than noise (Patterson, Price & Reich 2006, PLoS Genetics).

    This is a property of the genotype matrix alone -- it involves no clustering
    and therefore no K, so it yields a single nPC that cannot vary across K.

    Assumes loci are independent; LD inflates the count, so prune first on
    empirical data.
    """
    Z = np.asarray(X, dtype=float)
    n, p = Z.shape
    if center:
        Z = Z - Z.mean(axis=0)
    if scale:
        sd = Z.std(axis=0)
        sd[sd == 0] = 1.0
        Z = Z / sd
    ev = np.linalg.svd(Z, compute_uv=False) ** 2 / p
    edge = (1.0 + np.sqrt(n / p)) ** 2
    return int(max(1, (ev > edge).sum()))
