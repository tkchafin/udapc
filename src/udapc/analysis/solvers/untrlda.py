# udapc/analysis/solvers/untrlda.py
#
# Implements Un-TRLDA (Unsupervised Trace Ratio LDA) from:
#   Wang et al., "Unsupervised Linear Discriminant Analysis for Jointly
#   Clustering and Subspace Learning", IEEE TKDE 33(3), 2021.
#   Algorithm 4 (Un-TRLDA.v2).

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from udapc.utils.linalg import trace_ratio


def _mat_pow_neg_half(A):
    """Compute A^{-1/2} for symmetric positive-definite A via eigendecomposition."""
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.maximum(eigvals, 1e-12)
    return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T


class UnTRLDA:
    def __init__(self, c, Npc, Ninit=10, tol=1e-6, max_iter=100, Ntry=10,
                 center=True, no_pca=False, gamma=1e-6, **kwargs):
        self.c = c
        self.Npc = Npc
        self.Ninit = Ninit
        self.tol = tol
        self.max_iter = max_iter
        self.Ntry = Ntry
        self.center = center
        self.no_pca = no_pca
        self.gamma = gamma

        self.W2 = None
        self.St = None
        self.train_mean = None
        self.obj_log = []
        self.T_train = None
        self.G_train = None

    def _center_matrix(self, n):
        return np.eye(n) - np.ones((n, n)) / n if self.center else np.eye(n)

    def fit(self, X):
        """
        Train the Un-TRLDA model (Algorithm 4 from Wang et al. 2021).

        Key equations (paper numbers):
          T = W^T XH                                            (Alg. 3/4, line 6)
          TRLDA: max Tr(W^T Sb W)/Tr(W^T St W), s.t. W^T W=I  (Eq. 9)
          obj = Tr(W^T Sb W) / Tr(W^T St W)                    (Eq. 15)
        """
        n, d = X.shape
        self.train_mean = X.mean(axis=0)
        H = self._center_matrix(n)
        St = X.T @ H @ X
        self.St = St

        it = 0
        obj_old = -np.inf
        obj_new = 0.0

        # Paper: W^(0) ← PCA(XH)
        pca = PCA(n_components=min(self.Npc, d))
        pca.fit(X)
        W2 = pca.components_.T

        while (not np.isclose(obj_old, obj_new, atol=self.tol) or it == 0) and it < self.max_iter:
            it += 1
            obj_old = obj_new

            # Whitened projection for K-means: T = (W^T St W)^{-1/2} W^T XH
            # Without whitening, PCA initialization creates a fixed point where
            # K-means clusters stay PCA-aligned → Sb stays PCA-aligned → W stays PCA.
            # Whitening makes projected data isotropic, breaking this feedback loop.
            A = W2.T @ St @ W2
            A_reg = A + self.gamma * np.eye(A.shape[0])
            A_inv_sqrt = _mat_pow_neg_half(A_reg)
            T = (A_inv_sqrt @ W2.T @ X.T @ H).T

            best_inertia = float("inf")
            best_labels = None
            for _ in range(self.Ntry):
                kmeans = KMeans(n_clusters=self.c, n_init=self.Ninit,
                                max_iter=self.max_iter, tol=self.tol)
                labels = kmeans.fit_predict(T)
                if kmeans.inertia_ < best_inertia:
                    best_inertia = kmeans.inertia_
                    best_labels = labels

            Ypre = best_labels
            Yp = np.eye(self.c)[Ypre]

            # Paper: TRLDA — max Tr(W^T Sb W)/Tr(W^T St W), s.t. W^T W = I
            Sb = X.T @ H @ Yp @ np.linalg.inv(Yp.T @ Yp) @ Yp.T @ H @ X
            W2, obj_new = trace_ratio(Sb, St, self.Npc)
            self.obj_log.append(obj_new)

        if it == self.max_iter:
            print(f"Warning: UnTRLDA did not converge within {self.max_iter} iterations.")

        # Unwhitened output projection — preserves eigenvalue ordering so
        # first columns are the most discriminant directions for plotting.
        # (Whitening is only used inside the loop to help K-means.)
        T = (W2.T @ X.T @ H).T

        self.W2 = W2
        self.T_train = T
        self.G_train = Ypre
        return T, Ypre, W2, self.obj_log

    def transform(self, X):
        if self.W2 is None:
            raise RuntimeError("Model has not been fit yet.")
        X_centered = X - self.train_mean
        return X_centered @ self.W2

    def predict(self, X):
        T = self.transform(X)
        return KMeans(n_clusters=self.c, n_init=self.Ninit).fit_predict(T)
