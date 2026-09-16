# udapc/analysis/solvers/ldakm.py
#
# Implements LDA-Km (LDA + K-means) from:
#   Wang et al., "Unsupervised Linear Discriminant Analysis for Jointly
#   Clustering and Subspace Learning", IEEE TKDE 33(3), 2021.
#   Algorithm 6 (LDA-Km.v2), originally from Ding & Li (2007).
#
# Key equations:
#   GEVD: Sb W = (Sw + γI) W Λ                                (Eq. 23)
#   obj = Tr((W^T (Sw + γI) W)^{-1} W^T Sb W)                (Eq. 24)

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from udapc.utils.linalg import gevd


class LDAKm:
    def __init__(self, c, Npc, Ninit=10, gamma=1e-6, tol=1e-6,
                 max_iter=100, Ntry=10, center=True, no_pca=False, **kwargs):
        self.c = c
        self.Npc = Npc
        self.Ninit = Ninit
        self.gamma = gamma
        self.tol = tol
        self.max_iter = max_iter
        self.Ntry = Ntry
        self.center = center
        self.no_pca = no_pca

        self.W2 = None
        self.train_mean = None
        self.obj_log = []
        self.T_train = None
        self.G_train = None

    def _center_matrix(self, n):
        return np.eye(n) - np.ones((n, n)) / n if self.center else np.eye(n)

    def fit(self, X):
        n, d = X.shape
        self.train_mean = X.mean(axis=0)
        H = self._center_matrix(n)
        St = X.T @ H @ X

        it = 0
        obj_old = -np.inf
        obj_new = 0.0
        m = self.Npc

        # Paper: W^(0) ← PCA(XH)
        if self.no_pca:
            pca = PCA(n_components=min(m, d))
            pca.fit(X)
            W2 = pca.components_.T
        else:
            pca = PCA(n_components=min(m, d))
            pca.fit(X)
            W2 = pca.components_.T

        obj_log = []

        while (not np.isclose(obj_old, obj_new, atol=self.tol) or it == 0) and it < self.max_iter:
            it += 1
            obj_old = obj_new

            # Paper line 6: T ← W^T XH
            T = (W2.T @ X.T @ H).T

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

            Sb = X.T @ H @ Yp @ np.linalg.inv(Yp.T @ Yp) @ Yp.T @ H @ X
            # Paper Eq. 23: Sb W = (Sw + γI) W Λ
            Sw_reg = St - Sb + self.gamma * np.eye(d)

            model = gevd(Sb, Sw_reg, is_max=True)
            W2 = model['W'][:, :m]

            # Paper Eq. 24: obj = Tr((W^T (Sw+γI) W)^{-1} W^T Sb W)
            obj_new = np.trace(np.linalg.pinv(W2.T @ Sw_reg @ W2) @ W2.T @ Sb @ W2)
            obj_log.append(obj_new)

        if it == self.max_iter:
            print(f"Warning: LDAKm did not converge within {self.max_iter} iterations.")

        self.W2 = W2
        self.T_train = T
        self.G_train = Ypre
        self.obj_log = obj_log
        return T, Ypre, W2, obj_log

    def transform(self, X):
        if self.W2 is None:
            raise RuntimeError("Model has not been fit yet.")
        X_centered = X - self.train_mean
        return X_centered @ self.W2

    def predict(self, X):
        T = self.transform(X)
        return KMeans(n_clusters=self.c, n_init=self.Ninit).fit_predict(T)
