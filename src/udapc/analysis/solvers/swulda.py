# SWULDA (Self-Weighted Unsupervised LDA) - DISABLED
#
# This solver has a fundamental lambda feedback issue: when within-class
# variance exceeds between-class variance, lambda < 1 causes W to collapse
# to PCA directions. Multiple fix attempts (lambda clamp, whitened init/
# G-update, epsilon regularization) all failed on adversarial data.
# Keeping the code here for future reference but it should not be used.
#
# Reference: Li et al. 2023, IEEE TNNLS 34(3)

# import numpy as np
# from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans
#
#
# class SWULDA:
#     def __init__(self, c, Npc, Ninit=10, tol=1e-6, max_iter=100, center=True,
#                  no_pca=False, lambda_min=1.0, **kwargs):
#         self.c = c
#         self.Npc = Npc
#         self.Ninit = Ninit
#         self.tol = tol
#         self.max_iter = max_iter
#         self.center = center
#         self.no_pca = no_pca
#         self.lambda_min = lambda_min
#
#         self.W = None
#         self.Lambda = None
#         self.train_mean = None
#         self.obj_log = []
#         self.T_train = None
#         self.G_train = None
#
#     def fit(self, X):
#         self.train_mean = np.mean(X, axis=0)
#         X = X - self.train_mean if self.center else X
#         n, d = X.shape
#         m = self.Npc
#
#         pca = PCA(n_components=m)
#         pca.fit(X)
#         W = pca.components_.T
#
#         T_init = X @ W
#         stds = T_init.std(axis=0)
#         stds[stds < 1e-12] = 1.0
#         kmeans = KMeans(n_clusters=self.c, n_init=self.Ninit)
#         init_labels = kmeans.fit_predict(T_init / stds)
#         G = np.zeros((n, self.c))
#         G[np.arange(n), init_labels] = 1
#
#         F_init = W.T @ X.T @ G @ np.linalg.pinv(G.T @ G)
#         FGt_init = F_init @ G.T
#         num_init = np.trace(W.T @ X.T @ X @ W)
#         denom_init = 2 * np.linalg.norm(W.T @ X.T - FGt_init, 'fro')**2
#         Lambda = max(num_init / max(denom_init, 1e-12), self.lambda_min)
#
#         obj_old = -np.inf
#         obj_new = 0.0
#         obj_log = []
#         it = 0
#
#         while (not np.isclose(obj_old, obj_new, atol=self.tol) or it == 0) and it < self.max_iter:
#             it += 1
#             obj_old = obj_new
#
#             eps = 1e-4
#             A = (Lambda**2 - Lambda + eps) * np.eye(n)
#             B = Lambda**2 * G @ np.linalg.pinv(G.T @ G) @ G.T
#             M = X.T @ (A - B) @ X
#
#             eigvals, eigvecs = np.linalg.eigh(M)
#             W = eigvecs[:, :m]
#
#             F = W.T @ X.T @ G @ np.linalg.pinv(G.T @ G)
#             FGt = F @ G.T
#
#             T_all = (W.T @ X.T).T
#             stds = T_all.std(axis=0)
#             stds[stds < 1e-12] = 1.0
#             T_w = T_all / stds
#             F_w = F / stds.reshape(-1, 1)
#             dists = np.linalg.norm(T_w[:, :, np.newaxis] - F_w[np.newaxis, :, :],
#                                    axis=1)
#             G = np.zeros((n, self.c))
#             G[np.arange(n), np.argmin(dists, axis=1)] = 1
#
#             num = np.trace(W.T @ X.T @ X @ W)
#             denom = 2 * np.linalg.norm(W.T @ X.T - FGt, 'fro')**2
#             Lambda = max(num / denom, self.lambda_min)
#
#             obj_new = Lambda**2 * np.linalg.norm(W.T @ X.T - FGt, 'fro')**2 - Lambda * num
#             obj_log.append(obj_new)
#
#         if it == self.max_iter:
#             print(f"Warning: SWULDA did not converge within {self.max_iter} iterations.")
#
#         T = X @ W
#         Ypre = np.argmax(G, axis=1)
#
#         self.W = W
#         self.Lambda = Lambda
#         self.T_train = T
#         self.G_train = Ypre
#         self.obj_log = obj_log
#         return T, Ypre, W, obj_log
#
#     def transform(self, X):
#         if self.W is None:
#             raise RuntimeError("Model has not been fit yet.")
#         X_centered = X - self.train_mean if self.center else X
#         return X_centered @ self.W
#
#     def predict(self, X):
#         T = self.transform(X)
#         return KMeans(n_clusters=self.c, n_init=self.Ninit).fit_predict(T)
