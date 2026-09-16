# udapc/utils/linalg.py

import numpy as np
from scipy.linalg import eigh

def gevd(A, B, is_max=True):
    """
    Generalized eigendecomposition of two symmetric matrices A and B.

    Args:
        A (np.ndarray): Square symmetric matrix (e.g., between-class scatter).
        B (np.ndarray): Square symmetric matrix (e.g., total or within-class scatter).
        is_max (bool): Whether to sort eigenvalues in descending order (default True).

    Returns:
        dict: {
            'W': Eigenvectors sorted by eigenvalue,
            'D': Diagonal matrix of sorted eigenvalues
        }
    """
    eigvals, eigvecs = eigh(A, B)
    idx = np.argsort(eigvals)[::-1] if is_max else np.argsort(eigvals)
    eigvals_sorted = eigvals[idx]
    eigvecs_sorted = eigvecs[:, idx]
    return {
        "W": eigvecs_sorted,
        "D": np.diag(eigvals_sorted)
    }

def trace_ratio(A, B, dim, is_max=True, tol=1e-6, max_iter=20):
    """
    Solve the Trace Ratio problem: max_{W'*W=I} tr(W'*A*W)/tr(W'*B*W)

    Args:
        A (np.ndarray): Between-class scatter matrix (symmetric).
        B (np.ndarray): Total scatter matrix (positive semi-definite).
        dim (int): Number of components.
        is_max (bool): Maximize the ratio if True, else minimize.
        tol (float): Convergence tolerance.
        max_iter (int): Max number of iterations.

    Returns:
        W (np.ndarray): Projection matrix.
        obj (float): Final trace ratio value.
    """
    n = A.shape[0]
    W = np.eye(n, dim)
    ob = np.trace(W.T @ A @ W) / np.trace(W.T @ B @ W)

    for _ in range(max_iter):
        M = A - ob * B
        M = (M + M.T) / 2
        eigvals, eigvecs = np.linalg.eigh(M)
        idx = np.argsort(eigvals)[::-1 if is_max else 1]
        W = eigvecs[:, idx[:dim]]
        ob_new = np.trace(W.T @ A @ W) / np.trace(W.T @ B @ W)
        if np.abs(ob - ob_new) < tol:
            return W, ob_new
        ob = ob_new

    print("Warning: Trace ratio did not converge")
    return W, ob
