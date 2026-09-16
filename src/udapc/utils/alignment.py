"""
Label alignment across multiple clustering runs.

Two strategies:
1. Hungarian (pairwise): align each run to a reference via optimal bipartite matching
2. Exhaustive: try all K! permutations per run, maximizing global assignment consistency

For K <= 8, exhaustive is feasible. For K > 8, fall back to Hungarian.
"""

import numpy as np
from itertools import permutations
from scipy.optimize import linear_sum_assignment


def hungarian_align(ref_labels, pred_labels, k):
    """Align pred_labels to ref_labels via Hungarian algorithm on confusion matrix."""
    cost = np.zeros((k, k), dtype=int)
    for i in range(k):
        for j in range(k):
            cost[i, j] = np.sum((ref_labels == i) & (pred_labels == j))
    r, c = linear_sum_assignment(-cost)
    mapping = {col: row for row, col in zip(r, c)}
    return np.array([mapping.get(l, l) for l in pred_labels])


def _assignment_props(label_matrix, k):
    """Compute per-sample assignment proportions (Q-matrix).

    Args:
        label_matrix: (n_samples, n_runs) integer labels
        k: number of clusters

    Returns:
        Q: (n_samples, k) matrix where Q[i, j] = fraction of runs
           assigning sample i to cluster j
    """
    n_samples, n_runs = label_matrix.shape
    Q = np.zeros((n_samples, k))
    for j in range(k):
        Q[:, j] = np.mean(label_matrix == j, axis=1)
    return Q


def _mean_max_assignment(label_matrix, k):
    """Mean of per-sample max assignment proportion.

    This is the objective maximized by exhaustive alignment:
    higher = more consistent assignments across runs.
    """
    Q = _assignment_props(label_matrix, k)
    return float(np.mean(np.max(Q, axis=1)))


def exhaustive_align(label_matrix, k, max_passes=10):
    """Align all runs by exhaustively trying all K! label permutations per run.

    Strategy: Hungarian initialization + iterative exhaustive refinement.
    1. First align all runs to the best reference using Hungarian
    2. Then iteratively sweep all runs, trying all K! perms for each,
       until convergence (no improvement in a full pass).

    Args:
        label_matrix: (n_samples, n_runs) integer labels
        k: number of clusters
        max_passes: maximum number of refinement sweeps

    Returns:
        aligned: (n_samples, n_runs) aligned label matrix
    """
    n_samples, n_runs = label_matrix.shape
    unique_labels = list(range(k))
    all_perms = list(permutations(unique_labels))

    # Step 1: Hungarian initialization — align all to first run
    aligned = label_matrix.copy()
    ref = aligned[:, 0]
    for col in range(1, n_runs):
        aligned[:, col] = hungarian_align(ref, label_matrix[:, col], k)

    # Step 2: Iterative exhaustive refinement
    current_score = _mean_max_assignment(aligned, k)

    for pass_num in range(max_passes):
        improved = False
        for col in range(n_runs):
            best_score = current_score
            best_col = aligned[:, col].copy()

            for perm in all_perms:
                mapping = {orig: new for orig, new in zip(unique_labels, perm)}
                remapped = np.array([mapping[l] for l in label_matrix[:, col]])

                test = aligned.copy()
                test[:, col] = remapped
                score = _mean_max_assignment(test, k)

                if score > best_score + 1e-10:
                    best_score = score
                    best_col = remapped

            if best_score > current_score + 1e-10:
                aligned[:, col] = best_col
                current_score = best_score
                improved = True

        if not improved:
            break

    return aligned


def multiway_align(label_matrix, k, method="auto"):
    """Align labels across runs using the best available method.

    Args:
        label_matrix: (n_samples, n_runs) integer labels
        k: number of clusters
        method: "exhaustive", "hungarian", or "auto" (exhaustive if K<=8)

    Returns:
        aligned: (n_samples, n_runs) aligned label matrix
        Q: (n_samples, k) assignment proportion matrix
    """
    if method == "auto":
        method = "exhaustive" if k <= 8 else "hungarian"

    if method == "exhaustive":
        aligned = exhaustive_align(label_matrix, k)
    else:
        # Hungarian: align all to first run
        aligned = label_matrix.copy()
        ref = aligned[:, 0]
        for col in range(1, label_matrix.shape[1]):
            aligned[:, col] = hungarian_align(ref, label_matrix[:, col], k)

    Q = _assignment_props(aligned, k)
    return aligned, Q


def membership_consistency(Q):
    """Per-sample consistency: max assignment proportion.

    Returns array of length n_samples, each in [1/K, 1.0].
    1.0 = always assigned to same cluster.
    1/K = uniformly distributed across clusters.
    """
    return np.max(Q, axis=1)
