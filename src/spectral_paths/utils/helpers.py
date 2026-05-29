"""This module contains helper functions for the spectral path models."""

import math
from typing import List, Sequence, Tuple

import numpy as np
from numba import njit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split as tts

from spectral_paths.types import PR, Array, MVec


def train_test_split(
    X: Array, y: Array, test_size: float, random_state: int = 42
) -> Tuple[Array, Array, Array, Array]:
    """Sklearn warpper."""
    split = tts(X, y, test_size=test_size, random_state=random_state)
    assert len(split) == 4, "Expected sklearn train_test_split to return 4 arrays."
    X_tr, X_test, y_tr, y_test = split
    return (
        np.asarray(X_tr),
        np.asarray(X_test),
        np.asarray(y_tr),
        np.asarray(y_test),
    )


def check_dimensions(X: Array, y: Array) -> None:
    """Helper function. Ensures X is 2D and dimensiosn of X and y are equal."""
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X rows {X.shape[0]} != y rows {y.shape[0]}")

@njit(cache=True)
def _cos_chebyshev(phi: float, r: int) -> float:
    """
    Compute T_r(cos(phi)) using the Chebyshev recurrence.

    Args:
        phi (float): Angular coordinate in radians.
        r (int): Non-negative Chebyshev order.

    Returns:
        The Chebyshev polynomial of the first kind evaluated at cos(phi).
    """
    if r == 0:
        return 1.0
    cos_x = math.cos(phi)
    if r == 1:
        return cos_x
    t_nm2 = 1.0  # T_0
    t_nm1 = cos_x  # T_1
    for _ in range(2, r + 1):
        t_n = 2.0 * cos_x * t_nm1 - t_nm2  # T_n recurrence
        t_nm2, t_nm1 = t_nm1, t_n
    return t_nm1


@njit(cache=True)
def _build_feature_matrix(
    theta_batch: Array, path_matrix: Array, r_arr: Array, include_intercept: bool = True
) -> Array:
    """
    Build directional cosine-harmonic features for a batch of inputs.

    For each sample `theta` and each path/order pair `(p_q, r_q)`, computes:
        cos(r_q * <theta, p_q>)

    using the identity `T_r(cos(phi)) = cos(r * phi)`, where `phi = <theta, p_q>`.
    Optionally prepends an intercept column of ones.

    Args:
        theta_batch: Input angles, shape `(B, D)`.
        path_matrix: Direction vectors, shape `(Q, D)`.
        r_arr: Harmonic orders, shape `(Q,)`.
        include_intercept: If `True`, output column 0 is all ones.

    Returns:
        Feature matrix of shape `(B, Q + 1)` if `include_intercept`, else `(B, Q)`.
    """
    B = theta_batch.shape[0]
    Q = path_matrix.shape[0]
    M = Q + (1 if include_intercept else 0)
    out = np.empty((B, M), dtype=theta_batch.dtype)

    start_col = 0
    if include_intercept:
        out[:, 0] = 1.0
        start_col = 1

    phases = theta_batch @ path_matrix.T
    out[:, start_col:] = np.cos(phases * r_arr[None, :])

    return out

def _compute_initial_importance(theta_tr: Array, y_tr: Array) -> Array:
        """
        Compute a heuristic feature-importance prior for search ordering.

        Args:
            theta_tr (Array): Training inputs in angular coordinates with shape (N, D).
            y_tr(Array): Training targets with shape (N,)
            D (int): Number of features.

        Returns:
            Non-negative importance weights of length D. If any signal is present, the
            weights are normalised to sum to one; otherwise all entries are zero.
        """
        D = theta_tr.shape[1]
        importance = np.zeros(D, dtype=float)

        for j in range(D):
            # Compute correlation between cos(theta_j) and y
            cos_theta_j = np.cos(theta_tr[:, j])
            corr = np.abs(np.corrcoef(cos_theta_j, y_tr)[0, 1])
            importance[j] = corr if not np.isnan(corr) else 0.0

        # Normalize
        if importance.sum() > 0:
            importance = importance / importance.sum()

        return importance

def _gcd_list(vals: Sequence[int]) -> int:
    """
    Compute the greatest common divisor of a sequence of integers.

    Args:
        vals (Sequence[int]): Sequence of integer values.

    Returns:
        int: Greatest common divisor of the absolute values in the sequence.
            Returns 0 if all values are zero.
    """
    g = 0
    for v in vals:
        g = math.gcd(g, int(abs(v)))
    return g

def _primitive_and_order(path: MVec) -> PR:
    """
    Decompose a frequency vector into a primitive direction and harmonic order.

    Example: (2,4,6,0,0) -> ((1,2,3,0,0), 2)

    Args:
        path (Tuple[int, ...]): A spectral path.

    Returns:
        Tuple[Tuple[int, ...], int]: A tuple (p, r) where p is the primitive direction
        vector and r is the positive integer harmonic order such that m = r * p.
    """
    nonzeros = [v for v in path if v != 0]
    g = _gcd_list(nonzeros) if nonzeros else 1
    if g == 0:
        g = 1
    p = tuple((vi // g) for vi in path)
    r = g
    return p, r

def _group_by_primitive(paths: Sequence[MVec]) -> List[PR]:
    """
    Group frequency vectors by primitive direction.

    Args:
        paths (Sequence[Tuple[int, ...]]): The chosen spectral paths.

    Returns:
        result (List[Tuple[int, ...], int]): A list of (primitive, order) pairs
        corresponding to the input vectors, and a mapping from each primitive
        direction to the maximum harmonic order observed along that ray.
    """
    pr_list = []
    for path in paths:
        primitive_path, order = _primitive_and_order(path)
        pr_list.append((primitive_path, order))
    return pr_list

def _pr_list_to_arrays(pr_list: List[PR]) -> Tuple[Array, Array]:
    """
    Convert a list of primitive-ray specifications into array form.

    Args:
        pr_list (List[PR]): List of (primitive direction, harmonic order) pairs.

    Returns:
        Tuple[Array, Array]: A pair (path_matrix, r_arr) where path_matrix is an integer
        array of primitive directions with shape (Q, D) and r_arr contains
        the corresponding harmonic orders.
    """
    path_matrix = np.asarray([path for path, _ in pr_list], dtype=np.float64)
    r_arr = np.asarray([order for _, order in pr_list], dtype=np.int64)
    return path_matrix, r_arr

def _path_matrix_and_r_arr(paths: Sequence[MVec]) -> Tuple[Array, Array]:
    """."""
    pr_list = _group_by_primitive(paths)
    path_matrix, r_arr = _pr_list_to_arrays(pr_list)
    return path_matrix, r_arr

def _metrics(y_true: Array, y_pred: Array) -> Tuple[float, float]:
    """
    Compute regression performance metrics.

    Args:
        y_true (Array): Ground-truth target values.
        y_pred (Array): Predicted target values.

    Returns:
        Tuple[float, float]: Root mean squared error (RMSE) and coefficient of
        determination (R²).
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return rmse, r2


def _sigmoid(logits: Array) -> Array:
    """Compute the logistic sigmoid in a numerically stable way."""
    logits = np.asarray(logits, dtype=float)
    out = np.empty_like(logits, dtype=float)
    positive = logits >= 0.0
    negative = ~positive
    out[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_logits = np.exp(logits[negative])
    out[negative] = exp_logits / (1.0 + exp_logits)
    return out


def _clip_probabilities(probabilities: Array, eps: float = 1e-12) -> Array:
    """Clip probabilities away from 0 and 1 for stable log-loss calculations."""
    return np.clip(np.asarray(probabilities, dtype=float), eps, 1.0 - eps)


def _binary_log_loss(y_true: Array, y_prob: Array, eps: float = 1e-12) -> float:
    """Compute binary cross-entropy from labels and positive-class probabilities."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_prob = _clip_probabilities(y_prob, eps=eps).ravel()
    return float(-np.mean(y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob)))


def _binary_accuracy(y_true: Array, y_prob: Array, threshold: float = 0.5) -> float:
    """Compute binary classification accuracy from probabilities."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = (np.asarray(y_prob, dtype=float).ravel() >= threshold).astype(float)
    return float(np.mean(y_pred == y_true))
