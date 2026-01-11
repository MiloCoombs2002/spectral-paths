import math
from typing import Sequence, Tuple, List, Dict
from numba import njit, prange
import numpy as np

from sklearn.metrics import r2_score, mean_squared_error

from spectral_paths.types import Array, MVec, PR, PVec

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


@njit(parallel=True, cache=True)
def _build_features_numba(
    theta_batch: Array,
    p_mat: Array,
    r_arr: Array,
    include_intercept: bool
    ) -> Array:
    """
    Construct spectral path features using directional Chebyshev harmonics.

    For each sample θ ∈ R^D and each primitive direction p_q with harmonic
    order r_q, this function computes the feature
        cos(r_q * <p_q, θ>)
    using a stable Chebyshev recurrence. Features are evaluated in two stages:
    (i) directional phases <p_q, θ> are formed via dot products, and
    (ii) Chebyshev harmonics are applied along each primitive ray.

    The computation is fully parallelised over samples and features and is
    designed to be used inside streamed Gram-matrix construction.

    Args:
        theta_batch (Array): Array of angular inputs with shape (B, D),
            where B is the batch size and D the number of features.
        p_mat (Array): Matrix of primitive direction vectors with shape (Q, D).
        r_arr (Array): Array of non-negative harmonic orders of length Q.
        include_intercept (bool): Whether to prepend a constant intercept
            feature equal to 1.

    Returns:
        Array: Feature matrix of shape (B, Q + 1) if include_intercept is True,
        otherwise shape (B, Q). Each column corresponds to a directional
        Chebyshev feature evaluated at the batch inputs.
    """
    B, D = theta_batch.shape
    Q = p_mat.shape[0]
    M = Q + (1 if include_intercept else 0)
    out = np.empty((B, M), dtype=theta_batch.dtype)

    # Intercept first to keep parity with previous layout
    start_col = 0
    if include_intercept:
        out[:, 0] = 1.0
        start_col = 1

    # Compute phases in parallel over samples
    phases = np.zeros((B, Q), dtype=theta_batch.dtype)
    for i in prange(B):
        for q in range(Q):
            phi = 0.0
            for d in range(D):
                phi += theta_batch[i, d] * p_mat[q, d]
            phases[i, q] = phi

    # Apply Chebyshev transform in parallel over features
    for q in prange(Q):
        r = int(r_arr[q])
        for i in range(B):
            out[i, start_col + q] = _cos_chebyshev(phases[i, q], r)

    return out

def _compute_initial_importance(theta_tr: Array, y_tr: Array, D: int) -> Array:
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

def _primitive_and_order(m: MVec) -> PR:
    """
    Decompose a frequency vector into a primitive direction and harmonic order.

    Args:
        m (MVec): Integer frequency vector.

    Returns:
        PR: A tuple (p, r) where p is the primitive direction vector and
        r is the positive integer harmonic order such that m = r * p.
    """
    nonzeros = [v for v in m if v != 0]
    g = _gcd_list(nonzeros) if nonzeros else 1
    if g == 0:
        g = 1
    p = tuple((vi // g) for vi in m)
    r = g
    return p, r

def _group_by_primitive(
        indices_nonzero: Sequence[MVec]
    ) -> Tuple[List[PR], Dict[PVec, int]]:
    """
    Group frequency vectors by primitive direction.

    Args:
        indices_nonzero (Sequence[MVec]): Collection of nonzero frequency vectors.

    Returns:
        Tuple[List[PR], Dict[PVec, int]]: A list of (primitive, order) pairs
        corresponding to the input vectors, and a mapping from each primitive
        direction to the maximum harmonic order observed along that ray.
    """
    pr_list: List[PR] = []
    p_to_orders: Dict[PVec, set] = {}
    for m in indices_nonzero:
        p, r = _primitive_and_order(m)
        pr_list.append((p, r))
        p_to_orders.setdefault(p, set()).add(r)
    p_to_maxr = {p: max(orders) for p, orders in p_to_orders.items()}
    return pr_list, p_to_maxr

def _pr_list_to_arrays(pr_list: List[PR], D: int | None) -> Tuple[Array, Array]:
    """
    Convert a list of primitive-ray specifications into array form.

    Args:
        pr_list (List[PR]): List of (primitive direction, harmonic order) pairs.
        D (int | None): Expected dimensionality of the primitive vectors. If None,
            the dimensionality is inferred from the first entry.

    Returns:
        Tuple[Array, Array]: A pair (p_mat, r_arr) where p_mat is an integer
        array of primitive directions with shape (Q, D) and r_arr contains
        the corresponding harmonic orders.
    """
    if not pr_list:
        D_eff = int(D) if D is not None else 0
        return np.zeros((0, D_eff), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    D_eff = len(pr_list[0][0]) if D is None else int(D)
    p_mat = np.zeros((len(pr_list), D_eff), dtype=np.int64)
    r_arr = np.zeros((len(pr_list),), dtype=np.int64)
    for idx, (p, r) in enumerate(pr_list):
        p_mat[idx, :] = np.array(p, dtype=np.int64)
        r_arr[idx] = int(r)
    return p_mat, r_arr

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
