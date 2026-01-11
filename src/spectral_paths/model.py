from __future__ import annotations

import math
import time
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
from numba import njit, prange
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer


Array = np.ndarray
MVec = Tuple[int, ...]
PVec = Tuple[int, ...]
PR = Tuple[PVec, int]


@njit(cache=True)
def _cos_chebyshev(phi: float, r: int) -> float:
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
def _build_features_numba(theta_batch: Array, p_mat: Array, r_arr: Array, include_intercept: bool) -> Array:
    """Optimized feature building with parallel computation."""
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


@dataclass
class FitReport:
    lambda_star: float
    selected_count: int
    greedy_time_sec: float
    final_solve_time_sec: float
    history: List[Tuple[int, int, float, float]]  # (k_added, block_size, lambda_used, r2_val_after)
    stopped_early: bool
    feature_importance: Optional[Array] = None


class SpectralPathRegressorCosineOnly:
    """
    Improved Spectral Path Regressor with:
    - Parallel feature computation
    - Early stopping in greedy selection
    - Smart feature importance-based path generation
    - Improved input scaling with margins
    - Adaptive block sizing
    - Memory optimizations
    """

    def __init__(
        self,
        *,
        total_cols: int,
        block_size: int,
        lambda_grid: Sequence[float] = (0.01, 0.1, 1.0),
        L_max: Optional[int] = None,
        batch_rows: int = 2048,
        k_values: Sequence[int] = (1, 2, 3),
        val_size: float = 0.25,
        random_state: int = 42,
        verbose: bool = True,
        final_lambda_refit: bool = True,
        normalize_columns: bool = True,
        normalize_intercept: bool = False,
        eps_col_norm: float = 1e-12,
        use_float32: bool = False,
        # New parameters
        early_stopping_patience: int = 3,
        early_stopping_tol: float = 1e-4,

        # -------- NEW: scaler config --------
        scaler_type: Literal[
            "standard_minmax",
            "standard_percentile_minmax",
            "robust_qclip_minmax",
            "robust_sigclip_minmax",
            "quantile_uniform"
        ] = "standard_percentile_minmax",
        # used by standard_minmax / standard_percentile_minmax:
        scaling_margin: float = 0.1,
        percentile_range: Tuple[float, float] = (1.0, 99.0),

        # used by robust_qclip_minmax:
        robust_clip_quantiles: Tuple[float, float] = (2.0, 98.0),

        # used by robust_sigclip_minmax:
        robust_sigma_clip: float = 3.0,

        # used by quantile_uniform:
        quantile_n_quantiles: int = 1000,
        quantile_subsample: int = 200000,
        quantile_random_state: int = 42,

        adaptive_block_size: bool = True,
        min_block_size: int = 1,
        use_importance_ordering: bool = True,

        # (kept for backward-compat; if you still pass them, we’ll map to scaler_type)
        use_percentile_scaling: bool = False,
    ):
        self.total_cols = int(total_cols)
        self.block_size = int(block_size)
        self.lambda_grid = list(lambda_grid)
        self.L_max = L_max
        self.batch_rows = int(batch_rows)
        self.k_values = tuple(int(k) for k in k_values)
        self.val_size = float(val_size)
        self.random_state = int(random_state)
        self.verbose = bool(verbose)
        self.final_lambda_refit = bool(final_lambda_refit)
        self.normalize_columns = bool(normalize_columns)
        self.normalize_intercept = bool(normalize_intercept)
        self.eps_col_norm = float(eps_col_norm)
        self.use_float32 = bool(use_float32)

        self.early_stopping_patience = int(early_stopping_patience)
        self.early_stopping_tol = float(early_stopping_tol)

        # ---- scaling config ----
        # Backward-compat: old flag maps to old behavior
        if use_percentile_scaling and scaler_type == "standard_minmax":
            scaler_type = "standard_percentile_minmax"

        self.scaler_type = str(scaler_type)
        self.scaling_margin = float(scaling_margin)
        self.percentile_range = tuple(float(x) for x in percentile_range)
        self.robust_clip_quantiles = tuple(float(x) for x in robust_clip_quantiles)
        self.robust_sigma_clip = float(robust_sigma_clip)
        self.quantile_n_quantiles = int(quantile_n_quantiles)
        self.quantile_subsample = int(quantile_subsample)
        self.quantile_random_state = int(quantile_random_state)

        self.adaptive_block_size = bool(adaptive_block_size)
        self.min_block_size = int(min_block_size)
        self.use_importance_ordering = bool(use_importance_ordering)

        # Learned attributes
        self.is_fitted_: bool = False
        self.n_features_in_: Optional[int] = None

        # -------- NEW: we generalize "scaler_" into a base transformer + post params --------
        self.base_scaler_ = None  # StandardScaler / RobustScaler / QuantileTransformer
        self.x_min_: Optional[Array] = None   # for affine map to [-1,1]
        self.span_: Optional[Array] = None    # for affine map to [-1,1]
        self.z_clip_lo_: Optional[Array] = None  # clipping bounds in z-space (robust clip / sigma clip)
        self.z_clip_hi_: Optional[Array] = None

        # (rest unchanged)
        self.selected_indices_: Optional[List[MVec]] = None
        self.pr_list_: Optional[List[PR]] = None
        self.p_to_maxr_: Optional[Dict[PVec, int]] = None
        self.lambda_: Optional[float] = None
        self.coef_: Optional[Array] = None
        self.fit_report_: Optional[FitReport] = None
        self.p_mat_: Optional[Array] = None
        self.r_arr_: Optional[Array] = None
        self.feature_importance_: Optional[Array] = None
        self._feature_buffer_: Optional[Array] = None

    # ------------------- Public API -------------------
    def _fit_transformer(self, X_tr: Array) -> None:
        """
        Fit scaling so that X_u is in [-1,1] while reducing outlier influence.

        Supported scaler_type:
          - "standard_minmax"            : StandardScaler -> min/max (+ margin) -> [-1,1]
          - "standard_percentile_minmax" : StandardScaler -> percentile range -> [-1,1]
          - "robust_qclip_minmax"        : RobustScaler -> clip quantiles -> [-1,1]
          - "robust_sigclip_minmax"      : RobustScaler -> clip +/- k*sigma -> [-1,1]
          - "quantile_uniform"           : QuantileTransformer(uniform) -> [-1,1]
        """
        st = self.scaler_type

        if st == "standard_minmax":
            self.base_scaler_ = StandardScaler().fit(X_tr)
            Z = self.base_scaler_.transform(X_tr)

            z_min_raw = Z.min(axis=0)
            z_max_raw = Z.max(axis=0)
            # margins to reduce accidental test overflow
            margin_low = self.scaling_margin * np.abs(z_min_raw)
            margin_high = self.scaling_margin * np.abs(z_max_raw)
            z_min = z_min_raw - margin_low
            z_max = z_max_raw + margin_high

            self.x_min_ = z_min
            self.span_ = np.where((z_max - z_min) == 0.0, 1.0, (z_max - z_min))
            self.z_clip_lo_ = None
            self.z_clip_hi_ = None
            return

        if st == "standard_percentile_minmax":
            self.base_scaler_ = StandardScaler().fit(X_tr)
            Z = self.base_scaler_.transform(X_tr)

            lo, hi = self.percentile_range
            z_min = np.percentile(Z, lo, axis=0)
            z_max = np.percentile(Z, hi, axis=0)

            self.x_min_ = z_min
            self.span_ = np.where((z_max - z_min) == 0.0, 1.0, (z_max - z_min))
            self.z_clip_lo_ = None
            self.z_clip_hi_ = None
            return

        if st == "robust_qclip_minmax":
            self.base_scaler_ = RobustScaler(with_centering=True, with_scaling=True).fit(X_tr)
            Z = self.base_scaler_.transform(X_tr)

            lo, hi = self.robust_clip_quantiles
            z_lo = np.percentile(Z, lo, axis=0)
            z_hi = np.percentile(Z, hi, axis=0)

            # clip then affine-map clipped support to [-1,1]
            self.z_clip_lo_ = z_lo
            self.z_clip_hi_ = z_hi
            self.x_min_ = z_lo
            self.span_ = np.where((z_hi - z_lo) == 0.0, 1.0, (z_hi - z_lo))
            return

        if st == "robust_sigclip_minmax":
            self.base_scaler_ = RobustScaler(with_centering=True, with_scaling=True).fit(X_tr)
            Z = self.base_scaler_.transform(X_tr)

            # After RobustScaler, per-feature spread is roughly IQR-scaled, not std.
            # Still: sigma-clip is a decent “simple lever” in that scaled space.
            k = float(self.robust_sigma_clip)
            z_lo = -k * np.ones(Z.shape[1], dtype=float)
            z_hi = +k * np.ones(Z.shape[1], dtype=float)

            self.z_clip_lo_ = z_lo
            self.z_clip_hi_ = z_hi
            self.x_min_ = z_lo
            self.span_ = np.where((z_hi - z_lo) == 0.0, 1.0, (z_hi - z_lo))
            return

        if st == "quantile_uniform":
            self.base_scaler_ = QuantileTransformer(
                n_quantiles=self.quantile_n_quantiles,
                output_distribution="uniform",
                subsample=self.quantile_subsample,
                random_state=self.quantile_random_state,
                copy=True,
            ).fit(X_tr)

            # QuantileTransformer outputs U in [0,1]. We'll map directly to [-1,1].
            self.x_min_ = None
            self.span_ = None
            self.z_clip_lo_ = None
            self.z_clip_hi_ = None
            return

        raise ValueError(
            f"Unknown scaler_type={st!r}. "
            "Use one of: "
            "'standard_minmax', 'standard_percentile_minmax', "
            "'robust_qclip_minmax', 'robust_sigclip_minmax', 'quantile_uniform'."
        )
    def fit(
        self,
        X: Array,
        y: Array,
        *,
        X_val: Optional[Array] = None,
        y_val: Optional[Array] = None,
    ) -> "SpectralPathRegressorCosineOnly":
        X = np.asarray(X)
        y = np.asarray(y, dtype=float).ravel()

        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X rows {X.shape[0]} != y rows {y.shape[0]}")

        self.n_features_in_ = X.shape[1]

        # Split train/val if not provided
        if X_val is None or y_val is None:
            X_tr, X_val2, y_tr, y_val2 = train_test_split(
                X, y, test_size=self.val_size, random_state=self.random_state
            )
            X_val, y_val = X_val2, y_val2
        else:
            X_tr, y_tr = X, y
            X_val = np.asarray(X_val)
            y_val = np.asarray(y_val, dtype=float).ravel()
            if X_val.ndim != 2 or X_val.shape[1] != X_tr.shape[1]:
                raise ValueError("X_val must be 2D with same number of columns as X.")
            if X_val.shape[0] != y_val.shape[0]:
                raise ValueError("X_val rows must match y_val length.")

        # Fit transform parameters on training only
        self._fit_transformer(X_tr)
        theta_tr = self._X_to_theta(X_tr)
        theta_val = self._X_to_theta(X_val)
        D = theta_tr.shape[1]
        
        # Compute initial feature importance for smart path ordering
        if self.use_importance_ordering:
            self.feature_importance_ = self._compute_initial_importance(theta_tr, y_tr, D)
        else:
            self.feature_importance_ = None

        # 1) Greedy dictionary selection using incremental Gram on training split
        t0 = time.perf_counter()
        selected_indices, lambda_star_greedy, history, G_tr_final, b_tr_final, stopped_early = self._greedy_k_mix_cos_incremental(
            theta_tr, y_tr, theta_val, y_val, D
        )
        t1 = time.perf_counter()

        # 2) Optional: re-sweep lambda for FINAL dictionary (fit on train, score on val)
        if self.final_lambda_refit and len(self.lambda_grid) > 0:
            lambda_star_final = self._select_lambda_from_gram(
                G_tr_final, b_tr_final, theta_val, y_val, selected_indices, self.lambda_grid
            )
        else:
            lambda_star_final = float(lambda_star_greedy)

        # 3) Final fit on (train + val) using lambda_star_final
        theta_all = np.vstack([theta_tr, theta_val])
        y_all = np.concatenate([y_tr, y_val])

        t2 = time.perf_counter()
        G_all, b_all = self._streamed_gram_cos(theta_all, y_all, selected_indices)
        w_star = self._solve_ridge_from_gram(G_all, b_all, float(lambda_star_final))
        t3 = time.perf_counter()

        # Save learned state
        self.selected_indices_ = selected_indices
        self.lambda_ = float(lambda_star_final)
        self.coef_ = w_star.astype(float, copy=False)

        # Cache ray structures for predict
        self.pr_list_, self.p_to_maxr_ = self._group_by_primitive(self.selected_indices_)
        self.p_mat_, self.r_arr_ = self._pr_list_to_arrays(self.pr_list_, self.n_features_in_)
        
        # Compute feature importance from final model
        final_importance = self._compute_feature_importance_from_model()

        self.fit_report_ = FitReport(
            lambda_star=float(lambda_star_final),
            selected_count=len(selected_indices),
            greedy_time_sec=float(t1 - t0),
            final_solve_time_sec=float(t3 - t2),
            history=history,
            stopped_early=stopped_early,
            feature_importance=final_importance,
        )

        if self.verbose and self.final_lambda_refit:
            if float(lambda_star_final) != float(lambda_star_greedy):
                print(f"[Final λ sweep] greedy λ*={lambda_star_greedy} → final λ*={lambda_star_final}")
            else:
                print(f"[Final λ sweep] final λ*={lambda_star_final} (same as greedy)")

        self.is_fitted_ = True
        return self

    def predict(self, X: Array) -> Array:
        self._check_fitted()
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, expected {self.n_features_in_}")

        theta = self._X_to_theta(X)
        # Use cached structures for speed
        yhat = self._streamed_predict_from_struct(
            theta, self.pr_list_, self.p_to_maxr_, self.coef_, self.p_mat_, self.r_arr_
        )
        return yhat

    def score(self, X: Array, y: Array) -> float:
        y = np.asarray(y, dtype=float).ravel()
        yhat = self.predict(X)
        return float(r2_score(y, yhat))

    # ------------------- Transformer (IMPROVED) -------------------

    # def _fit_transformer(self, X_tr: Array) -> None:
    #     """Improved transformer with margins and percentile-based scaling."""
    #     self.scaler_ = StandardScaler().fit(X_tr)
    #     X_tr_z = self.scaler_.transform(X_tr)

    #     if self.use_percentile_scaling:
    #         # Use percentiles to be robust to outliers
    #         x_min = np.percentile(X_tr_z, self.percentile_range[0], axis=0)
    #         x_max = np.percentile(X_tr_z, self.percentile_range[1], axis=0)
    #     else:
    #         # Add margin to avoid boundary issues with test data
    #         x_min_raw = X_tr_z.min(axis=0)
    #         x_max_raw = X_tr_z.max(axis=0)
            
    #         margin_low = self.scaling_margin * np.abs(x_min_raw)
    #         margin_high = self.scaling_margin * np.abs(x_max_raw)
            
    #         x_min = x_min_raw - margin_low
    #         x_max = x_max_raw + margin_high

    #     span = np.where((x_max - x_min) == 0.0, 1.0, (x_max - x_min))

    #     self.x_min_ = x_min
    #     self.span_ = span

    def _X_to_theta(self, X: Array) -> Array:
        if self.base_scaler_ is None:
            raise RuntimeError("Transformer not fitted.")

        st = self.scaler_type

        if st == "quantile_uniform":
            U = self.base_scaler_.transform(X)  # in [0,1]
            X_u = (2.0 * U) - 1.0
            X_u = np.clip(X_u, -1.0, 1.0)
            return np.arccos(X_u)

        # everything else: base_scaler_ -> Z, optional clip in Z-space -> affine to [-1,1]
        Z = self.base_scaler_.transform(X)

        if self.z_clip_lo_ is not None and self.z_clip_hi_ is not None:
            Z = np.clip(Z, self.z_clip_lo_, self.z_clip_hi_)

        if self.x_min_ is None or self.span_ is None:
            raise RuntimeError("Affine params missing (x_min_/span_).")

        X_u = (2.0 * (Z - self.x_min_) / self.span_) - 1.0
        X_u = np.clip(X_u, -1.0, 1.0)
        return np.arccos(X_u)

    # ------------------- Feature Importance -------------------
    
    def _compute_initial_importance(self, theta_tr: Array, y_tr: Array, D: int) -> Array:
        """Compute initial feature importance using univariate correlations in angular space."""
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
    
    def _compute_feature_importance_from_model(self) -> Array:
        """Compute feature importance from learned coefficients."""
        if self.selected_indices_ is None or self.coef_ is None:
            return None
        
        D = self.n_features_in_
        importance = np.zeros(D, dtype=float)
        
        # Skip intercept (index 0)
        for idx, m in enumerate(self.selected_indices_):
            coef_val = abs(self.coef_[idx + 1])  # +1 for intercept
            for j, m_j in enumerate(m):
                if m_j != 0:
                    importance[j] += coef_val * abs(m_j)
        
        # Normalize
        if importance.sum() > 0:
            importance = importance / importance.sum()
        
        return importance

    # ------------------- Enumerators (IMPROVED) -------------------

    def _balanced_compositions(self, L: int, r: int) -> List[List[int]]:
        comps: List[List[int]] = []

        def rec(prefix: List[int], remaining: int, slots: int) -> None:
            if slots == 1:
                comps.append(prefix + [remaining])
                return
            for a in range(1, remaining - slots + 2):
                rec(prefix + [a], remaining - a, slots - 1)

        rec([], L, r)

        def key(c: List[int]) -> Tuple[int, Tuple[int, ...]]:
            l_inf = max(c)
            return (l_inf, tuple([-x for x in sorted(c, reverse=True)]))

        return sorted(comps, key=key)

    def _r_sparse_stream(
        self, 
        D: int, 
        r: int, 
        L_max: Optional[int],
        feature_importance: Optional[Array] = None
    ) -> Iterable[MVec]:
        """Generate k-sparse paths, optionally prioritizing high-importance features."""
        if feature_importance is not None and self.use_importance_ordering:
            # Sort features by importance (descending)
            sorted_feats = np.argsort(-feature_importance)
        else:
            sorted_feats = np.arange(D)
        
        L = 1
        while L_max is None or L <= L_max:
            for comp in self._balanced_compositions(L, r):
                for S in combinations(sorted_feats, r):
                    m = [0] * D
                    for idx, val in zip(S, comp):
                        m[idx] = val
                    yield tuple(m)
            L += 1

    # ------------------- Primitive rays -------------------

    def _gcd_list(self, vals: Sequence[int]) -> int:
        g = 0
        for v in vals:
            g = math.gcd(g, int(abs(v)))
        return g

    def _primitive_and_order(self, m: MVec) -> PR:
        nonzeros = [v for v in m if v != 0]
        g = self._gcd_list(nonzeros) if nonzeros else 1
        if g == 0:
            g = 1
        p = tuple((vi // g) for vi in m)
        r = g
        return p, r

    def _group_by_primitive(self, indices_nonzero: Sequence[MVec]) -> Tuple[List[PR], Dict[PVec, int]]:
        pr_list: List[PR] = []
        p_to_orders: Dict[PVec, set] = {}
        for m in indices_nonzero:
            p, r = self._primitive_and_order(m)
            pr_list.append((p, r))
            p_to_orders.setdefault(p, set()).add(r)
        p_to_maxr = {p: max(orders) for p, orders in p_to_orders.items()}
        return pr_list, p_to_maxr

    def _pr_list_to_arrays(self, pr_list: List[PR], D: Optional[int]) -> Tuple[Array, Array]:
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

    # ------------------- Feature building (uses optimized numba) -------------------

    def _build_batch_features_graph_cos(
        self,
        theta_batch: Array,
        pr_list: List[PR],
        p_to_maxr: Dict[PVec, int],
        *,
        include_intercept: bool,
    ) -> Array:
        p_mat, r_arr = self._pr_list_to_arrays(pr_list, theta_batch.shape[1])
        theta_local = theta_batch.astype(np.float32 if self.use_float32 else np.float64, copy=False)
        return _build_features_numba(theta_local, p_mat, r_arr, include_intercept)

    # ------------------- Column normalization + ridge solve -------------------

    def _column_scales_from_gram(self, G: Array) -> Array:
        # s_j = ||Phi_j||_2 = sqrt(G_jj)
        s = np.sqrt(np.maximum(np.diag(G), 0.0))
        s = np.where(s < self.eps_col_norm, 1.0, s)
        if not self.normalize_intercept and s.size > 0:
            s[0] = 1.0
        return s

    def _solve_ridge_from_gram(self, G: Array, b: Array, lam: float) -> Array:
        """
        Solve ridge from Gram, optionally with implicit column normalization.

        If normalized:
            Phi_tilde = Phi * diag(1/s)
            Solve (G_tilde + lam I) w_tilde = b_tilde
            Return w = (1/s) * w_tilde (so predictions use original Phi).
        """
        lam_f = float(lam)

        if self.normalize_columns:
            s = self._column_scales_from_gram(G)
            inv_s = 1.0 / s
            Gs = (inv_s[:, None] * G) * inv_s[None, :]
            bs = inv_s * b

            evals, U = np.linalg.eigh(Gs)
            w_tilde = self._solve_with_cached_eigs(evals, U, bs, lam_f)

            w = inv_s * w_tilde
            return w

        evals, U = np.linalg.eigh(G)
        w = self._solve_with_cached_eigs(evals, U, b, lam_f)
        return w

    def _solve_with_cached_eigs(self, evals: Array, U: Array, b: Array, lam: float) -> Array:
        UTb = U.T @ b
        inv_diag = 1.0 / (evals + lam)
        return U @ (inv_diag * UTb)

    # ------------------- Streamed Gram / predict -------------------

    def _streamed_gram_cos(self, theta: Array, y: Array, indices_nonzero: Sequence[MVec]) -> Tuple[Array, Array]:
        pr_list, _ = self._group_by_primitive(indices_nonzero)
        p_mat, r_arr = self._pr_list_to_arrays(pr_list, theta.shape[1])
        M = 1 + len(pr_list)
        G = np.zeros((M, M), dtype=float)
        b = np.zeros(M, dtype=float)

        N = theta.shape[0]
        for start in range(0, N, self.batch_rows):
            end = min(N, start + self.batch_rows)
            theta_b = theta[start:end].astype(np.float32 if self.use_float32 else np.float64, copy=False)
            Phi_b = _build_features_numba(theta_b, p_mat, r_arr, True)
            y_b = y[start:end]
            G += Phi_b.T @ Phi_b
            b += Phi_b.T @ y_b

        return G, b

    def _streamed_predict_from_struct(
        self,
        theta: Array,
        pr_list: List[PR],
        p_to_maxr: Dict[PVec, int],
        w: Array,
        p_mat: Optional[Array] = None,
        r_arr: Optional[Array] = None,
    ) -> Array:
        if p_mat is None or r_arr is None:
            if self.p_mat_ is not None and self.r_arr_ is not None:
                p_mat = self.p_mat_
                r_arr = self.r_arr_
            else:
                p_mat, r_arr = self._pr_list_to_arrays(pr_list, theta.shape[1])

        N = theta.shape[0]
        yhat = np.empty(N, dtype=float)

        for start in range(0, N, self.batch_rows):
            end = min(N, start + self.batch_rows)
            Phi_b = _build_features_numba(
                theta[start:end].astype(np.float32 if self.use_float32 else np.float64, copy=False),
                p_mat,
                r_arr,
                True,
            )
            yhat[start:end] = Phi_b @ w

        return yhat

    def _metrics(self, y_true: Array, y_pred: Array) -> Tuple[float, float]:
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred))
        return rmse, r2

    # ------------------- Final λ selection on fixed dictionary -------------------

    def _select_lambda_from_gram(
        self,
        G_tr: Array,
        b_tr: Array,
        theta_val: Array,
        y_val: Array,
        indices_nonzero: Sequence[MVec],
        lambda_grid: Sequence[float],
    ) -> float:
        best_r2 = -1e18
        best_lam = float(lambda_grid[0])

        # Precompute structures once for validation prediction speed
        pr_list, p_to_maxr = self._group_by_primitive(indices_nonzero)
        p_mat, r_arr = self._pr_list_to_arrays(pr_list, theta_val.shape[1])

        # Cache eigendecomp for lambda sweep
        if self.normalize_columns:
            s = self._column_scales_from_gram(G_tr)
            inv_s = 1.0 / s
            Gs = (inv_s[:, None] * G_tr) * inv_s[None, :]
            bs = inv_s * b_tr
            evals, U = np.linalg.eigh(Gs)

            def solve_for_lambda(lam_val: float) -> Array:
                w_tilde = self._solve_with_cached_eigs(evals, U, bs, lam_val)
                return inv_s * w_tilde

        else:
            evals, U = np.linalg.eigh(G_tr)

            def solve_for_lambda(lam_val: float) -> Array:
                return self._solve_with_cached_eigs(evals, U, b_tr, lam_val)

        for lam in lambda_grid:
            w = solve_for_lambda(float(lam))
            y_val_hat = self._streamed_predict_from_struct(theta_val, pr_list, p_to_maxr, w, p_mat, r_arr)
            _, r2v = self._metrics(y_val, y_val_hat)
            if r2v > best_r2:
                best_r2 = r2v
                best_lam = float(lam)

        return float(best_lam)

    # ------------------- Greedy selection (IMPROVED with early stopping) -------------------

    def _greedy_k_mix_cos_incremental(
        self,
        theta_tr: Array, y_tr: Array,
        theta_val: Array, y_val: Array,
        D: int,
    ) -> Tuple[List[MVec], float, List[Tuple[int, int, float, float]], Array, Array, bool]:
        """
        Improved greedy selection with:
        - Early stopping when validation performance plateaus
        - Adaptive block sizing
        - Feature importance-based path ordering
        """
        gens = {
            k: self._r_sparse_stream(D, k, L_max=self.L_max, feature_importance=self.feature_importance_) 
            for k in self.k_values
        }

        selected: List[MVec] = []
        history: List[Tuple[int, int, float, float]] = []
        lambda_star: Optional[float] = None
        
        # Early stopping trackers
        best_val_overall = -1e18
        no_improve_count = 0
        stopped_early = False
        
        # Adaptive block size
        current_block_size = self.block_size

        # Start with intercept-only on training split
        Ntr = theta_tr.shape[0]
        G_old = np.array([[float(Ntr)]], dtype=float)
        b_old = np.array([float(y_tr.sum())], dtype=float)

        while len(selected) < self.total_cols:
            remaining = self.total_cols - len(selected)
            bs = min(current_block_size, remaining)

            # Propose blocks from each k
            candidates: Dict[int, List[MVec]] = {}
            for k in self.k_values:
                blk: List[MVec] = []
                for _ in range(bs):
                    try:
                        blk.append(next(gens[k]))
                    except StopIteration:
                        break
                if blk:
                    candidates[k] = blk

            if not candidates:
                if self.verbose:
                    print("All generators exhausted before reaching total_cols.")
                break

            # Structures for current dictionary (train + val prediction)
            pr_old, pmax_old = self._group_by_primitive(selected)
            p_mat_old, r_arr_old = self._pr_list_to_arrays(pr_old, D)
            M_old = 1 + len(pr_old)

            # Structures for each candidate block
            cand_struct: Dict[int, Tuple[List[PR], Dict[PVec, int], Array, Array]] = {}
            for k, blk in candidates.items():
                pr_blk, pmax_blk = self._group_by_primitive(blk)
                p_mat_blk, r_arr_blk = self._pr_list_to_arrays(pr_blk, D)
                cand_struct[k] = (pr_blk, pmax_blk, p_mat_blk, r_arr_blk)

            # Allocate accumulators per candidate: C (M_old x Qnew), Gnew (Qnew x Qnew), bnew (Qnew)
            C_map: Dict[int, Array] = {}
            Gnew_map: Dict[int, Array] = {}
            bnew_map: Dict[int, Array] = {}
            for k, (pr_blk, _pmax_blk, _p_mat_blk, _r_arr_blk) in cand_struct.items():
                Qnew = len(pr_blk)
                C_map[k] = np.zeros((M_old, Qnew), dtype=float)
                Gnew_map[k] = np.zeros((Qnew, Qnew), dtype=float)
                bnew_map[k] = np.zeros(Qnew, dtype=float)

            # Single streaming pass over TRAIN to compute candidate block stats
            N = theta_tr.shape[0]
            for start in range(0, N, self.batch_rows):
                end = min(N, start + self.batch_rows)
                y_b = y_tr[start:end]

                theta_b = theta_tr[start:end].astype(np.float32 if self.use_float32 else np.float64, copy=False)
                Phi_old_b = _build_features_numba(
                    theta_b,
                    p_mat_old,
                    r_arr_old,
                    True,
                )  # (B, M_old)

                for k, (_pr_blk, _pmax_blk, p_mat_blk, r_arr_blk) in cand_struct.items():
                    Phi_new_b = _build_features_numba(
                        theta_b,
                        p_mat_blk,
                        r_arr_blk,
                        False,
                    )  # (B, Qnew)

                    C_map[k] += Phi_old_b.T @ Phi_new_b
                    Gnew_map[k] += Phi_new_b.T @ Phi_new_b
                    bnew_map[k] += Phi_new_b.T @ y_b

            # Evaluate candidates (solve on train using block Gram; score on val)
            best_val = -1e18
            best_choice = None  # (k, blk, G_trial, b_trial, lam_used)

            for k, blk in candidates.items():
                C = C_map[k]
                Gnew = Gnew_map[k]
                bnew = bnew_map[k]

                # Build trial Gram/b
                G_trial = np.block([
                    [G_old, C],
                    [C.T,  Gnew],
                ])
                b_trial = np.concatenate([b_old, bnew])

                # Solve (possibly with lambda sweep on first commit)
                if lambda_star is None:
                    best_r2_this = -1e18
                    best_lam_this = float(self.lambda_grid[0]) if self.lambda_grid else 0.0
                    # Precompute validation structures once for this trial dict
                    trial_indices = selected + blk
                    pr_trial, pmax_trial = self._group_by_primitive(trial_indices)
                    p_mat_trial, r_arr_trial = self._pr_list_to_arrays(pr_trial, theta_val.shape[1])

                    if self.normalize_columns:
                        s_trial = self._column_scales_from_gram(G_trial)
                        inv_s_trial = 1.0 / s_trial
                        Gs_trial = (inv_s_trial[:, None] * G_trial) * inv_s_trial[None, :]
                        bs_trial = inv_s_trial * b_trial
                        evals_trial, U_trial = np.linalg.eigh(Gs_trial)

                        def solve_for_lambda(lam_val: float) -> Array:
                            w_tilde = self._solve_with_cached_eigs(evals_trial, U_trial, bs_trial, lam_val)
                            return inv_s_trial * w_tilde

                    else:
                        evals_trial, U_trial = np.linalg.eigh(G_trial)

                        def solve_for_lambda(lam_val: float) -> Array:
                            return self._solve_with_cached_eigs(evals_trial, U_trial, b_trial, lam_val)

                    for lam in self.lambda_grid:
                        w = solve_for_lambda(float(lam))
                        y_val_hat = self._streamed_predict_from_struct(
                            theta_val, pr_trial, pmax_trial, w, p_mat_trial, r_arr_trial
                        )
                        _, r2v = self._metrics(y_val, y_val_hat)
                        if r2v > best_r2_this:
                            best_r2_this = r2v
                            best_lam_this = float(lam)

                    cand_val = best_r2_this
                    cand_lam = float(best_lam_this)
                else:
                    trial_indices = selected + blk
                    pr_trial, pmax_trial = self._group_by_primitive(trial_indices)
                    p_mat_trial, r_arr_trial = self._pr_list_to_arrays(pr_trial, theta_val.shape[1])
                    w = self._solve_ridge_from_gram(G_trial, b_trial, float(lambda_star))
                    y_val_hat = self._streamed_predict_from_struct(
                        theta_val, pr_trial, pmax_trial, w, p_mat_trial, r_arr_trial
                    )
                    _, cand_val = self._metrics(y_val, y_val_hat)
                    cand_lam = float(lambda_star)

                if cand_val > best_val:
                    best_val = cand_val
                    best_choice = (k, blk, G_trial, b_trial, cand_lam)

            # Commit best
            k_win, blk_win, G_new_old, b_new_old, lam_win = best_choice  # type: ignore
            selected = selected + blk_win
            G_old = G_new_old
            b_old = b_new_old

            if lambda_star is None:
                lambda_star = float(lam_win)

            history.append((int(k_win), int(len(blk_win)), float(lambda_star), float(best_val)))

            # Early stopping check
            if best_val > best_val_overall + self.early_stopping_tol:
                best_val_overall = best_val
                no_improve_count = 0
                
                # Increase block size if doing well
                if self.adaptive_block_size and current_block_size < self.block_size:
                    current_block_size = min(self.block_size, current_block_size + 1)
            else:
                no_improve_count += 1
                
                # Decrease block size if plateauing
                if self.adaptive_block_size and current_block_size > self.min_block_size:
                    current_block_size = max(self.min_block_size, current_block_size - 1)
                
                if no_improve_count >= self.early_stopping_patience:
                    if self.verbose:
                        print(f"[Early stopping] No improvement for {self.early_stopping_patience} rounds at {len(selected)} paths")
                    stopped_early = True
                    break

            if self.verbose:
                cols = len(selected)
                print(
                    f"[Greedy] Added k={k_win} block of {len(blk_win)} → total={cols} | "
                    f"λ_used={lambda_star} | R²_val={best_val:0.4f} | block_size={current_block_size}"
                )

        if lambda_star is None:
            lambda_star = float(self.lambda_grid[0]) if self.lambda_grid else 0.0

        # Return final training Gram/b for the selected dictionary (already in G_old/b_old)
        return selected, float(lambda_star), history, G_old, b_old, stopped_early

    # ------------------- Misc -------------------

    def _check_fitted(self) -> None:
        if not self.is_fitted_ or self.coef_ is None or self.selected_indices_ is None:
            raise RuntimeError("Model is not fitted yet. Call fit(X, y) first.")


# ------------------- Example usage -------------------
if __name__ == "__main__":
    from ucimlrepo import fetch_ucirepo

    ds = fetch_ucirepo(id=327)
    X = ds.data.features.values
    y = ds.data.targets.values.astype(float).ravel()

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42)

    D = X.shape[1]

    mdl = SpectralPathRegressorCosineOnly(
        total_cols=22 * D,
        block_size=1 * D,
        lambda_grid=[0.0005,0.001, 0.003, 0.01, 0.03],
        L_max=None,
        scaler_type="standard_percentile_minmax",
        robust_clip_quantiles=(2.0, 98.0),
        batch_rows=2048,
        verbose=True,
        random_state=42,
        val_size=0.25,
        final_lambda_refit=True,
        normalize_columns=True,
        normalize_intercept=False,
        k_values=(1,2,3,4),
        # New improved parameters
        early_stopping_patience=5,
        early_stopping_tol= 1e-4,
        scaling_margin=0.1,
        use_percentile_scaling=False,
        adaptive_block_size=True,
        min_block_size=1,
        use_importance_ordering=True,
    )
    print("Training model...")
    t0 = time.perf_counter()
    mdl.fit(X_tr, y_tr)
    t1 = time.perf_counter()
    
    print(f"\nTotal training time: {t1-t0:.3f}s")

    yhat = mdl.predict(X_te)

    rmse = float(np.sqrt(mean_squared_error(y_te, yhat)))
    r2 = float(r2_score(y_te, yhat))

    print("\n=== Test ===")
    print(f"R²:   {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")

    if mdl.fit_report_ is not None:
        print("\n=== Timing ===")
        print(f"Greedy selection: {mdl.fit_report_.greedy_time_sec:.3f}s")
        print(f"Final solve:      {mdl.fit_report_.final_solve_time_sec:.3f}s")
        print(f"Selected cols:    {mdl.fit_report_.selected_count} (+ intercept)")
        print(f"λ*:               {mdl.fit_report_.lambda_star}")
        print(f"Stopped early:    {mdl.fit_report_.stopped_early}")
        
        if mdl.fit_report_.feature_importance is not None:
            print("\n=== Top 5 Most Important Features ===")
            top_idx = np.argsort(-mdl.fit_report_.feature_importance)[:5]
            for rank, idx in enumerate(top_idx, 1):
                print(f"{rank}. Feature {idx}: {mdl.fit_report_.feature_importance[idx]:.4f}")