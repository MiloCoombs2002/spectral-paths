from __future__ import annotations

import time
from itertools import combinations
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from spectral_paths.schemas import FitReport
from spectral_paths.types import Array, MVec, PVec, PR, ScalerType
from spectral_paths.utils.preprocessing import AngularTransformer
from spectral_paths.utils.helpers import (
    _build_features_numba,
    _compute_initial_importance,
    _group_by_primitive,
    _pr_list_to_arrays,
    _metrics
)


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
        L_max: int | None = None,
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
        early_stopping_patience: int = 3,
        early_stopping_tol: float = 1e-4,

        # -------- scaler config --------
        scaler_type: ScalerType = "standard_percentile_minmax",
        iqr_percentile_range: Tuple[float, float] = (25.0, 75.0),
        bound_percentiles: Tuple[float, float] = (2.0, 98.0),

        adaptive_block_size: bool = True,
        min_block_size: int = 1,
        use_importance_ordering: bool = True,
    ) -> None:
        """
        Initialise a spectral path regression model.

        Parameters:
            total_cols (int): Total number of input features (after preprocessing).
            block_size (int): Number of spectral paths added per greedy expansion step.
            lambda_grid (Sequence[float]): Candidate ridge regularisation strengths to
                evaluate during validation-based selection.
            L_max (int | None): Maximum harmonic order allowed along any primitive ray.
                If None, no explicit harmonic cutoff is enforced.
            batch_rows : int, default=2048
                Number of samples processed per batch when streaming Gram matrices.
            k_values (Sequence[int]): Allowed sparsity levels for spectral paths (number
                of interacting features per path).
            val_size (float): Fraction of data reserved for validation during greedy
                selection.
            random_state (int): Random seed used for data splitting and stochastic
                components.
            verbose (bool): Whether to emit progress and diagnostic information during
                fitting.
            final_lambda_refit (bool): Whether to refit coefficients using the selected
                regularisation parameter on the combined train and validation data.
            normalize_columns (bool): Whether to normalise feature columns prior to
                regression.
            normalize_intercept (bool): Whether to normalise the intercept term 
                alongside feature columns.
            eps_col_norm (float): Small constant used to stabilise column normalisation.
            use_float32 (bool): Whether to use float32 arithmetic internally instead of
                float64.
            early_stopping_patience (int): Number of consecutive non-improving rounds
                tolerated before early stopping of greedy path selection.
            early_stopping_tol (float): Minimum validation improvement required to reset
                early stopping.
            scaler_type (ScalerType): Scaling strategy used to map inputs to the
                interval [-1, 1] prior to the angular transformation.
            iqr_percentile_range (Tuple[float, float]): Lower and upper percentiles used
                for percentile-based scaling.
            adaptive_block_size (bool): Whether to adapt the block size during greedy
                selection based on validation behaviour.
            min_block_size (int): Minimum block size allowed when adaptive scheduling is
                enabled.
            use_importance_ordering (bool): Whether to prioritise candidate paths using
                univariate importanceheuristics.
    """
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

        self.scaler_type = str(scaler_type)
        self.transformer_: AngularTransformer | None = None
        self.iqr_percentile_range = iqr_percentile_range
        self.bound_percentiles = bound_percentiles
        self.adaptive_block_size = bool(adaptive_block_size)
        self.min_block_size = int(min_block_size)
        self.use_importance_ordering = bool(use_importance_ordering)
        self.is_fitted_: bool = False
        self.n_features_in_: int | None = None
        
        self.selected_indices_: List[MVec] | None = None
        self.pr_list_: List[PR] | None = None
        self.lambda_: float | None = None
        self.coef_: Array | None = None
        self.fit_report_: FitReport | None = None
        self.p_mat_: Array | None = None
        self.r_arr_: Array | None = None
        self.feature_importance_: Array | None = None
        self._feature_buffer_: Array | None = None

    def _make_transformer(self) -> AngularTransformer:
        """
        Construct an AngularTransformer instance from this model's scaler config.
        
        Returns:
            AngularTransformer: an instance of AngularTransformer
        """
        if self.scaler_type not in (
            "standard_percentile_minmax", "robust_percentile_minmax",
            "standard_tanh","robust_tanh", "minmax"
        ):
            raise ValueError("Scaler type not supported")

        return AngularTransformer(
            scaler=self.scaler_type, # type: ignore
            percentile_range=self.iqr_percentile_range,
            bound_percentiles=self.bound_percentiles,
            eps=float(getattr(self, "eps_scaler", self.eps_col_norm)),
        )

    def fit(
        self,
        X: Array,
        y: Array,
        *,
        X_val: Array | None = None,
        y_val: Array | None = None,
    ) -> "SpectralPathRegressorCosineOnly":
        """Fit the model."""
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
        self.transformer_ = self._make_transformer()
        theta_tr = self.transformer_.fit_transform(X_tr)
        theta_val = self.transformer_.transform(X_val)

        D = theta_tr.shape[1]
        
        # Compute initial feature importance for smart path ordering
        if self.use_importance_ordering:
            self.feature_importance_ = _compute_initial_importance(
                theta_tr,
                y_tr,
                D
            )
        else:
            self.feature_importance_ = None

        # 1) Greedy dictionary selection using incremental Gram on training split
        t0 = time.perf_counter()
        (
            selected_indices,
            lambda_star_greedy,
            history,
            G_tr_final,
            b_tr_final,
            stopped_early
        ) = self._greedy_k_mix_cos_incremental(theta_tr, y_tr, theta_val, y_val, D)

        t1 = time.perf_counter()

        # 2) Optional: re-sweep lambda for FINAL dictionary (fit on train, score on val)
        if self.final_lambda_refit and len(self.lambda_grid) > 0:
            lambda_star_final = self._select_lambda_from_gram(
                G_tr_final,
                b_tr_final,
                theta_val,
                y_val,
                selected_indices,
                self.lambda_grid
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
        self.pr_list_ = _group_by_primitive(self.selected_indices_)
        self.p_mat_, self.r_arr_ = _pr_list_to_arrays(
            self.pr_list_,
            self.n_features_in_
        )
        
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
                print(
                    f"[Final λ sweep] greedy λ*={lambda_star_greedy} "
                    f"→ final λ*={lambda_star_final}"
                )
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
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.n_features_in_}"
            )

        if self.transformer_ is None:
            raise ValueError("Transformer has not been fitted yet")
        theta = self.transformer_.transform(X)

        # Use cached structures for speed
        yhat = self._streamed_predict_from_struct(
            theta, self.pr_list_, self.coef_, self.p_mat_, self.r_arr_
        )
        return yhat

    def score(self, X: Array, y: Array) -> float:
        y = np.asarray(y, dtype=float).ravel()
        yhat = self.predict(X)
        return float(r2_score(y, yhat))
    
    def _compute_feature_importance_from_model(self) -> Array | None:
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
        L_max: int | None,
        feature_importance: Array | None = None
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

    def _solve_with_cached_eigs(
            self,
            evals: Array,
            U: Array,
            b: Array,
            lam: float
    ) -> Array:
        UTb = U.T @ b
        inv_diag = 1.0 / (evals + lam)
        return U @ (inv_diag * UTb)

    # ------------------- Streamed Gram / predict -------------------

    def _streamed_gram_cos(
            self,
            theta: Array,
            y: Array,
            indices_nonzero: Sequence[MVec]
        ) -> Tuple[Array, Array]:
        pr_list, _ = _group_by_primitive(indices_nonzero)
        p_mat, r_arr = _pr_list_to_arrays(pr_list, theta.shape[1])
        M = 1 + len(pr_list)
        G = np.zeros((M, M), dtype=float)
        b = np.zeros(M, dtype=float)

        N = theta.shape[0]
        for start in range(0, N, self.batch_rows):
            end = min(N, start + self.batch_rows)
            theta_b = theta[start:end].astype(
                np.float32 if self.use_float32 else np.float64,
                copy=False
            )
            Phi_b = _build_features_numba(theta_b, p_mat, r_arr, True)
            y_b = y[start:end]
            G += Phi_b.T @ Phi_b
            b += Phi_b.T @ y_b

        return G, b

    def _streamed_predict_from_struct(
        self,
        theta: Array,
        pr_list: List[PR],
        w: Array,
        p_mat: Array | None = None,
        r_arr: Array | None = None,
    ) -> Array:
        if p_mat is None or r_arr is None:
            if self.p_mat_ is not None and self.r_arr_ is not None:
                p_mat = self.p_mat_
                r_arr = self.r_arr_
            else:
                p_mat, r_arr = _pr_list_to_arrays(pr_list, theta.shape[1])

        N = theta.shape[0]
        yhat = np.empty(N, dtype=float)

        for start in range(0, N, self.batch_rows):
            end = min(N, start + self.batch_rows)
            Phi_b = _build_features_numba(
                theta[start:end].astype(
                    np.float32 if self.use_float32 else np.float64,
                    copy=False
                ),
                p_mat,
                r_arr,
                True,
            )
            yhat[start:end] = Phi_b @ w

        return yhat

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
        pr_list = _group_by_primitive(indices_nonzero)
        p_mat, r_arr = _pr_list_to_arrays(pr_list, theta_val.shape[1])

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
            y_val_hat = self._streamed_predict_from_struct(
                theta_val,
                pr_list,
                w,
                p_mat,
                r_arr
            )
            _, r2v = _metrics(y_val, y_val_hat)
            if r2v > best_r2:
                best_r2 = r2v
                best_lam = float(lam)

        return float(best_lam)

    # ------------------- Greedy selection (IMPROVED with early stopping) -------------

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
            k: self._r_sparse_stream(
                D,
                k,
                L_max=self.L_max,
                feature_importance=self.feature_importance_
            ) 
            for k in self.k_values
        }

        selected: List[MVec] = []
        history: List[Tuple[int, int, float, float]] = []
        lambda_star: float | None = None
        
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
            pr_old, pmax_old = _group_by_primitive(selected)
            p_mat_old, r_arr_old = _pr_list_to_arrays(pr_old, D)
            M_old = 1 + len(pr_old)

            # Structures for each candidate block
            cand_struct: Dict[int, Tuple[List[PR], Dict[PVec, int], Array, Array]] = {}
            for k, blk in candidates.items():
                pr_blk, pmax_blk = _group_by_primitive(blk)
                p_mat_blk, r_arr_blk = _pr_list_to_arrays(pr_blk, D)
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

                theta_b = theta_tr[start:end].astype(
                    np.float32 if self.use_float32 else np.float64,
                    copy=False
                )
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
                    pr_trial = _group_by_primitive(trial_indices)
                    p_mat_trial, r_arr_trial = _pr_list_to_arrays(
                        pr_trial,
                        theta_val.shape[1]
                    )

                    if self.normalize_columns:
                        s_trial = self._column_scales_from_gram(G_trial)
                        inv_s_trial = 1.0 / s_trial
                        Gs_trial = (inv_s_trial[:, None] * G_trial) * inv_s_trial[None, :]
                        bs_trial = inv_s_trial * b_trial
                        evals_trial, U_trial = np.linalg.eigh(Gs_trial)

                        def solve_for_lambda(lam_val: float) -> Array:
                            w_tilde = self._solve_with_cached_eigs(
                                evals_trial,
                                U_trial,
                                bs_trial,
                                lam_val
                            )
                            return inv_s_trial * w_tilde

                    else:
                        evals_trial, U_trial = np.linalg.eigh(G_trial)

                        def solve_for_lambda(lam_val: float) -> Array:
                            return self._solve_with_cached_eigs(
                                evals_trial,
                                U_trial,
                                b_trial,
                                lam_val
                            )

                    for lam in self.lambda_grid:
                        w = solve_for_lambda(float(lam))
                        y_val_hat = self._streamed_predict_from_struct(
                            theta_val, pr_trial, w, p_mat_trial, r_arr_trial
                        )
                        _, r2v = _metrics(y_val, y_val_hat)
                        if r2v > best_r2_this:
                            best_r2_this = r2v
                            best_lam_this = float(lam)

                    cand_val = best_r2_this
                    cand_lam = float(best_lam_this)
                else:
                    trial_indices = selected + blk
                    pr_trial = _group_by_primitive(trial_indices)
                    p_mat_trial, r_arr_trial = _pr_list_to_arrays(
                        pr_trial,
                        theta_val.shape[1]
                    )
                    w = self._solve_ridge_from_gram(G_trial, b_trial, float(lambda_star))
                    y_val_hat = self._streamed_predict_from_struct(
                        theta_val, pr_trial, w, p_mat_trial, r_arr_trial
                    )
                    _, cand_val = _metrics(y_val, y_val_hat)
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

            history.append(
                (int(k_win), int(len(blk_win)), float(lambda_star), float(best_val))
            )

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
                        print(
                            f"[Early stopping] No improvement for "
                            f"{self.early_stopping_patience} rounds at {len(selected)} "
                            "paths"
                        )
                    stopped_early = True
                    break

            if self.verbose:
                cols = len(selected)
                print(
                    f"[Greedy] Added k={k_win} block of {len(blk_win)} → total={cols} |"
                    f" λ_used={lambda_star} | R²_val={best_val:0.4f} | "
                    f"block_size={current_block_size}"
                )

        if lambda_star is None:
            lambda_star = float(self.lambda_grid[0]) if self.lambda_grid else 0.0

        # Return final training Gram/b for the selected dict (already in G_old/b_old)
        return selected, float(lambda_star), history, G_old, b_old, stopped_early

    # ------------------- Misc -------------------

    def _check_fitted(self) -> None:
        if not self.is_fitted_ or self.coef_ is None or self.selected_indices_ is None:
            raise RuntimeError("Model is not fitted yet. Call fit(X, y) first.")
