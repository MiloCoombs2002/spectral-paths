"""
Spectral path regression model.

This module implements the `SpectralPathRegressor` model.
"""
from __future__ import annotations

import time
from itertools import combinations
from typing import Dict, Iterator, List, Sequence, Tuple, cast

import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from spectral_paths.schemas import FitReport, Stats
from spectral_paths.types import Array, MVec, ScalerType
from spectral_paths.utils.helpers import (
    _build_feature_matrix,
    _compute_initial_importance,
    _metrics,
    _path_matrix_and_r_arr,
    check_dimensions,
)
from spectral_paths.utils.preprocessing import AngularTransformer


class SpectralPathRegressor:
    """
    Spectral-path regression model.

    This estimator builds spectral path features, selects a sparse dictionary
    via a greedy procedure, and fits ridge-style coefficients with optional
    early stopping and final refit.
    """
    def __init__(
        self,
        *,
        max_paths: int,
        block_size: int,
        lambda_grid: Sequence[float] = (0.01, 0.1, 1.0),
        l_max: int | None = None,
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
            max_paths (int): Total number of input features (after preprocessing).
            block_size (int): Number of spectral paths added per greedy expansion step.
            lambda_grid (Sequence[float]): Candidate ridge regularisation strengths to
                evaluate during validation-based selection.
            l_max (int | None): Maximum harmonic order allowed along any primitive ray.
                If None, no explicit harmonic cutoff is enforced.
            batch_rows (int): default=2048
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
        if max_paths <= 0:
            raise ValueError("max_paths must be a non-negative integer")
        if not isinstance(max_paths, int):
            raise ValueError("max_paths must be an integer")
        self.max_paths = max_paths
        self.block_size = int(block_size)
        self.lambda_grid = [float(lam) for lam in lambda_grid]
        if len(self.lambda_grid) == 0:
            raise ValueError("lambda_grid must contain at least one value.")
        if any((not np.isfinite(lam)) or lam < 0.0 for lam in self.lambda_grid):
            raise ValueError("lambda_grid values must be finite and >= 0.")
        self.l_max = l_max
        if self.l_max is not None and int(self.l_max) < 1:
            raise ValueError("l_max must be >= 1 when provided.")
        self.batch_rows = int(batch_rows)
        self.k_values = tuple(int(k) for k in k_values)
        if len(self.k_values) == 0:
            raise ValueError("k_values must contain at least one integer.")
        if any(k < 1 for k in self.k_values):
            raise ValueError("All k_values must be >= 1.")
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
        self.n_features_in_: int | None = None

        self.selected_paths_: List[MVec] | None = None
        self.lambda_: float | None = None
        self.coef_: Array | None = None
        self.fit_report_: FitReport | None = None
        self.p_mat_: Array | None = None
        self.r_arr_: Array | None = None
        self.feature_importance_: Array | None = None

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

    def prepare_train_val_split(
        self, X: Array, y: Array, X_val: Array | None, y_val: Array | None
    ) -> tuple[Array, Array, Array, Array]:
        """Prepares training and validation data."""
        if X_val is None or y_val is None:
            X_tr, X_val2, y_tr, y_val2 = cast(
                tuple[Array, Array, Array, Array],
                train_test_split(
                    X, y, test_size=self.val_size, random_state=self.random_state
                ),
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

        return X_tr, y_tr, X_val, y_val

    def _transform_data(self, X_tr: Array, X_val: Array) -> tuple[Array, Array]:
        """Initalise transformer. Scale, and angualr transform X_tr, and X_val."""
        self.transformer_ = self._make_transformer()
        theta_tr = self.transformer_.fit_transform(X_tr)
        theta_val = self.transformer_.transform(X_val)

        return theta_tr, theta_val

    def _compute_feature_importance(self, theta_tr: Array, y_tr: Array) -> None:
        """If use_importance_ordering, compute_initial feature importance."""
        if self.use_importance_ordering:
            self.feature_importance_ = _compute_initial_importance(theta_tr, y_tr)
        else:
            self.feature_importance_ = None

    def _save_learned_state(
        self, paths: List[MVec], lambda_star: float, coeffs: Array
    ) -> None:
        """Save indices, lambda* and coefficents to self."""
        self.selected_paths_ = paths
        self.lambda_ = lambda_star
        self.coef_ = coeffs.astype(float, copy=False)

    def _cache_ray_structures(self, paths: List[MVec]) -> None:
        """Cache ray structures in self to improve inference speed."""
        self.p_mat_, self.r_arr_ = _path_matrix_and_r_arr(paths)

    def _calculate_coeffs(
        self, theta: Array, y: Array, paths: List[MVec], lambda_star: float
    ) -> tuple[float, Array]:
        """Compute normal equation, solve it, return coefficients + time taken."""
        t2 = time.perf_counter()
        gram_matrix, target_col = self._compute_normal_eqn(theta, y, paths)
        coefficients = self._solve_normal_eqn(gram_matrix, target_col, lambda_star)
        t3 = time.perf_counter()
        return t3-t2, coefficients

    def fit(
        self,
        X: Array,
        y: Array,
        *,
        X_val: Array | None = None,
        y_val: Array | None = None
    ) -> "SpectralPathRegressor":
        """Fit the model."""
        # Preparation
        X = np.asarray(X)
        y = np.asarray(y, dtype=float).ravel()
        check_dimensions(X,y)
        self.n_features_in_ = cast(int, X.shape[1])
        k_max = max(self.k_values)
        if k_max > self.n_features_in_:
            raise ValueError(
                f"Invalid k_values: max(k_values)={k_max} exceeds n_features."
            )
        X_tr, y_tr, X_val, y_val = self.prepare_train_val_split(X,y,X_val,y_val)

        theta_tr, theta_val = self._transform_data(X_tr, X_val)

        self._compute_feature_importance(theta_tr, y_tr)

        paths, lambda_star, stats = self._select_paths_and_lambda(
            theta_tr, y_tr, theta_val, y_val
        )

        theta_all = cast(Array, np.vstack([theta_tr, theta_val]))
        y_all = cast(Array, np.concatenate([y_tr, y_val]))

        solve_time, coefficients = self._calculate_coeffs(
            theta_all, y_all, paths, lambda_star
        )

        self._save_learned_state(paths, lambda_star, coefficients)
        self._cache_ray_structures(paths)

        feature_importance = self._compute_feature_importance_from_model()

        self.fit_report_ = FitReport(
            lambda_star=lambda_star,
            selected_count=len(paths),
            greedy_time_sec=stats.time_taken,
            final_solve_time_sec=solve_time,
            history=stats.history,
            stopped_early=stats.stopped_early,
            feature_importance=feature_importance,
        )

        return self

    def predict(self, X: Array) -> Array:
        """
        Predict target values for input samples.

        Args:
            X (Array): Input samples with shape (n_samples, n_features).

        Returns:
            Predicted target values with shape (n_samples,).
        """
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

        yhat = self._stream_predict(theta)
        return yhat

    def score(self, X: Array, y: Array) -> float:
        """Return the R^2 score on the given data."""
        y = np.asarray(y, dtype=float).ravel()
        yhat = self.predict(X)
        return r2_score(y, yhat)

    def _compute_feature_importance_from_model(self) -> Array:
        """
        Compute feature importance from selected paths & coefficients.

        Returns an Array of size D. Each elemnt is between 1 and 0 and is the
        importance of a feature.
        """
        if (
            self.selected_paths_ is None or self.coef_ is None or
            self.n_features_in_ is None
        ):
            raise ValueError("Selected indices or coeffs or n_features are None")

        importance = np.zeros(self.n_features_in_, dtype=float)

        # Loop through paths
        for idx, path in enumerate(self.selected_paths_):
            coefficient = abs(self.coef_[idx + 1])  # +1 for intercept
            for j, m_j in enumerate(path):
                if m_j != 0:
                    importance[j] += coefficient * abs(m_j)

        # Normalize
        if importance.sum() > 0:
            importance = importance / importance.sum()

        return importance

    def _balanced_compositions(self, L: int, r: int) -> List[List[int]]:
        comps = []

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

    def _path_generator(self, k: int) -> Iterator[MVec]:
        """Generate k-sparse paths, optionally prioritizing high-importance features."""
        D = self.n_features_in_
        if D is None:
            raise ValueError("D is not set yet, cannot create path generators.")
        if self.feature_importance_ is not None and self.use_importance_ordering:
            # Sort features by importance (descending)
            sorted_feats = np.argsort(-self.feature_importance_)
        else:
            sorted_feats = np.arange(D)

        L = 1
        while self.l_max is None or L <= self.l_max:
            for comp in self._balanced_compositions(L, k):
                for S in combinations(sorted_feats, k):
                    m = [0] * D
                    for idx, val in zip(S, comp, strict=True):
                        m[idx] = val
                    yield tuple(m)
            L += 1

    def _calc_scaling_vector(self, G: Array) -> Array:
        # s_j = ||Phi_j||_2 = sqrt(G_jj)
        s = np.sqrt(np.maximum(np.diag(G), 0.0))
        s = np.where(s < self.eps_col_norm, 1.0, s)
        if not self.normalize_intercept and s.size > 0:
            s[0] = 1.0
        return s

    def _solve_normal_eqn(self, G: Array, b: Array, lambda_: float) -> Array:
        """
        Solve ridge from Gram, optionally with implicit column normalization.

        If normalized:
            Phi_tilde = Phi * diag(1/s)
            Solve (G_tilde + lam I) w_tilde = b_tilde
            Return w = (1/s) * w_tilde (so predictions use original Phi).
        """
        if self.normalize_columns:
            scaling_vector = self._calc_scaling_vector(G)
            inv_s = 1.0 / scaling_vector
            G = (inv_s[:, None] * G) * inv_s[None, :]
            b = inv_s * b

            evals, U = np.linalg.eigh(G)
            coeffs = self._ridge_solve_for_coeffs(evals, U, b, lambda_)
            coeffs = inv_s * coeffs

            return coeffs

        evals, U = np.linalg.eigh(G)
        coeffs = self._ridge_solve_for_coeffs(evals, U, b, lambda_)
        return coeffs

    def _ridge_solve_for_coeffs(
        self, evals: Array, U: Array, b: Array, lam: float
    ) -> Array:
        """
        Solve the ridge linear system using a precomputed eigendecomposition.

        Given G = U diag(evals) U^T, this returns
            w = (G + lam I)^(-1) b
        by applying the spectral form
            w = U diag(1 / (evals + lam)) U^T b.

        Args:
            evals: Eigenvalues of the symmetric Gram matrix G.
            U: Orthonormal eigenvectors of G (columns of U).
            b: Right-hand-side vector.
            lam: Nonnegative ridge regularization strength.

        Returns:
            Coefficient vector w solving (G + lam I) w = b.
        """
        UTb = U.T @ b
        inv_diag = 1.0 / (evals + lam)
        return U @ (inv_diag * UTb)

    def _compute_normal_eqn(
        self, theta: Array, y: Array, paths: Sequence[MVec]
    ) -> Tuple[Array, Array]:
        """
        Build normal-equation terms given a lsit of paths, in streaming batches.

        Constructs the feature map implied by `paths` (with intercept) and accumulates:
            G = Phi^T Phi
            b = Phi^T y
        across row batches of size `self.batch_rows`.

        Args:
            theta (Array): Angular-transformed inputs with shape (N, D).
            y (Array): Target vector with shape (N,).
            paths (Sequence[Tuple[int, ...]]): Selected spectral paths defining feature
                columns (excluding intercept).

        Returns:
            out (Tuple[Array, Array]): A tuple `(G, b)` where:
            - `G` has shape (M, M) with `M = 1 + len(paths)`,
            - `b` has shape (M,).
        """
        path_matrix, orders = _path_matrix_and_r_arr(paths)
        M = 1 + len(paths)

        # Initialize Gram matrix and target cold as empty arrays
        G = np.zeros((M, M), dtype=float)
        b = np.zeros(M, dtype=float)

        N = theta.shape[0]
        for start in range(0, N, self.batch_rows):
            end = min(N, start + self.batch_rows)
            theta_batch = self._batch(theta, start, end)
            Phi_b = _build_feature_matrix(theta_batch, path_matrix, orders, True)
            y_batch = y[start:end]
            G += Phi_b.T @ Phi_b
            b += Phi_b.T @ y_batch

        return G, b

    def _stream_predict(
        self,
        theta: Array,
        coeffs: Array | None = None,
        p_mat: Array | None = None,
        r_arr: Array | None = None,
    ) -> Array:
        if p_mat is None or r_arr is None:
            if self.p_mat_ is not None and self.r_arr_ is not None:
                p_mat = self.p_mat_
                r_arr = self.r_arr_
            else:
                raise ValueError("p_mat and r_arr not parsed as args or in self")
        if coeffs is None:
            if self.coef_ is not None:
                coeffs = self.coef_
            else:
                raise ValueError("No coefficicnets found in self nor args")

        N = theta.shape[0]
        yhat = np.empty(N, dtype=float)

        for start in range(0, N, self.batch_rows):
            end = min(N, start + self.batch_rows)
            theta_batch = self._batch(theta, start, end)
            Phi_b = _build_feature_matrix(theta_batch, p_mat, r_arr)
            yhat[start:end] = Phi_b @ coeffs

        return yhat

    def _select_lambda_from_gram(
        self,
        G_tr: Array,
        b_tr: Array,
        theta_val: Array,
        y_val: Array,
        paths: Sequence[MVec],
    ) -> float:
        best_r2 = -1e18
        best_lam = self.lambda_grid[0]

        # Precompute structures once for validation prediction speed
        p_mat, r_arr = _path_matrix_and_r_arr(paths)

        # Cache eigendecomp for lambda sweep
        if self.normalize_columns:
            s = self._calc_scaling_vector(G_tr)
            inv_s = 1.0 / s
            Gs = (inv_s[:, None] * G_tr) * inv_s[None, :]
            bs = inv_s * b_tr
            evals, U = np.linalg.eigh(Gs)

            def solve_for_coeffs(lam_val: float) -> Array:
                w_tilde = self._ridge_solve_for_coeffs(evals, U, bs, lam_val)
                return inv_s * w_tilde

        else:
            evals, U = np.linalg.eigh(G_tr)

            def solve_for_coeffs(lam_val: float) -> Array:
                return self._ridge_solve_for_coeffs(evals, U, b_tr, lam_val)

        for lam in self.lambda_grid:
            w = solve_for_coeffs(lam)
            y_val_hat = self._stream_predict(
                theta=theta_val, coeffs=w, p_mat=p_mat, r_arr=r_arr
            )
            _, r2v = _metrics(y_val, y_val_hat)
            if r2v > best_r2:
                best_r2 = r2v
                best_lam = lam

        return best_lam

    def _select_paths_and_lambda(
        self, theta_tr: Array, y_tr: Array, theta_val: Array, y_val: Array,
    ) -> Tuple[List[MVec], float, Stats]:
        """Greedily select k-sparse path features."""
        t0 = time.perf_counter()

        # Set up
        generators = {k: self._path_generator(k) for k in self.k_values}
        selected_paths: List[MVec] = []
        history: List[Tuple[int, int, float, float]] = []
        lambda_star: float | None = None
        best_r2_score_overall = -1e18
        no_improve_count = 0
        stopped_early = False
        current_block_size = self.block_size
        Ntr: int = theta_tr.shape[0]
        G_old = np.array([[Ntr]], dtype=float)
        b_old = np.array([float(y_tr.sum())], dtype=float)

        while len(selected_paths) < self.max_paths:
            remaining_number_of_paths = self.max_paths - len(selected_paths)
            block_size = min(current_block_size, remaining_number_of_paths)

            # Set up candidates (Dict mapping k vals to candidate paths)
            candidates = self._generate_candidates(generators, block_size)

            if not candidates:
                self._log("All generators exhausted before reaching max_paths.")
                break

            # Generate "old" data
            if not selected_paths:
                p_mat_old = np.empty((0, theta_tr.shape[1]))
                r_arr_old = np.empty((0,), dtype=np.int64)
            else:
                p_mat_old, r_arr_old = _path_matrix_and_r_arr(selected_paths)
            M_old = 1 + len(selected_paths)

            # Generate maps & candidate structures
            cand_struct, C_map, Gnew_map, bnew_map = self._init_candidate_evaluation(
                candidates, M_old
            )

            # Loop through training data batch by batch to build C, G, and b
            C_map, Gnew_map, bnew_map = self._build_c_g_and_b_via_stream(
                y_tr, theta_tr, p_mat_old, r_arr_old, cand_struct,
                C_map, Gnew_map, bnew_map
            )

            # Evaluate candidates (solve on train using block Gram; score on val)
            best_r2_score = -1e18
            best_choice = None

            for k, paths in candidates.items():
                C = C_map[k]
                Gnew = Gnew_map[k]
                bnew = bnew_map[k]

                # Build trial data
                G_trial = np.block([[G_old, C], [C.T,  Gnew]])
                b_trial = np.concatenate([b_old, bnew])
                trial_paths = selected_paths + paths
                p_mat_trial, r_arr_trial = _path_matrix_and_r_arr(trial_paths)

                # Solve (possibly with lambda sweep on first commit)
                if lambda_star is None:
                    best_r2_this = -1e18
                    best_lam_this = self.lambda_grid[0]

                    if self.normalize_columns:
                        scaling_vector = self._calc_scaling_vector(G_trial)
                        inv_s_trial = 1.0 / scaling_vector
                        Gs_trial = (
                            (inv_s_trial[:, None] * G_trial) * inv_s_trial[None, :]
                        )
                        scaled_b_trial = inv_s_trial * b_trial
                        scaled_evals_trial, U_trial = np.linalg.eigh(Gs_trial)
                        for lambda_ in self.lambda_grid:
                            scaled_coeffs = self._ridge_solve_for_coeffs(
                                scaled_evals_trial, U_trial, scaled_b_trial, lambda_
                            )
                            normalized_coeffs = scaled_coeffs * inv_s_trial
                            y_val_hat = self._stream_predict(
                                theta_val, normalized_coeffs, p_mat_trial, r_arr_trial
                            )
                            _, r2v = _metrics(y_val, y_val_hat)
                            if r2v > best_r2_this:
                                best_r2_this = r2v
                                best_lam_this = lambda_
                    else:
                        evals_trial, U_trial = np.linalg.eigh(G_trial)
                        for lambda_ in self.lambda_grid:
                            coeffs = self._ridge_solve_for_coeffs(
                                evals_trial, U_trial, b_trial, lambda_
                            )
                            y_val_hat = self._stream_predict(
                                theta_val, coeffs, p_mat_trial, r_arr_trial
                            )
                            _, r2v = _metrics(y_val, y_val_hat)
                            if r2v > best_r2_this:
                                best_r2_this = r2v
                                best_lam_this = lambda_

                    cand_r2_score = best_r2_this
                    cand_lambda = best_lam_this
                else:
                    coeffs = self._solve_normal_eqn(G_trial, b_trial, lambda_star)
                    y_val_hat = self._stream_predict(
                        theta_val, coeffs, p_mat_trial, r_arr_trial
                    )
                    _, cand_r2_score = _metrics(y_val, y_val_hat)
                    cand_lambda = lambda_star

                if cand_r2_score > best_r2_score:
                    best_r2_score = cand_r2_score
                    best_choice = (k, paths, G_trial, b_trial, cand_lambda)

            # Update with new best things
            assert best_choice is not None # (for type checkers)
            k_win, block_win, G_new_old, b_new_old, lam_win = best_choice
            selected_paths.extend(block_win)
            G_old = G_new_old
            b_old = b_new_old
            if lambda_star is None:
                lambda_star = lam_win
            history.append((k_win, len(block_win), lambda_star, best_r2_score))

            # Early stopping check
            improving = best_r2_score > best_r2_score_overall + self.early_stopping_tol
            if improving:
                best_r2_score_overall = best_r2_score
                no_improve_count = 0

                # Increase block size if doing well
                if self.adaptive_block_size and current_block_size < self.block_size:
                    current_block_size = min(self.block_size, current_block_size + 1)
            else:
                no_improve_count += 1

                # Given we're not improving, decrease block size if allowed
                if (
                    current_block_size > self.min_block_size
                    and self.adaptive_block_size
                    ):
                    current_block_size=max(self.min_block_size, current_block_size - 1)

                if no_improve_count >= self.early_stopping_patience:
                    self._log(
                        f"[Early stopping] No improvement for "
                        f"{self.early_stopping_patience} rounds at "
                        f"{len(selected_paths)} paths"
                    )
                    stopped_early = True
                    break
            self._log(
                f"[Greedy] Added k={k_win} block of {len(block_win)} → total="
                f"{len(selected_paths)} | λ_used={lambda_star} | R²_val="
                f"{best_r2_score:0.4f} | block_size={current_block_size}"
            )

        if lambda_star is None:
            lambda_star = self.lambda_grid[0]

        # Optional re-sweep
        if self.final_lambda_refit and len(self.lambda_grid) > 0:
            lambda_star = self._select_lambda_from_gram(
                G_old, b_old, theta_val, y_val, selected_paths
            )
        t1 = time.perf_counter()

        stats = Stats(stopped_early=stopped_early, history=history, time_taken=t1-t0)
        return selected_paths, lambda_star, stats

    # ------------------- Misc -------------------
    def _build_c_g_and_b_via_stream(
        self,
        y_tr: Array,
        theta_tr: Array,
        p_mat_old: Array,
        r_arr_old: Array,
        cand_struct: Dict[int, Tuple[Array, Array]],
        C_map: Dict[int, Array],
        Gnew_map: Dict[int, Array],
        bnew_map: Dict[int, Array]
    ) -> tuple[Dict[int, Array], Dict[int, Array], Dict[int, Array]]:
        Ntr = theta_tr.shape[0]
        for start in range(0, Ntr, self.batch_rows):
            end = min(Ntr, start + self.batch_rows)

            y_batch = self._batch(y_tr, start, end)
            theta_batch = self._batch(theta_tr, start, end)
            Phi_old_batch = _build_feature_matrix(theta_batch, p_mat_old, r_arr_old)

            # Loop through k values, e.g., 1, 2, 3
            for k, (p_mat_block, r_arr_block) in cand_struct.items():
                Phi_new_batch = _build_feature_matrix(
                    theta_batch, p_mat_block, r_arr_block, False
                )

                C_map[k] += Phi_old_batch.T @ Phi_new_batch
                Gnew_map[k] += Phi_new_batch.T @ Phi_new_batch
                bnew_map[k] += Phi_new_batch.T @ y_batch

        return C_map, Gnew_map, bnew_map
    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def _batch(self, theta_tr: Array, start: int, end: int) -> Array:
        """Batching helper."""
        return theta_tr[start:end].astype(
            np.float32 if self.use_float32 else np.float64, copy=False
        )

    def _generate_candidates(
        self, generators: Dict[int, Iterator[MVec]], block_size: int
    ) -> Dict[int, List[MVec]]:
        """Helper function which generates block_size worth of candidate paths."""
        candidates: Dict[int, List[MVec]] = {}
        for k in self.k_values:
            block: List[MVec] = []
            for _ in range(block_size):
                try:
                    block.append(next(generators[k]))
                except StopIteration:
                    break
            if block:
                candidates[k] = block
        return candidates

    def _init_candidate_evaluation(
        self, candidates: Dict[int, List[MVec]], M_old: int,
    ) -> Tuple[
            Dict[int, Tuple[Array, Array]],
            Dict[int, Array], Dict[int, Array], Dict[int, Array]]:
        """
        Generates maps for greedy algorithm.

        Args:
            candidates (Dict[int, List[tuple(int, ...)]]): Dictionary mapping k values
                to candidate paths.
            M_old (int): Current number of paths in path matrix.

        Returns:
            cand_struct (Dict[int, Tuple[Array, Array]]): dictionary where each key is a
                k_value, and value is (path matrix, order array) tuple for a block.
            C_map (Dict[int, Array]): Map between k_values and zero arrays.
        """
        cand_struct: Dict[int, Tuple[Array, Array]] = {}
        C_map: Dict[int, Array] = {}
        Gnew_map: Dict[int, Array] = {}
        bnew_map: Dict[int, Array] = {}
        for k, block in candidates.items():
            cand_struct[k] = _path_matrix_and_r_arr(block)
            Qnew = len(block)
            C_map[k] = np.zeros((M_old, Qnew), dtype=float)
            Gnew_map[k] = np.zeros((Qnew, Qnew), dtype=float)
            bnew_map[k] = np.zeros(Qnew, dtype=float)
        return cand_struct, C_map, Gnew_map, bnew_map
