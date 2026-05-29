"""Spectral path regression and classification models."""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from itertools import combinations
from typing import Callable, Dict, Iterator, List, Literal, Sequence, Tuple, TypedDict

import numpy as np
from sklearn.metrics import r2_score
from threadpoolctl import threadpool_limits

from spectral_paths.schemas import BlasThreadInfo, FitReport, PhaseTimings, ScalerType, Stats
from spectral_paths.types import Array, MVec
from spectral_paths.utils.helpers import (
    _binary_accuracy,
    _binary_log_loss,
    _build_feature_matrix,
    _clip_probabilities,
    _compute_initial_importance,
    _metrics,
    _path_matrix_and_r_arr,
    _sigmoid,
    check_dimensions,
    train_test_split,
)
from spectral_paths.utils.preprocessing import AngularTransformer


class EquationTerm(TypedDict):
    """Typed representation of one ranked symbolic term."""

    path: MVec
    coef: float
    energy: float


class CandidateEvaluation(TypedDict):
    """Typed representation of one greedy candidate-block evaluation."""

    k: int
    paths: List[MVec]
    G_trial: Array
    b_trial: Array
    cand_lambda: float
    cand_r2_score: float


class ClassifierCandidateEvaluation(TypedDict):
    """Typed representation of one greedy classifier candidate-block evaluation."""

    k: int
    paths: List[MVec]
    cand_lambda: float
    cand_log_loss: float


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
        lambda_parallel_workers: int = 1,
        blas_thread_policy: Literal["auto", "none", "single", "manual"] = "auto",
        blas_threads: int | None = None,
        early_stopping_patience: int = 3,
        early_stopping_tol: float = 1e-4,
        greedy_subsample: float | int | None = None,

        # -------- scaler config --------
        scaler_type: ScalerType | str = ScalerType.STANDARD_PERCENTILE_MINMAX,
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
            lambda_parallel_workers (int): Number of threads used for lambda scoring
                after a shared eigendecomposition. Values <= 1 disable outer
                parallelism to avoid nested oversubscription by default.
            blas_thread_policy (Literal["auto", "none", "single", "manual"]):
                Policy controlling BLAS/OpenMP thread limits during fit-time hot paths.
                ``"auto"`` resolves from workload shape, ``"none"`` leaves threadpools
                untouched, ``"single"`` forces one BLAS thread, and ``"manual"``
                requires `blas_threads`.
            blas_threads (int | None): Explicit BLAS thread cap used when
                `blas_thread_policy="manual"`.
            early_stopping_patience (int): Number of consecutive non-improving rounds
                tolerated before early stopping of greedy path selection.
            early_stopping_tol (float): Minimum validation improvement required to reset
                early stopping.
            greedy_subsample (float | int | None): Optional subsample size used for
                greedy path evaluation on the training split. Floats are interpreted
                as fractions in (0, 1], integers as sample counts, and None uses the
                full training data.
            scaler_type (ScalerType | str): Scaling strategy used to map inputs to
                the interval [-1, 1] prior to the angular transformation.
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
        self.lambda_parallel_workers = int(lambda_parallel_workers)
        self.blas_thread_policy = str(blas_thread_policy)
        self.blas_threads = None if blas_threads is None else int(blas_threads)
        self.early_stopping_patience = int(early_stopping_patience)
        self.early_stopping_tol = float(early_stopping_tol)
        self.greedy_subsample = greedy_subsample
        self._internal_dtype = np.float32 if self.use_float32 else np.float64
        self._resolved_blas_threads_: int | None = None

        if self.lambda_parallel_workers < 1:
            raise ValueError("lambda_parallel_workers must be >= 1.")
        if self.blas_thread_policy not in {"auto", "none", "single", "manual"}:
            raise ValueError(
                "blas_thread_policy must be one of 'auto', 'none', 'single', or 'manual'."
            )
        if self.blas_thread_policy == "manual":
            if self.blas_threads is None or self.blas_threads < 1:
                raise ValueError(
                    "blas_threads must be a positive integer when blas_thread_policy='manual'."
                )
        elif self.blas_threads is not None and self.blas_threads < 1:
            raise ValueError("blas_threads must be >= 1 when provided.")

        if greedy_subsample is not None:
            if isinstance(greedy_subsample, float):
                if not (0.0 < greedy_subsample <= 1.0):
                    raise ValueError("greedy_subsample float values must lie in (0, 1].")
            elif isinstance(greedy_subsample, int):
                if greedy_subsample < 1:
                    raise ValueError("greedy_subsample integer values must be >= 1.")
            else:
                raise ValueError("greedy_subsample must be a float, int, or None.")

        try:
            self.scaler_type = ScalerType(scaler_type)
        except ValueError as exc:
            allowed = ", ".join(mode.value for mode in ScalerType)
            raise ValueError(
                f"Unsupported scaler_type {scaler_type!r}. Allowed values: {allowed}."
            ) from exc
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
        """Construct an AngularTransformer instance from this model's scaler config."""
        return AngularTransformer(
            scaler=self.scaler_type,
            percentile_range=self.iqr_percentile_range,
            bound_percentiles=self.bound_percentiles,
            eps=self.eps_col_norm,
        )

    def _prepare_train_val_split(
        self, X: Array, y: Array, X_val: Array | None, y_val: Array | None
    ) -> tuple[Array, Array, Array, Array]:
        """Prepares training and validation data."""
        if X_val is None or y_val is None:
            X_tr, X_val2, y_tr, y_val2 = train_test_split(X, y, self.val_size, self.random_state)
            X_val, y_val = X_val2, y_val2

        else:
            X_tr, y_tr = X, y
            y_val = np.asarray(y_val, dtype=float).ravel()

            if X_val.ndim != 2 or X_val.shape[1] != X_tr.shape[1]:
                raise ValueError("X_val must be 2D with same number of columns as X.")
            if X_val.shape[0] != y_val.shape[0]:
                raise ValueError("X_val rows must match y_val length.")

        return X_tr, y_tr, X_val, y_val

    def _transform_data(self, X_tr: Array, X_val: Array) -> tuple[Array, Array]:
        """Initalise transformer. Scale, and angualr transform X_tr, and X_val."""
        self.transformer_ = self._make_transformer()
        θ_tr = self.transformer_.fit_transform(X_tr)
        θ_val = self.transformer_.transform(X_val)

        return θ_tr, θ_val

    def _as_internal_dtype(self, arr: Array) -> Array:
        """Convert arrays once to the model's internal dtype."""
        return np.asarray(arr, dtype=self._internal_dtype)

    def _compute_feature_importance(self, θ_tr: Array, y_tr: Array) -> None:
        """If use_importance_ordering, compute_initial feature importance."""
        if self.use_importance_ordering:
            self.feature_importance_ = _compute_initial_importance(θ_tr, y_tr)
        else:
            self.feature_importance_ = None

    def _save_learned_state(self, paths: List[MVec], lambda_star: float, coeffs: Array) -> None:
        """Save indices, lambda* and coefficents to self."""
        self.selected_paths_ = paths
        self.lambda_ = lambda_star
        self.coef_ = coeffs.astype(float, copy=False)

    def _cache_ray_structures(self, paths: List[MVec]) -> None:
        """Cache ray structures in self to improve inference speed."""
        p_mat, r_arr = _path_matrix_and_r_arr(paths)
        self.p_mat_ = self._as_internal_dtype(p_mat)
        self.r_arr_ = r_arr

    def _calculate_coeffs(
        self, θ: Array, y: Array, paths: List[MVec], lambda_star: float
    ) -> tuple[float, float, Array]:
        """Compute normal equation, solve it, and return timings plus coefficients."""
        t2 = time.perf_counter()
        gram_matrix, target_col = self._compute_normal_eqn(θ, y, paths)
        t3 = time.perf_counter()
        coefficients = self._solve_normal_eqn(gram_matrix, target_col, lambda_star)
        t4 = time.perf_counter()
        return t3 - t2, t4 - t3, coefficients

    def _subsample_greedy_training_data(self, θ_tr: Array, y_tr: Array) -> tuple[Array, Array]:
        """Optionally subsample training data for greedy path evaluation."""
        if self.greedy_subsample is None:
            return θ_tr, y_tr

        n_train = θ_tr.shape[0]
        if isinstance(self.greedy_subsample, int):
            sample_size = min(self.greedy_subsample, n_train)
        else:
            sample_size = min(n_train, max(1, int(np.ceil(self.greedy_subsample * n_train))))

        if sample_size >= n_train:
            return θ_tr, y_tr

        rng = np.random.default_rng(self.random_state)
        indices = np.sort(rng.choice(n_train, size=sample_size, replace=False))
        self._log(
            f"[Greedy] Using training subsample of {sample_size}/{n_train} rows "
            "for candidate evaluation"
        )
        return θ_tr[indices], y_tr[indices]

    def fit(
        self, X: Array, y: Array, *, X_val: Array | None = None, y_val: Array | None = None
    ) -> "SpectralPathRegressor":
        """Fit the model."""
        t_fit_start = time.perf_counter()
        X = np.asarray(X)
        y = np.asarray(y, dtype=float).ravel()
        check_dimensions(X,y)

        self.n_features_in_ = int(X.shape[1])
        k_max = max(self.k_values)
        if k_max > self.n_features_in_:
            raise ValueError(
                f"Invalid k_values: max(k_values)={k_max} exceeds n_features."
            )

        X_tr, y_tr, X_val, y_val = self._prepare_train_val_split(X, y, X_val, y_val)
        t_pre_start = time.perf_counter()
        θ_tr, θ_val = self._transform_data(X_tr, X_val)
        θ_tr = self._as_internal_dtype(θ_tr)
        θ_val = self._as_internal_dtype(θ_val)
        y_tr = self._as_internal_dtype(y_tr)
        y_val = self._as_internal_dtype(y_val)
        preprocessing_time = time.perf_counter() - t_pre_start

        self._compute_feature_importance(θ_tr, y_tr)
        θ_tr_greedy, y_tr_greedy = self._subsample_greedy_training_data(θ_tr, y_tr)
        resolved_blas_threads = self._resolve_blas_threads(θ_tr_greedy.shape[0])
        self._resolved_blas_threads_ = resolved_blas_threads
        self._log(
            "[BLAS] policy="
            f"{self.blas_thread_policy} resolved_threads={resolved_blas_threads}"
        )

        with self._blas_thread_limit_context(resolved_blas_threads):
            paths, lambda_star, stats = self._select_paths_and_lambda(
                θ_tr_greedy, y_tr_greedy, θ_val, y_val
            )

        θ_all = np.vstack([θ_tr, θ_val])
        y_all =  np.concatenate([y_tr, y_val])
        with self._blas_thread_limit_context(resolved_blas_threads):
            normal_eqn_time, solve_time, coefficients = self._calculate_coeffs(
                θ_all, y_all, paths, lambda_star
            )

        self._save_learned_state(paths, lambda_star, coefficients)
        self._cache_ray_structures(paths)

        feature_importance = self._compute_feature_importance_from_model()

        phase_timings = PhaseTimings(
            preprocessing_sec=preprocessing_time,
            greedy_accumulation_sec=stats.accumulation_time_sec,
            greedy_scoring_sec=stats.scoring_time_sec,
            lambda_sweep_sec=stats.lambda_sweep_time_sec,
            final_normal_eqn_sec=normal_eqn_time,
            final_solve_sec=solve_time,
            total_fit_sec=time.perf_counter() - t_fit_start,
        )

        self.fit_report_ = FitReport(
            lambda_star=lambda_star,
            selected_count=len(paths),
            greedy_time_sec=stats.time_taken,
            final_solve_time_sec=solve_time,
            history=stats.history,
            stopped_early=stats.stopped_early,
            feature_importance=feature_importance,
            phase_timings=phase_timings,
            blas_threads=BlasThreadInfo(
                policy=self.blas_thread_policy,
                resolved_threads=resolved_blas_threads,
            ),
        )

        return self

    def predict(self, X: Array) -> Array:
        """Predict target values for input samples."""
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.n_features_in_}"
            )

        if self.transformer_ is None:
            raise ValueError("Transformer has not been fitted yet")

        θ = self._as_internal_dtype(self.transformer_.transform(X))
        yhat = self._stream_predict(θ)
        return yhat

    def score(self, X: Array, y: Array) -> float:
        """Return the R^2 score on the given data."""
        y = np.asarray(y, dtype=float).ravel()
        yhat = self.predict(X)
        return r2_score(y, yhat)

    def _check_is_fitted(self) -> None:
        """Ensure fitted model state required for prediction-style methods exists."""
        if self.transformer_ is None or self.coef_ is None:
            raise ValueError("Model is not fitted yet. Call fit(X, y) first.")
        if self.n_features_in_ is None:
            raise ValueError("Model is missing n_features_in_. Call fit(X, y) first.")
        if self.p_mat_ is None or self.r_arr_ is None:
            raise ValueError("Cached path structures are unavailable. Call fit(X, y) first.")

    def input_gradients(self, X: Array) -> Array:
        """
        Compute analytic raw-input gradients of the fitted predictor.

        For the fitted model

            y_hat(theta) = c0 + sum_q A_q cos(m_q^T theta),

        the chain rule gives, for each feature j,

            d y_hat / d x_j
            = - (d theta_j / d x_j) * sum_q [A_q * m_qj * sin(m_q^T theta)].

        When the scaler is ``*_tanh``, the preprocessing is

            z_j = tanh((x_j - c_j) / s_j),    theta_j = arccos(z_j),

        so

            d theta_j / d x_j = -sech((x_j - c_j) / s_j) / s_j,

        and the expression simplifies to the form requested in the plotting
        workflow. This implementation evaluates the analytic chain rule exactly
        through the fitted transformer, without finite differences.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.n_features_in_}"
            )

        assert self.transformer_ is not None
        assert self.coef_ is not None
        assert self.p_mat_ is not None
        assert self.r_arr_ is not None

        theta, dtheta_dx = self.transformer_.transform_with_jacobian(X)

        # Each learned basis function is cos(r_q * <theta, p_q>) with m_q = r_q p_q.
        phase = (theta @ self.p_mat_.T) * self.r_arr_[None, :]
        sin_phase = np.sin(phase)
        m_matrix = self.p_mat_ * self.r_arr_[:, None]
        phase_sum = (sin_phase * self.coef_[1:][None, :]) @ m_matrix

        return -dtheta_dx * phase_sum

    def feature_importance_gradient(
        self,
        X: np.ndarray,
        normalize: bool = True,
        as_percentage: bool = True,
        aggregate: str = "mean_abs",
    ) -> np.ndarray:
        """
        Aggregate raw-input gradient magnitudes into feature importances.

        Args:
            X: Raw input matrix in the original feature units.
            normalize: Whether to normalize the aggregated importances.
            as_percentage: If ``True`` with normalization enabled, return values that
                sum to 100.
            aggregate: Aggregation rule. Supported values are ``"mean_abs"`` and
                ``"rms"``.
        """
        if as_percentage and not normalize:
            raise ValueError("as_percentage=True requires normalize=True.")

        gradients = self.input_gradients(X)

        if aggregate == "mean_abs":
            importance = np.mean(np.abs(gradients), axis=0)
        elif aggregate == "rms":
            importance = np.sqrt(np.mean(np.square(gradients), axis=0))
        else:
            raise ValueError("aggregate must be either 'mean_abs' or 'rms'.")

        if not normalize:
            return importance

        total = float(importance.sum())
        if total <= 0.0:
            return np.zeros_like(importance)

        importance = importance / total
        if as_percentage:
            importance = 100.0 * importance
        return importance

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

    def top_equation_terms(
        self,
        *,
        n_terms: int = 12,
        energy_power: float = 2.0,
    ) -> List[EquationTerm]:
        """
        Return the top learned spectral terms ranked by coefficient magnitude.

        Terms are ranked by ``abs(coef) ** energy_power`` and returned as dictionaries
        containing the path, coefficient, and ranking energy.
        """
        if self.selected_paths_ is None or self.coef_ is None:
            raise ValueError("Model must be fitted before extracting symbolic terms.")

        ranked_terms: List[EquationTerm] = []
        for idx, path in enumerate(self.selected_paths_):
            coefficient = float(self.coef_[idx + 1])  # intercept is coef_[0]
            ranked_terms.append(
                {
                    "path": path,
                    "coef": coefficient,
                    "energy": abs(coefficient) ** float(energy_power),
                }
            )

        ranked_terms.sort(key=lambda term: float(term["energy"]), reverse=True)
        return ranked_terms[: int(n_terms)]

    def _format_equation_term(
        self,
        path: MVec,
        coefficient: float,
        feature_names: Sequence[str] | None,
    ) -> str:
        """Format one learned cosine term for terminal display."""
        phase_parts: List[str] = []
        for idx, path_value in enumerate(path):
            if path_value == 0:
                continue
            if feature_names is None:
                theta_name = f"theta{idx + 1}"
            else:
                theta_name = f"theta[{feature_names[idx]}]"

            if path_value == 1:
                phase_parts.append(theta_name)
            else:
                phase_parts.append(f"{path_value}*{theta_name}")

        phase = " + ".join(phase_parts) if phase_parts else "0"
        return f"{coefficient:+.6g} * cos({phase})"

    def format_top_equation(
        self,
        *,
        n_terms: int = 12,
        feature_names: Sequence[str] | None = None,
        include_intercept: bool = True,
    ) -> str:
        """
        Format a truncated symbolic model string for terminal display.

        The printed variables are angular coordinates, where
        ``theta[j] = acos(transformed_feature[j])`` after the model's fitted scaler.
        """
        if self.coef_ is None:
            raise ValueError("Model must be fitted before formatting an equation.")

        pieces: List[str] = []
        if include_intercept:
            pieces.append(f"{float(self.coef_[0]):.6g}")

        for term in self.top_equation_terms(n_terms=n_terms):
            pieces.append(
                self._format_equation_term(
                    term["path"],
                    float(term["coef"]),
                    feature_names,
                )
            )

        return "y_hat = " + " ".join(pieces)

    def print_top_equation(
        self,
        *,
        n_terms: int = 12,
        feature_names: Sequence[str] | None = None,
        include_intercept: bool = True,
    ) -> str:
        """Print a truncated symbolic model and return it as a string."""
        equation = self.format_top_equation(
            n_terms=n_terms,
            feature_names=feature_names,
            include_intercept=include_intercept,
        )

        print("\n=== Top terms (by |coef|^2) ===")
        for rank, term in enumerate(self.top_equation_terms(n_terms=n_terms), start=1):
            print(
                f"{rank:02d}. energy={float(term['energy']):.6g} | "
                f"coef={float(term['coef']):+.6g} | path={term['path']}"
            )

        print("\nNote: theta[j] = acos(transformed_feature[j]) after the fitted scaler.")
        print("\n=== Symbolic truncated model ===")
        print(equation)
        return equation

    def format_feature_transforms(
        self,
        *,
        feature_names: Sequence[str] | None = None,
    ) -> List[str]:
        """
        Format the fitted per-feature preprocessing transforms.

        Each line describes ``theta_j = acos(s_j(x_j))`` where ``s_j`` is the fitted
        scaler mapping into ``[-1, 1]`` prior to the angular transform.
        """
        if self.transformer_ is None or self.n_features_in_ is None:
            raise ValueError("Model must be fitted before formatting feature transforms.")

        transformer = self.transformer_
        mode = transformer._canonical_mode(transformer.scaler)
        lines: List[str] = []

        for idx in range(self.n_features_in_):
            name = feature_names[idx] if feature_names is not None else f"x{idx + 1}"

            if mode is ScalerType.MINMAX:
                if transformer.min_ is None or transformer.max_ is None:
                    raise ValueError("Transformer min-max parameters are unavailable.")
                min_arr = transformer.min_
                max_arr = transformer.max_
                min_ = float(min_arr[idx])
                max_ = float(max_arr[idx])
                lines.append(
                    f"theta[{name}] = acos(2 * (({name} - {min_:.6g}) / "
                    f"({max_:.6g} - {min_:.6g})) - 1)"
                )
                continue

            if transformer.center_ is None or transformer.scale_ is None:
                raise ValueError("Transformer z-score parameters are unavailable.")

            center_arr = transformer.center_
            scale_arr = transformer.scale_
            center = float(center_arr[idx])
            scale = float(scale_arr[idx])
            z_term = f"(({name} - {center:.6g}) / {scale:.6g})"

            if mode is ScalerType.STANDARD_TANH or mode is ScalerType.ROBUST_TANH:
                lines.append(f"theta[{name}] = acos(tanh{z_term})")
                continue

            if transformer.z_lo_ is None or transformer.z_hi_ is None:
                raise ValueError("Transformer percentile bounds are unavailable.")

            z_lo_arr = transformer.z_lo_
            z_hi_arr = transformer.z_hi_
            z_lo = float(z_lo_arr[idx])
            z_hi = float(z_hi_arr[idx])
            lines.append(
                ("theta[{name}] = acos(2 * (({z} - {z_lo:.6g}) / "
                "({z_hi:.6g} - {z_lo:.6g})) - 1)").format(
                    name=name,
                    z=z_term,
                    z_lo=z_lo,
                    z_hi=z_hi,
                )
            )

        return lines

    def print_feature_transforms(
        self, *, feature_names: Sequence[str] | None = None,
    ) -> List[str]:
        """Print the fitted per-feature preprocessing transforms."""
        lines = self.format_feature_transforms(feature_names=feature_names)
        print("\n=== Fitted Feature Transforms ===")
        for line in lines:
            print(line)
        return lines

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
        # s_j = ||Φ_j||_2 = sqrt(G_jj)
        s = np.sqrt(np.maximum(np.diag(G), 0.0))
        s = np.where(s < self.eps_col_norm, 1.0, s)
        if not self.normalize_intercept and s.size > 0:
            s[0] = 1.0
        return s

    def _solve_normal_eqn(self, G: Array, b: Array, lambda_: float) -> Array:
        """
        Solve ridge from Gram, optionally with implicit column normalization.

        If normalized:
            Φ_tilde = Φ * diag(1/s)
            Solve (G_tilde + lam I) w_tilde = b_tilde
            Return w = (1/s) * w_tilde (so predictions use original Φ).
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
            self, θ: Array, y: Array, paths: Sequence[MVec]
    ) -> Tuple[Array, Array]:
        """
        Build normal-equation terms given a list of paths, in streaming batches.

        Constructs the feature map implied by `paths` (with intercept) and accumulates:
            G = Φ^T Φ
            b = Φ^T y
        across row batches of size `self.batch_rows`.

        Args:
            θ (Array): Angular-transformed inputs with shape (N, D).
            y (Array): Target vector with shape (N,).
            paths (Sequence[Tuple[int, ...]]): Selected spectral paths defining feature
                columns (excluding intercept).

        Returns:
            out (Tuple[Array, Array]): A tuple `(G, b)` where:
            - `G` has shape (M, M) with `M = 1 + len(paths)`,
            - `b` has shape (M,).
        """
        path_matrix, orders = _path_matrix_and_r_arr(paths)
        path_matrix = self._as_internal_dtype(path_matrix)
        M = 1 + len(paths)

        # Initialize Gram matrix and target cold as empty arrays
        G = np.zeros((M, M), dtype=self._internal_dtype)
        b = np.zeros(M, dtype=self._internal_dtype)

        N = θ.shape[0]
        for start in range(0, N, self.batch_rows):
            end = min(N, start + self.batch_rows)

            θ_b = self._batch(θ, start, end)
            Φ_b = _build_feature_matrix(θ_b, path_matrix, orders)
            y_b = y[start:end]

            G += Φ_b.T @ Φ_b
            b = b + np.asarray(Φ_b.T @ y_b, dtype=self._internal_dtype)

        return G, b

    def _stream_predict(
        self,
        θ: Array,
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
                raise ValueError("No coefficients found in self nor args")

        N = θ.shape[0]
        yhat = np.empty(N, dtype=float)

        for start in range(0, N, self.batch_rows):
            end = min(N, start + self.batch_rows)
            θ_b = self._batch(θ, start, end)
            Φ_b = _build_feature_matrix(θ_b, p_mat, r_arr)
            yhat[start:end] = Φ_b @ coeffs

        return yhat

    def _blas_thread_env_active(self) -> bool:
        """Return whether common BLAS thread env vars are set above one thread."""
        for env_name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            raw = os.environ.get(env_name)
            if raw is None:
                continue
            try:
                if int(raw) > 1:
                    return True
            except ValueError:
                continue
        return False

    def _outer_parallelism_enabled(self) -> bool:
        """Decide whether outer Python thread parallelism should be used."""
        if self.lambda_parallel_workers <= 1:
            return False
        if len(self.lambda_grid) <= 1:
            return False
        if self._resolved_blas_threads_ is not None and self._resolved_blas_threads_ > 1:
            return False
        return not self._blas_thread_env_active()

    def _resolve_blas_threads(self, n_train_rows: int) -> int | None:
        """Resolve BLAS thread cap from policy and workload shape."""
        if self.blas_thread_policy == "none":
            return None
        if self.blas_thread_policy == "single":
            return 1
        if self.blas_thread_policy == "manual":
            assert self.blas_threads is not None
            return self.blas_threads

        assert self.n_features_in_ is not None
        width = self.n_features_in_
        max_paths = self.max_paths
        lambda_count = len(self.lambda_grid)

        # Width-dominated heuristic derived from benchmark results in this repo.
        if width >= 16:
            return 1
        if max_paths >= 256 and lambda_count >= 16:
            return 1
        if width >= 8 and max_paths >= 512:
            return 1
        if width >= 12 and n_train_rows >= 4000:
            return 1
        return None

    def _blas_thread_limit_context(self, resolved_threads: int | None):
        """Return a context manager that optionally caps BLAS threads."""
        if resolved_threads is None:
            return nullcontext()
        return threadpool_limits(limits=resolved_threads, user_api="blas")

    def _score_lambda_candidates(
        self,
        solve_for_coeffs: Callable[[float], Array],
        p_mat: Array,
        r_arr: Array,
        θ_val: Array,
        y_val: Array,
    ) -> tuple[float, float, float]:
        """Score all lambda candidates and return best lambda, score, and sweep time."""

        def evaluate_one(lam: float) -> tuple[float, float]:
            coeffs = solve_for_coeffs(lam)
            y_val_hat = self._stream_predict(
                θ=θ_val, coeffs=coeffs, p_mat=p_mat, r_arr=r_arr
            )
            _, r2v = _metrics(y_val, y_val_hat)
            return lam, r2v

        t0 = time.perf_counter()
        if self._outer_parallelism_enabled():
            max_workers = min(self.lambda_parallel_workers, len(self.lambda_grid))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(evaluate_one, self.lambda_grid))
        else:
            results = [evaluate_one(lam) for lam in self.lambda_grid]

        best_lam, best_r2 = max(results, key=lambda item: (item[1], -item[0]))
        return float(best_lam), float(best_r2), time.perf_counter() - t0

    def _select_lambda_from_gram(
        self,
        G_tr: Array,
        b_tr: Array,
        θ_val: Array,
        y_val: Array,
        paths: Sequence[MVec],
    ) -> float:
        # Precompute structures once for validation prediction speed
        p_mat, r_arr = _path_matrix_and_r_arr(paths)
        p_mat = self._as_internal_dtype(p_mat)

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

        best_lam, _, _ = self._score_lambda_candidates(
            solve_for_coeffs=solve_for_coeffs,
            p_mat=p_mat,
            r_arr=r_arr,
            θ_val=θ_val,
            y_val=y_val,
        )
        return best_lam

    def _select_paths_and_lambda(
        self, θ_tr: Array, y_tr: Array, θ_val: Array, y_val: Array,
    ) -> Tuple[List[MVec], float, Stats]:
        """Greedily select k-sparse path features."""
        t0 = time.perf_counter()
        accumulation_time = 0.0
        scoring_time = 0.0
        lambda_sweep_time = 0.0

        # Set up
        generators = {k: self._path_generator(k) for k in self.k_values}
        selected_paths: List[MVec] = []
        history: List[Tuple[int, int, float, float]] = []
        lambda_star: float | None = None
        best_r2_score_overall = -1e18
        no_improve_count = 0
        stopped_early = False
        current_block_size = self.block_size
        Ntr: int = θ_tr.shape[0]
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
                p_mat_old = np.empty((0, θ_tr.shape[1]), dtype=self._internal_dtype)
                r_arr_old = np.empty((0,), dtype=np.int64)
            else:
                p_mat_old, r_arr_old = _path_matrix_and_r_arr(selected_paths)
                p_mat_old = self._as_internal_dtype(p_mat_old)
            M_old = 1 + len(selected_paths)

            # Generate maps & candidate structures
            cand_struct, C_map, Gnew_map, bnew_map = self._init_candidate_evaluation(
                candidates, M_old
            )

            # Loop through training data batch by batch to build C, G, and b
            t_accum_start = time.perf_counter()
            C_map, Gnew_map, bnew_map = self._build_c_g_and_b_via_stream(
                y_tr, θ_tr, p_mat_old, r_arr_old, cand_struct,
                C_map, Gnew_map, bnew_map
            )
            accumulation_time += time.perf_counter() - t_accum_start

            # Evaluate candidates (solve on train using block Gram; score on val)
            best_r2_score = -1e18
            best_choice: CandidateEvaluation | None = None

            t_score_start = time.perf_counter()
            for k, paths in candidates.items():
                candidate_eval, candidate_lambda_sweep = self._evaluate_candidate_block(
                    k=k,
                    paths=paths,
                    selected_paths=selected_paths,
                    G_old=G_old,
                    b_old=b_old,
                    C=C_map[k],
                    Gnew=Gnew_map[k],
                    bnew=bnew_map[k],
                    θ_val=θ_val,
                    y_val=y_val,
                    lambda_star=lambda_star,
                )
                lambda_sweep_time += candidate_lambda_sweep

                if candidate_eval["cand_r2_score"] > best_r2_score:
                    best_r2_score = candidate_eval["cand_r2_score"]
                    best_choice = candidate_eval
            scoring_time += time.perf_counter() - t_score_start

            # Update with new best things
            assert best_choice is not None # (for type checkers)
            k_win = best_choice["k"]
            block_win = best_choice["paths"]
            G_new_old = best_choice["G_trial"]
            b_new_old = best_choice["b_trial"]
            lam_win = best_choice["cand_lambda"]
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
                if current_block_size > self.min_block_size and self.adaptive_block_size:
                    current_block_size=max(self.min_block_size, current_block_size - 1)

                if no_improve_count >= self.early_stopping_patience:
                    self._log(
                        f"[Early stopping] No improvement for {self.early_stopping_patience} "
                        f"rounds at {len(selected_paths)} paths"
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
            t_lambda_refit_start = time.perf_counter()
            lambda_star = self._select_lambda_from_gram(
                G_old, b_old, θ_val, y_val, selected_paths
            )
            lambda_sweep_time += time.perf_counter() - t_lambda_refit_start
        t1 = time.perf_counter()

        stats = Stats(
            stopped_early=stopped_early,
            history=history,
            time_taken=t1-t0,
            accumulation_time_sec=accumulation_time,
            scoring_time_sec=scoring_time,
            lambda_sweep_time_sec=lambda_sweep_time,
        )
        return selected_paths, lambda_star, stats

    def _evaluate_candidate_block(
        self,
        *,
        k: int,
        paths: List[MVec],
        selected_paths: List[MVec],
        G_old: Array,
        b_old: Array,
        C: Array,
        Gnew: Array,
        bnew: Array,
        θ_val: Array,
        y_val: Array,
        lambda_star: float | None,
    ) -> tuple[CandidateEvaluation, float]:
        """Evaluate one greedy candidate block and return score metadata."""
        G_trial = np.block([[G_old, C], [C.T, Gnew]])
        b_trial = np.concatenate([b_old, bnew])
        trial_paths = selected_paths + paths
        p_mat_trial, r_arr_trial = _path_matrix_and_r_arr(trial_paths)
        p_mat_trial = self._as_internal_dtype(p_mat_trial)

        lambda_sweep_time = 0.0
        if lambda_star is None:
            if self.normalize_columns:
                scaling_vector = self._calc_scaling_vector(G_trial)
                inv_s_trial = 1.0 / scaling_vector
                Gs_trial = (inv_s_trial[:, None] * G_trial) * inv_s_trial[None, :]
                scaled_b_trial = inv_s_trial * b_trial
                scaled_evals_trial, U_trial = np.linalg.eigh(Gs_trial)

                def solve_for_coeffs(lambda_: float) -> Array:
                    scaled_coeffs = self._ridge_solve_for_coeffs(
                        scaled_evals_trial, U_trial, scaled_b_trial, lambda_
                    )
                    return scaled_coeffs * inv_s_trial

            else:
                evals_trial, U_trial = np.linalg.eigh(G_trial)

                def solve_for_coeffs(lambda_: float) -> Array:
                    return self._ridge_solve_for_coeffs(
                        evals_trial, U_trial, b_trial, lambda_
                    )

            cand_lambda, cand_r2_score, lambda_sweep_time = self._score_lambda_candidates(
                solve_for_coeffs=solve_for_coeffs,
                p_mat=p_mat_trial,
                r_arr=r_arr_trial,
                θ_val=θ_val,
                y_val=y_val,
            )
        else:
            coeffs = self._solve_normal_eqn(G_trial, b_trial, lambda_star)
            y_val_hat = self._stream_predict(θ_val, coeffs, p_mat_trial, r_arr_trial)
            _, cand_r2_score = _metrics(y_val, y_val_hat)
            cand_lambda = lambda_star

        return (
            CandidateEvaluation(
                k=k,
                paths=paths,
                G_trial=G_trial,
                b_trial=b_trial,
                cand_lambda=float(cand_lambda),
                cand_r2_score=float(cand_r2_score),
            ),
            lambda_sweep_time,
        )

    def _build_c_g_and_b_via_stream(
        self,
        y_tr: Array,
        θ_tr: Array,
        p_mat_old: Array,
        r_arr_old: Array,
        cand_struct: Dict[int, Tuple[Array, Array]],
        C_map: Dict[int, Array],
        Gnew_map: Dict[int, Array],
        bnew_map: Dict[int, Array]
    ) -> tuple[Dict[int, Array], Dict[int, Array], Dict[int, Array]]:
        Ntr = θ_tr.shape[0]
        for start in range(0, Ntr, self.batch_rows):
            end = min(Ntr, start + self.batch_rows)

            y_batch = y_tr[start:end]
            θ_batch = θ_tr[start:end]
            Φ_old_batch = _build_feature_matrix(θ_batch, p_mat_old, r_arr_old)

            # Loop through k values, e.g., 1, 2, 3
            for k, (p_mat_block, r_arr_block) in cand_struct.items():
                Φ_new_batch = _build_feature_matrix(
                    θ_batch, p_mat_block, r_arr_block, False
                )

                C_map[k] += Φ_old_batch.T @ Φ_new_batch
                Gnew_map[k] += Φ_new_batch.T @ Φ_new_batch
                bnew_map[k] += Φ_new_batch.T @ y_batch

        return C_map, Gnew_map, bnew_map

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def _batch(self, θ_tr: Array, start: int, end: int) -> Array:
        """Batching helper."""
        return θ_tr[start:end]

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
            p_mat_block, r_arr_block = _path_matrix_and_r_arr(block)
            cand_struct[k] = (self._as_internal_dtype(p_mat_block), r_arr_block)
            Qnew = len(block)
            C_map[k] = np.zeros((M_old, Qnew), dtype=self._internal_dtype)
            Gnew_map[k] = np.zeros((Qnew, Qnew), dtype=self._internal_dtype)
            bnew_map[k] = np.zeros(Qnew, dtype=self._internal_dtype)
        return cand_struct, C_map, Gnew_map, bnew_map


class SpectralPathClassifier(SpectralPathRegressor):
    """
    Binary spectral-path classifier trained with penalized logistic IRLS.

    This estimator reuses the spectral path dictionary construction from
    ``SpectralPathRegressor`` but fits logistic coefficients for

        f(x) = w0 + sum_q w_q phi_q(x),
        P(y=1 | x) = sigmoid(f(x)).
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
        lambda_parallel_workers: int = 1,
        blas_thread_policy: Literal["auto", "none", "single", "manual"] = "auto",
        blas_threads: int | None = None,
        early_stopping_patience: int = 3,
        early_stopping_tol: float = 1e-4,
        greedy_subsample: float | int | None = None,
        scaler_type: ScalerType | str = ScalerType.STANDARD_PERCENTILE_MINMAX,
        iqr_percentile_range: Tuple[float, float] = (25.0, 75.0),
        bound_percentiles: Tuple[float, float] = (2.0, 98.0),
        adaptive_block_size: bool = True,
        min_block_size: int = 1,
        use_importance_ordering: bool = True,
        irls_max_iter: int = 100,
        irls_tol: float = 1e-6,
        irls_eps: float = 1e-8,
    ) -> None:
        """Initialise a binary spectral-path classifier."""
        super().__init__(
            max_paths=max_paths,
            block_size=block_size,
            lambda_grid=lambda_grid,
            l_max=l_max,
            batch_rows=batch_rows,
            k_values=k_values,
            val_size=val_size,
            random_state=random_state,
            verbose=verbose,
            final_lambda_refit=final_lambda_refit,
            normalize_columns=normalize_columns,
            normalize_intercept=normalize_intercept,
            eps_col_norm=eps_col_norm,
            use_float32=use_float32,
            lambda_parallel_workers=lambda_parallel_workers,
            blas_thread_policy=blas_thread_policy,
            blas_threads=blas_threads,
            early_stopping_patience=early_stopping_patience,
            early_stopping_tol=early_stopping_tol,
            greedy_subsample=greedy_subsample,
            scaler_type=scaler_type,
            iqr_percentile_range=iqr_percentile_range,
            bound_percentiles=bound_percentiles,
            adaptive_block_size=adaptive_block_size,
            min_block_size=min_block_size,
            use_importance_ordering=use_importance_ordering,
        )
        if irls_max_iter < 1:
            raise ValueError("irls_max_iter must be >= 1.")
        if irls_tol <= 0.0:
            raise ValueError("irls_tol must be > 0.")
        if not (0.0 < irls_eps < 0.5):
            raise ValueError("irls_eps must lie in (0, 0.5).")
        self.irls_max_iter = int(irls_max_iter)
        self.irls_tol = float(irls_tol)
        self.irls_eps = float(irls_eps)
        self.classes_ = np.array([0, 1], dtype=int)

    def _validate_binary_targets(self, y: Array) -> Array:
        """Ensure the classifier receives exactly 0/1 targets."""
        y = np.asarray(y, dtype=float).ravel()
        uniques = np.unique(y)
        if uniques.size == 0:
            raise ValueError("y must not be empty.")
        if not np.array_equal(uniques, np.array([0.0, 1.0])) and not np.array_equal(
            uniques, np.array([0.0])
        ) and not np.array_equal(uniques, np.array([1.0])):
            raise ValueError("SpectralPathClassifier requires binary labels encoded as 0/1.")
        return y

    def _build_design_matrix(self, theta: Array, paths: Sequence[MVec]) -> Array:
        """Build the spectral feature matrix for a fixed path set."""
        if len(paths) == 0:
            return np.ones((theta.shape[0], 1), dtype=self._internal_dtype)
        p_mat, r_arr = _path_matrix_and_r_arr(paths)
        p_mat = self._as_internal_dtype(p_mat)
        return _build_feature_matrix(theta, p_mat, r_arr)

    def _initial_logit(self, y: Array) -> float:
        """Return the intercept-only logit for the empirical positive rate."""
        mean_y = float(np.clip(np.mean(y), self.irls_eps, 1.0 - self.irls_eps))
        return float(np.log(mean_y / (1.0 - mean_y)))

    def _fit_logistic_irls(
        self,
        phi: Array,
        y: Array,
        lambda_: float,
        initial_coeffs: Array | None = None,
    ) -> Array:
        """Fit penalized logistic coefficients with IRLS on a fixed design matrix."""
        phi = np.asarray(phi, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if phi.ndim != 2:
            raise ValueError("phi must be 2D.")
        if phi.shape[0] != y.shape[0]:
            raise ValueError("phi rows must match y.")

        if self.normalize_columns:
            scaling_vector = self._calc_scaling_vector(phi.T @ phi)
            inv_s = 1.0 / scaling_vector
            phi_scaled = phi * inv_s[None, :]
        else:
            scaling_vector = np.ones(phi.shape[1], dtype=float)
            inv_s = scaling_vector
            phi_scaled = phi

        penalty = np.ones(phi_scaled.shape[1], dtype=float)
        penalty[0] = 0.0

        if initial_coeffs is None or np.asarray(initial_coeffs).shape[0] != phi_scaled.shape[1]:
            coeffs = np.zeros(phi_scaled.shape[1], dtype=float)
            coeffs[0] = self._initial_logit(y)
        else:
            coeffs = np.asarray(initial_coeffs, dtype=float) / scaling_vector

        ridge = float(lambda_) * penalty

        for _ in range(self.irls_max_iter):
            logits = phi_scaled @ coeffs
            probs = _clip_probabilities(_sigmoid(logits), eps=self.irls_eps)
            weights = np.maximum(probs * (1.0 - probs), self.irls_eps)
            work_response = logits + (y - probs) / weights

            weighted_phi = phi_scaled * weights[:, None]
            hessian = phi_scaled.T @ weighted_phi
            hessian += np.diag(ridge)
            rhs = phi_scaled.T @ (weights * work_response)

            try:
                new_coeffs = np.linalg.solve(hessian, rhs)
            except np.linalg.LinAlgError:
                new_coeffs = np.linalg.lstsq(hessian, rhs, rcond=None)[0]

            delta = np.linalg.norm(new_coeffs - coeffs)
            coeffs = new_coeffs
            if delta <= self.irls_tol * (1.0 + np.linalg.norm(coeffs)):
                break

        return coeffs / scaling_vector

    def _score_lambda_candidates_classifier(
        self,
        phi_tr: Array,
        y_tr: Array,
        phi_val: Array,
        y_val: Array,
        initial_coeffs: Array | None = None,
    ) -> tuple[float, float, Array, float]:
        """Evaluate all lambdas and return the best validation log loss."""

        def evaluate_one(lam: float) -> tuple[float, float, Array]:
            coeffs = self._fit_logistic_irls(phi_tr, y_tr, lam, initial_coeffs=initial_coeffs)
            val_probs = _sigmoid(phi_val @ coeffs)
            val_loss = _binary_log_loss(y_val, val_probs, eps=self.irls_eps)
            return float(lam), float(val_loss), coeffs

        t0 = time.perf_counter()
        if self._outer_parallelism_enabled():
            max_workers = min(self.lambda_parallel_workers, len(self.lambda_grid))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(evaluate_one, self.lambda_grid))
        else:
            results = [evaluate_one(lam) for lam in self.lambda_grid]

        best_lam, best_loss, best_coeffs = min(results, key=lambda item: (item[1], item[0]))
        return best_lam, best_loss, best_coeffs, time.perf_counter() - t0

    def _evaluate_candidate_block_classifier(
        self,
        *,
        k: int,
        paths: List[MVec],
        selected_paths: List[MVec],
        theta_tr: Array,
        y_tr: Array,
        theta_val: Array,
        y_val: Array,
        lambda_star: float | None,
        initial_coeffs: Array | None,
    ) -> tuple[ClassifierCandidateEvaluation, float]:
        """Evaluate one classifier candidate block and return validation log loss."""
        trial_paths = selected_paths + paths
        phi_tr = self._build_design_matrix(theta_tr, trial_paths)
        phi_val = self._build_design_matrix(theta_val, trial_paths)

        lambda_sweep_time = 0.0
        if lambda_star is None:
            cand_lambda, cand_log_loss, _, lambda_sweep_time = (
                self._score_lambda_candidates_classifier(
                    phi_tr, y_tr, phi_val, y_val, initial_coeffs=initial_coeffs
                )
            )
        else:
            coeffs = self._fit_logistic_irls(phi_tr, y_tr, lambda_star, initial_coeffs=initial_coeffs)
            cand_log_loss = _binary_log_loss(y_val, _sigmoid(phi_val @ coeffs), eps=self.irls_eps)
            cand_lambda = float(lambda_star)

        return (
            ClassifierCandidateEvaluation(
                k=k,
                paths=paths,
                cand_lambda=float(cand_lambda),
                cand_log_loss=float(cand_log_loss),
            ),
            lambda_sweep_time,
        )

    def _select_paths_and_lambda_classifier(
        self, theta_tr: Array, y_tr: Array, theta_val: Array, y_val: Array
    ) -> Tuple[List[MVec], float, Stats]:
        """Greedily select k-sparse paths using validation log loss."""
        t0 = time.perf_counter()
        generators = {k: self._path_generator(k) for k in self.k_values}
        selected_paths: List[MVec] = []
        history: List[Tuple[int, int, float, float]] = []
        lambda_star: float | None = None
        best_log_loss_overall = float("inf")
        no_improve_count = 0
        stopped_early = False
        current_block_size = self.block_size
        scoring_time = 0.0
        lambda_sweep_time = 0.0

        while len(selected_paths) < self.max_paths:
            remaining_number_of_paths = self.max_paths - len(selected_paths)
            block_size = min(current_block_size, remaining_number_of_paths)
            candidates = self._generate_candidates(generators, block_size)

            if not candidates:
                self._log("All generators exhausted before reaching max_paths.")
                break

            best_log_loss = float("inf")
            best_choice: ClassifierCandidateEvaluation | None = None

            t_score_start = time.perf_counter()
            phi_current = self._build_design_matrix(theta_tr, selected_paths)
            initial_coeffs = None
            if self.coef_ is not None and len(selected_paths) == len(self.selected_paths_ or []):
                initial_coeffs = self.coef_
            elif phi_current.shape[1] > 0:
                initial_coeffs = self._fit_logistic_irls(
                    phi_current,
                    y_tr,
                    lambda_star if lambda_star is not None else self.lambda_grid[0],
                )

            for k, paths in candidates.items():
                candidate_eval, candidate_lambda_sweep = self._evaluate_candidate_block_classifier(
                    k=k,
                    paths=paths,
                    selected_paths=selected_paths,
                    theta_tr=theta_tr,
                    y_tr=y_tr,
                    theta_val=theta_val,
                    y_val=y_val,
                    lambda_star=lambda_star,
                    initial_coeffs=initial_coeffs,
                )
                lambda_sweep_time += candidate_lambda_sweep

                if candidate_eval["cand_log_loss"] < best_log_loss:
                    best_log_loss = candidate_eval["cand_log_loss"]
                    best_choice = candidate_eval
            scoring_time += time.perf_counter() - t_score_start

            assert best_choice is not None
            k_win = best_choice["k"]
            block_win = best_choice["paths"]
            lam_win = best_choice["cand_lambda"]
            selected_paths.extend(block_win)
            if lambda_star is None:
                lambda_star = lam_win
            history.append((k_win, len(block_win), lambda_star, best_log_loss))

            improving = best_log_loss < best_log_loss_overall - self.early_stopping_tol
            if improving:
                best_log_loss_overall = best_log_loss
                no_improve_count = 0
                if self.adaptive_block_size and current_block_size < self.block_size:
                    current_block_size = min(self.block_size, current_block_size + 1)
            else:
                no_improve_count += 1
                if current_block_size > self.min_block_size and self.adaptive_block_size:
                    current_block_size = max(self.min_block_size, current_block_size - 1)
                if no_improve_count >= self.early_stopping_patience:
                    self._log(
                        f"[Early stopping] No improvement for {self.early_stopping_patience} "
                        f"rounds at {len(selected_paths)} paths"
                    )
                    stopped_early = True
                    break

            self._log(
                f"[Greedy] Added k={k_win} block of {len(block_win)} → total="
                f"{len(selected_paths)} | λ_used={lambda_star} | log_loss_val="
                f"{best_log_loss:0.6f} | block_size={current_block_size}"
            )

        if lambda_star is None:
            lambda_star = self.lambda_grid[0]

        if self.final_lambda_refit and len(self.lambda_grid) > 0:
            t_lambda_refit_start = time.perf_counter()
            phi_tr = self._build_design_matrix(theta_tr, selected_paths)
            phi_val = self._build_design_matrix(theta_val, selected_paths)
            lambda_star, _, _, extra_sweep = self._score_lambda_candidates_classifier(
                phi_tr, y_tr, phi_val, y_val
            )
            lambda_sweep_time += extra_sweep + (time.perf_counter() - t_lambda_refit_start) - extra_sweep

        stats = Stats(
            stopped_early=stopped_early,
            history=history,
            time_taken=time.perf_counter() - t0,
            accumulation_time_sec=0.0,
            scoring_time_sec=scoring_time,
            lambda_sweep_time_sec=lambda_sweep_time,
        )
        return selected_paths, lambda_star, stats

    def fit(
        self, X: Array, y: Array, *, X_val: Array | None = None, y_val: Array | None = None
    ) -> "SpectralPathClassifier":
        """Fit the binary classifier."""
        t_fit_start = time.perf_counter()
        X = np.asarray(X)
        y = self._validate_binary_targets(y)
        check_dimensions(X, y)

        self.n_features_in_ = int(X.shape[1])
        k_max = max(self.k_values)
        if k_max > self.n_features_in_:
            raise ValueError(f"Invalid k_values: max(k_values)={k_max} exceeds n_features.")

        X_tr, y_tr, X_val, y_val = self._prepare_train_val_split(X, y, X_val, y_val)
        y_tr = self._validate_binary_targets(y_tr)
        y_val = self._validate_binary_targets(y_val)

        t_pre_start = time.perf_counter()
        theta_tr, theta_val = self._transform_data(X_tr, X_val)
        theta_tr = self._as_internal_dtype(theta_tr)
        theta_val = self._as_internal_dtype(theta_val)
        y_tr = self._as_internal_dtype(y_tr)
        y_val = self._as_internal_dtype(y_val)
        preprocessing_time = time.perf_counter() - t_pre_start

        self._compute_feature_importance(theta_tr, y_tr)
        theta_tr_greedy, y_tr_greedy = self._subsample_greedy_training_data(theta_tr, y_tr)
        resolved_blas_threads = self._resolve_blas_threads(theta_tr_greedy.shape[0])
        self._resolved_blas_threads_ = resolved_blas_threads
        self._log(
            "[BLAS] policy="
            f"{self.blas_thread_policy} resolved_threads={resolved_blas_threads}"
        )

        with self._blas_thread_limit_context(resolved_blas_threads):
            paths, lambda_star, stats = self._select_paths_and_lambda_classifier(
                theta_tr_greedy, y_tr_greedy, theta_val, y_val
            )

        theta_all = np.vstack([theta_tr, theta_val])
        y_all = np.concatenate([y_tr, y_val])

        with self._blas_thread_limit_context(resolved_blas_threads):
            t_design_start = time.perf_counter()
            phi_all = self._build_design_matrix(theta_all, paths)
            t_design_end = time.perf_counter()
            coefficients = self._fit_logistic_irls(phi_all, y_all, lambda_star)
            t_solve_end = time.perf_counter()

        self._save_learned_state(paths, lambda_star, coefficients)
        self._cache_ray_structures(paths)

        feature_importance = self._compute_feature_importance_from_model()
        phase_timings = PhaseTimings(
            preprocessing_sec=preprocessing_time,
            greedy_accumulation_sec=stats.accumulation_time_sec,
            greedy_scoring_sec=stats.scoring_time_sec,
            lambda_sweep_sec=stats.lambda_sweep_time_sec,
            final_normal_eqn_sec=t_design_end - t_design_start,
            final_solve_sec=t_solve_end - t_design_end,
            total_fit_sec=time.perf_counter() - t_fit_start,
        )
        self.fit_report_ = FitReport(
            lambda_star=lambda_star,
            selected_count=len(paths),
            greedy_time_sec=stats.time_taken,
            final_solve_time_sec=t_solve_end - t_design_end,
            history=stats.history,
            stopped_early=stats.stopped_early,
            feature_importance=feature_importance,
            phase_timings=phase_timings,
            blas_threads=BlasThreadInfo(
                policy=self.blas_thread_policy,
                resolved_threads=resolved_blas_threads,
            ),
        )
        return self

    def decision_function(self, X: Array) -> Array:
        """Return the fitted logits for each sample."""
        return super().predict(X)

    def predict_proba(self, X: Array) -> Array:
        """Return class probabilities as a two-column array."""
        logits = self.decision_function(X)
        prob_pos = _clip_probabilities(_sigmoid(logits), eps=self.irls_eps)
        return np.column_stack([1.0 - prob_pos, prob_pos])

    def predict(self, X: Array) -> Array:
        """Predict hard binary labels using a 0.5 probability threshold."""
        prob_pos = self.predict_proba(X)[:, 1]
        return (prob_pos >= 0.5).astype(int)

    def score(self, X: Array, y: Array) -> float:
        """Return binary classification accuracy."""
        y_true = self._validate_binary_targets(y)
        prob_pos = self.predict_proba(X)[:, 1]
        return _binary_accuracy(y_true, prob_pos)
