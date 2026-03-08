"""Preprocessing utilities for scaling features and mapping to angular space."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from spectral_paths.schemas import ScalerType


@dataclass
class AngularTransformer:
    """
    Scale features into [-1, 1] then map to angular coordinates via arccos.

    The main job: produce X_scaled in [-1, 1] robustly and deterministically,
    then compute theta = arccos(X_scaled) in [0, pi].

    Parameters:
        scaler: Scaling strategy.
            - "minmax": per-feature minmax into [-1, 1]
            - "standard_tanh": z-score then tanh -> [-1, 1]
            - "robust_tanh": robust z-score (median/IQR) then tanh -> [-1, 1]
            - "standard_percentile_minmax": z-score then percentile minmax -> [-1, 1]
            - "robust_percentile_minmax": robust z-score -> percentile minmax -> [-1, 1]
            Aliases:
            - "standard" -> "standard_tanh"
            - "robust" -> "robust_tanh"
        clip: Whether to clip final scaled values into [-1, 1] before arccos.
              Usually True for numerical safety.
        eps: Small constant to avoid division by zero.
        percentile_range: Percentiles for IQR when using robust centering/scaling.
        bound_percentiles: Percentiles used to define the min/max bounds in
            "*_percentile_minmax" modes (computed on the scaled z values).
    """

    scaler: ScalerType | str = ScalerType.STANDARD_PERCENTILE_MINMAX
    clip: bool = True
    eps: float = 1e-12
    percentile_range: Tuple[float, float] = (25.0, 75.0)
    bound_percentiles: Tuple[float, float] = (1.0, 99.0)

    # Learned attributes
    n_features_in_: int | None = None

    # For z-scoring
    center_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    # For raw minmax
    min_: np.ndarray | None = None
    max_: np.ndarray | None = None

    # For percentile-minmax on z
    z_lo_: np.ndarray | None = None
    z_hi_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y=None) -> "AngularTransformer":
        """Learn scaling parameters from training inputs."""
        X = self._as_2d_float(X)
        self.n_features_in_ = X.shape[1]

        mode = self._canonical_mode(self.scaler)

        # Reset learned fields
        self.center_ = self.scale_ = None
        self.min_ = self.max_ = None
        self.z_lo_ = self.z_hi_ = None

        if mode is ScalerType.MINMAX:
            self.min_ = np.min(X, axis=0)
            self.max_ = np.max(X, axis=0)
            return self

        # Fit z-score params first
        if mode in (
            ScalerType.STANDARD_TANH,
            ScalerType.STANDARD_PERCENTILE_MINMAX,
        ):
            self.center_ = np.mean(X, axis=0)
            s = np.std(X, axis=0, ddof=0)
            self.scale_ = np.maximum(s, self.eps)

        elif mode in (
            ScalerType.ROBUST_TANH,
            ScalerType.ROBUST_PERCENTILE_MINMAX,
        ):
            lo, hi = self.percentile_range
            self.center_ = np.median(X, axis=0)
            q_lo = np.percentile(X, lo, axis=0)
            q_hi = np.percentile(X, hi, axis=0)
            iqr = np.maximum(q_hi - q_lo, self.eps)
            self.scale_ = iqr

        else:
            raise ValueError(f"Unknown scaler mode: {self.scaler!r}")

        # If percentile-minmax, we also need bounds in z-space from training data
        if mode in (
            ScalerType.STANDARD_PERCENTILE_MINMAX,
            ScalerType.ROBUST_PERCENTILE_MINMAX,
        ):
            center, scale = self._require_zscore_params()
            z = (X - center) / np.maximum(scale, self.eps)
            blo, bhi = self.bound_percentiles
            self.z_lo_ = np.percentile(z, blo, axis=0)
            self.z_hi_ = np.percentile(z, bhi, axis=0)

            # Avoid zero range in z-bounds
            z_lo, z_hi = self._require_percentile_bounds()
            rng = np.maximum(z_hi - z_lo, self.eps)
            self.z_hi_ = z_lo + rng

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply learned scaling and return angular coordinates."""
        self._check_is_fitted()
        X = self._as_2d_float(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.n_features_in_}."
            )

        Xs = self._scale_to_unit_interval(X)
        if self.clip:
            Xs = np.clip(Xs, -1.0, 1.0)

        return np.arccos(Xs)

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """Fit preprocessing parameters and transform `X` in one pass."""
        return self.fit(X, y).transform(X)

    # ---------- Internal methods ----------
    def _scale_to_unit_interval(self, X: np.ndarray) -> np.ndarray:
        """Scale features into [-1, 1] according to the configured scaler mode."""
        mode = self._canonical_mode(self.scaler)

        if mode is ScalerType.MINMAX:
            min_, max_ = self._require_minmax_params()
            rng = np.maximum(max_ - min_, self.eps)
            u01 = (X - min_) / rng
            return 2.0 * u01 - 1.0

        # z-score first
        center, scale = self._require_zscore_params()
        z = (X - center) / np.maximum(scale, self.eps)

        if mode in (ScalerType.STANDARD_TANH, ScalerType.ROBUST_TANH):
            return np.tanh(z)

        if mode in (
            ScalerType.STANDARD_PERCENTILE_MINMAX,
            ScalerType.ROBUST_PERCENTILE_MINMAX,
        ):
            # Map z into [0,1] using training percentile bounds, then to [-1,1]
            z_lo, z_hi = self._require_percentile_bounds()
            rng = np.maximum(z_hi - z_lo, self.eps)
            u01 = (z - z_lo) / rng
            return 2.0 * u01 - 1.0

        raise ValueError(f"Unhandled mode: {self.scaler!r}")

    @staticmethod
    def _canonical_mode(scaler: ScalerType | str) -> ScalerType:
        """Normalize aliases and validate scaler names against `ScalerType`."""
        try:
            mode = ScalerType(scaler)
        except ValueError as exc:
            allowed = ", ".join(member.value for member in ScalerType)
            raise ValueError(f"Unknown scaler mode {scaler!r}. Allowed values: {allowed}.") from exc

        if mode is ScalerType.STANDARD:
            return ScalerType.STANDARD_TANH
        if mode is ScalerType.ROBUST:
            return ScalerType.ROBUST_TANH
        return mode

    def _check_is_fitted(self) -> None:
        """Ensure the transformer has already been fitted."""
        if self.n_features_in_ is None:
            raise RuntimeError("AngularTransformer is not fitted. Call fit() first.")

    def _require_minmax_params(self) -> tuple[np.ndarray, np.ndarray]:
        """Return min/max arrays, or raise if they are unavailable."""
        if self.min_ is None or self.max_ is None:
            raise RuntimeError("Min-max parameters are unavailable. Fit transformer first.")
        return self.min_, self.max_

    def _require_zscore_params(self) -> tuple[np.ndarray, np.ndarray]:
        """Return z-score center/scale arrays, or raise if unavailable."""
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("Z-score parameters are unavailable. Fit transformer first.")
        return self.center_, self.scale_

    def _require_percentile_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return percentile bounds for z-space scaling, or raise if unavailable."""
        if self.z_lo_ is None or self.z_hi_ is None:
            raise RuntimeError(
                "Percentile bounds are unavailable. Fit transformer with a "
                "percentile-minmax scaler first."
            )
        return self.z_lo_, self.z_hi_

    @staticmethod
    def _as_2d_float(X: np.ndarray) -> np.ndarray:
        """Convert input to a 2D floating ndarray."""
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {X.shape}.")
        if not np.issubdtype(X.dtype, np.floating):
            X = X.astype(np.float64, copy=False)
        return X
