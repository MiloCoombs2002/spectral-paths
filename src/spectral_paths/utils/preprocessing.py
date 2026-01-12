from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np

from spectral_paths.types import ScalerType


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

    scaler: ScalerType = "standard_percentile_minmax"
    clip: bool = True
    eps: float = 1e-12
    percentile_range: Tuple[float, float] = (25.0, 75.0)
    bound_percentiles: Tuple[float, float] = (1.0, 99.0)

    # Learned attributes
    n_features_in_: Optional[int] = None

    # For z-scoring
    center_: Optional[np.ndarray] = None
    scale_: Optional[np.ndarray] = None

    # For raw minmax
    min_: Optional[np.ndarray] = None
    max_: Optional[np.ndarray] = None

    # For percentile-minmax on z
    z_lo_: Optional[np.ndarray] = None
    z_hi_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y=None) -> "AngularTransformer":
        X = self._as_2d_float(X)
        _, d = X.shape
        self.n_features_in_ = d

        mode = self._canonical_mode(self.scaler)

        # Reset learned fields
        self.center_ = self.scale_ = None
        self.min_ = self.max_ = None
        self.z_lo_ = self.z_hi_ = None

        if mode == "minmax":
            self.min_ = np.min(X, axis=0)
            self.max_ = np.max(X, axis=0)
            return self

        # Fit z-score params first
        if mode.startswith("standard_"):
            self.center_ = np.mean(X, axis=0)
            s = np.std(X, axis=0, ddof=0)
            self.scale_ = np.maximum(s, self.eps)

        elif mode.startswith("robust_"):
            lo, hi = self.percentile_range
            self.center_ = np.median(X, axis=0)
            q_lo = np.percentile(X, lo, axis=0)
            q_hi = np.percentile(X, hi, axis=0)
            iqr = np.maximum(q_hi - q_lo, self.eps)
            self.scale_ = iqr

        else:
            raise ValueError(f"Unknown scaler mode: {self.scaler!r}")

        # If percentile-minmax, we also need bounds in z-space from training data
        if mode.endswith("percentile_minmax"):
            z = (X - self.center_) / np.maximum(self.scale_, self.eps)
            blo, bhi = self.bound_percentiles
            self.z_lo_ = np.percentile(z, blo, axis=0)
            self.z_hi_ = np.percentile(z, bhi, axis=0)

            # Avoid zero range in z-bounds
            rng = np.maximum(self.z_hi_ - self.z_lo_, self.eps)
            self.z_hi_ = self.z_lo_ + rng

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
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
        return self.fit(X, y).transform(X)

    # ---------- Internal methods ----------
    def _scale_to_unit_interval(self, X: np.ndarray) -> np.ndarray:
        mode = self._canonical_mode(self.scaler)

        if mode == "minmax":
            rng = np.maximum(self.max_ - self.min_, self.eps)  # type: ignore
            u01 = (X - self.min_) / rng
            return 2.0 * u01 - 1.0

        # z-score first
        z = (X - self.center_) / np.maximum(self.scale_, self.eps)  # type: ignore

        if mode.endswith("_tanh"):
            return np.tanh(z)

        if mode.endswith("percentile_minmax"):
            # Map z into [0,1] using training percentile bounds, then to [-1,1]
            rng = np.maximum(self.z_hi_ - self.z_lo_, self.eps) # type: ignore
            u01 = (z - self.z_lo_) / rng
            return 2.0 * u01 - 1.0

        raise ValueError(f"Unhandled mode: {self.scaler!r}")

    @staticmethod
    def _canonical_mode(scaler: ScalerType) -> str:
        if scaler == "standard":
            return "standard_tanh"
        if scaler == "robust":
            return "robust_tanh"
        return scaler

    def _check_is_fitted(self) -> None:
        if self.n_features_in_ is None:
            raise RuntimeError("AngularTransformer is not fitted. Call fit() first.")

    @staticmethod
    def _as_2d_float(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {X.shape}.")
        if not np.issubdtype(X.dtype, np.floating):
            X = X.astype(np.float64, copy=False)
        return X
