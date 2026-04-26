"""Schema objects used by the spectral-path estimator."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import List, Tuple, TypedDict

from spectral_paths.types import Array


class ScalerType(StrEnum):
    """Allowed scaler modes for angular preprocessing."""

    MINMAX = "minmax"
    STANDARD_TANH = "standard_tanh"
    ROBUST_TANH = "robust_tanh"
    STANDARD_PERCENTILE_MINMAX = "standard_percentile_minmax"
    ROBUST_PERCENTILE_MINMAX = "robust_percentile_minmax"
    STANDARD = "standard"
    ROBUST = "robust"


@dataclass
class PhaseTimings:
    """Phase-level timing summary for one model fit."""

    preprocessing_sec: float = 0.0
    greedy_accumulation_sec: float = 0.0
    greedy_scoring_sec: float = 0.0
    lambda_sweep_sec: float = 0.0
    final_normal_eqn_sec: float = 0.0
    final_solve_sec: float = 0.0
    total_fit_sec: float = 0.0


@dataclass
class BlasThreadInfo:
    """Resolved BLAS threading policy for one model fit."""

    policy: str = "auto"
    resolved_threads: int | None = None


@dataclass
class FitReport:
    """
    Summary of a spectral path model fit.

    Attributes:
        lambda_star (float): Selected ridge regularisation parameter.
        selected_count (int): Total number of spectral paths selected.
        greedy_time_sec (float): Wall-clock time spent in greedy path selection.
        final_solve_time_sec (float): Wall-clock time spent in the final ridge solve.
        history (List[Tuple[int, int, float, float]]): Per-iteration greedy selection
            history. Each entry records (iteration index, cumulative path count,
            validation score, regularisation value).
        stopped_early (bool): Whether greedy selection terminated due to early stopping.
        feature_importance (Array | None): Feature-importance weights used for path
            ordering, if available.
        phase_timings (PhaseTimings): Aggregated timings for major fit phases.
        blas_threads (BlasThreadInfo): Resolved BLAS thread policy for this fit.
    """
    lambda_star: float
    selected_count: int
    greedy_time_sec: float
    final_solve_time_sec: float
    history: List[Tuple[int, int, float, float]]
    stopped_early: bool
    feature_importance: Array | None = None
    phase_timings: PhaseTimings = field(default_factory=PhaseTimings)
    blas_threads: BlasThreadInfo = field(default_factory=BlasThreadInfo)

@dataclass
class Stats:
    """Stats dataclass."""
    stopped_early: bool
    time_taken: float
    history: List[Tuple[int, int, float, float]]
    accumulation_time_sec: float = 0.0
    scoring_time_sec: float = 0.0
    lambda_sweep_time_sec: float = 0.0


class DatasetSpec(TypedDict):
    """DatasetSpec schema."""
    name: str
    openml_name: str
    version: int
