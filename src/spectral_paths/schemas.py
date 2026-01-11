from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from spectral_paths.types import Array

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
    """
    lambda_star: float
    selected_count: int
    greedy_time_sec: float
    final_solve_time_sec: float
    history: List[Tuple[int, int, float, float]]
    stopped_early: bool
    feature_importance: Array | None = None
