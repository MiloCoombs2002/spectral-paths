from typing import Literal, Tuple

import numpy as np

Array = np.ndarray
MVec = Tuple[int, ...]
PVec = Tuple[int, ...]
PR = Tuple[PVec, int]

ScalerType = Literal[
    "standard_minmax",
    "standard_percentile_minmax",
    "robust_qclip_minmax",
    "robust_sigclip_minmax",
    "robust_percentile_minmax",
    "quantile_uniform",
    "standard",
    "standard_tanh",
    "minmax",
    "robust",
    "robust_tanh"
]