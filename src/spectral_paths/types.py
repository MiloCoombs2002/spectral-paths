"""Shared low-level type aliases for spectral-path internals."""

from typing import Tuple

import numpy as np

Array = np.ndarray
MVec = Tuple[int, ...]
PVec = Tuple[int, ...]
PR = Tuple[PVec, int]
