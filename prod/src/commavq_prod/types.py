"""Shared numpy typing aliases."""

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int32]
TokenArray = NDArray[np.int16]
