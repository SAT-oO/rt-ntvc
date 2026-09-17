import numpy as np
from numpy.typing import NDArray

M: int

def encode(
    symbols: NDArray[np.int32],
    probs: NDArray[np.float32],
) -> bytes: ...

def decode(
    data: bytes,
    probs: NDArray[np.float32],
) -> NDArray[np.int32]: ...

class ClipDecoder:
    def __init__(self, data: bytes, n_frames: int) -> None: ...
    def decode_frame(self, probs: NDArray[np.float32]) -> NDArray[np.int32]: ...
