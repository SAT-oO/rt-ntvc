"""Adaptive rate tests."""

import numpy as np

from commavq_prod.constants import FRAMES, S, VOCAB
from commavq_prod.rate import assign_adaptive_tau, nll_bits, probs_with_tau


def test_nll_uniform_is_10_bits() -> None:
    tokens = np.zeros((8, S), dtype=np.int16)
    probs = np.full((8, S, VOCAB), 1.0 / VOCAB, dtype=np.float32)
    bits = nll_bits(tokens, probs)
    assert abs(bits - 8 * S * 10.0) < 1e-3


def test_adaptive_tau_shape() -> None:
    rng = np.random.default_rng(1)
    n_frames = 16
    tokens = rng.integers(0, VOCAB, size=(n_frames, S), dtype=np.int16)
    base = np.full((n_frames, S, VOCAB), 1.0 / VOCAB, dtype=np.float32)
    global_p = np.full(VOCAB, 1.0 / VOCAB, dtype=np.float32)
    tau, codebook = assign_adaptive_tau(tokens, base, global_p, global_tau=1.0)
    assert tau.shape == (n_frames,)
    assert codebook.shape == (n_frames,)
