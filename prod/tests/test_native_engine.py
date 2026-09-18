"""Native engine roundtrip (weights required)."""

import numpy as np
import pytest

from commavq_prod.codec import decode_clip_native, encode_clip_native, load_engine
from commavq_prod.constants import S


def test_native_roundtrip_short() -> None:
    try:
        load_engine()
    except FileNotFoundError:
        pytest.skip("no model weights")
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, 1024, size=(12, S), dtype=np.int32)
    blob = encode_clip_native(tokens, adaptive=False)
    got = decode_clip_native(blob)
    assert got.shape == tokens.shape
    assert np.array_equal(got, tokens)


def test_native_adaptive_lossless() -> None:
    try:
        load_engine()
    except FileNotFoundError:
        pytest.skip("no model weights")
    rng = np.random.default_rng(1)
    tokens = rng.integers(0, 1024, size=(8, S), dtype=np.int32)
    blob = encode_clip_native(tokens, adaptive=True)
    got = decode_clip_native(blob)
    assert np.array_equal(got, tokens)
