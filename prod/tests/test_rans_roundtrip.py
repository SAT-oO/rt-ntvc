"""Hypothesis round-trip tests for Rust rANS."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

rans = pytest.importorskip("commavq_prod._rans")


@given(
    t=st.integers(min_value=1, max_value=32),
    s=st.integers(min_value=1, max_value=64),
    vocab=st.integers(min_value=2, max_value=256),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=25, deadline=None)
def test_encode_decode_roundtrip(t: int, s: int, vocab: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    symbols = rng.integers(0, vocab, size=(t, s), dtype=np.int32)
    logits = rng.standard_normal((t, s, vocab)).astype(np.float32)
    probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)

    blob = rans.encode(symbols, probs)
    got = rans.decode(blob, probs)
    assert np.array_equal(got, symbols)


def test_clip_decoder_frame_order() -> None:
    t, s, vocab = 4, 8, 32
    rng = np.random.default_rng(0)
    symbols = rng.integers(0, vocab, size=(t, s), dtype=np.int32)
    probs = np.full((t, s, vocab), 1.0 / vocab, dtype=np.float32)
    blob = rans.encode(symbols, probs)

    dec = rans.ClipDecoder(blob, t)
    out = np.zeros((t, s), dtype=np.int32)
    for f in range(t):
        out[f] = dec.decode_frame(probs[f])
    assert np.array_equal(out, symbols)
