#!/usr/bin/env python3
"""B2 gate: Rust rANS vs frozen baseline_python_coder."""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baseline_python_coder import FrameDecoder, FrameEncoder  # noqa: E402

from commavq_prod import _rans  # noqa: E402
from commavq_prod.constants import FRAMES, S, VOCAB  # noqa: E402

WARMUP = 2
RUNS = 10
GATE_RATIO = 0.17


def _synthetic_clip(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    symbols = rng.integers(0, VOCAB, size=(FRAMES, S), dtype=np.int32)
    logits = rng.standard_normal((FRAMES, S, VOCAB)).astype(np.float32)
    # a few spikes
    for _ in range(20):
        logits[rng.integers(0, FRAMES), rng.integers(0, S), rng.integers(0, VOCAB)] += 4.0
    probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)
    return symbols, probs.astype(np.float32)


def bench_baseline(symbols: np.ndarray, probs: np.ndarray) -> float:
    t0 = time.perf_counter()
    enc = FrameEncoder()
    for f in range(FRAMES):
        enc.encode_frame(symbols[f], probs[f])
    blob = enc.to_bytes()
    dec = FrameDecoder(blob)
    for f in range(FRAMES):
        got = dec.decode_frame(probs[f])
        assert np.array_equal(got, symbols[f])
    return time.perf_counter() - t0


def bench_rust(symbols: np.ndarray, probs: np.ndarray) -> float:
    t0 = time.perf_counter()
    blob = _rans.encode(symbols, probs)
    got = _rans.decode(blob, probs)
    assert np.array_equal(got, symbols)
    return time.perf_counter() - t0


def main() -> None:
    symbols, probs = _synthetic_clip()
    for _ in range(WARMUP):
        bench_baseline(symbols, probs)
        bench_rust(symbols, probs)

    base_times = [bench_baseline(symbols, probs) for _ in range(RUNS)]
    rust_times = [bench_rust(symbols, probs) for _ in range(RUNS)]

    base_med = statistics.median(base_times)
    rust_med = statistics.median(rust_times)
    ratio = rust_med / base_med

    print(f"baseline_median_s={base_med:.4f}")
    print(f"rust_median_s={rust_med:.4f}")
    print(f"ratio={ratio:.4f} gate={GATE_RATIO}")
    print(f"PASS={ratio <= GATE_RATIO}")

    results = ROOT / "RESULTS.md"
    block = f"""
## B2 entropy bench ({time.strftime('%Y-%m-%d')})

| coder | median_s | notes |
|-------|----------|-------|
| baseline_python_coder | {base_med:.4f} | 1200x FrameEncoder |
| rust_rans | {rust_med:.4f} | 1x encode + 1x decode |
| ratio | {ratio:.4f} | gate <= {GATE_RATIO} |

**Gate miss notes:** Hot path is per-symbol CDF quantize ({FRAMES}x{S} symbols). Optimized
zero-copy numpy, batched CDF tables, inlined rANS; constriction still faster on this clip.
Next levers: parallel CDF build, shared frame-level tables, PGO.

Command: `python benches/bench_entropy.py`
Hardware: local dev machine. Warmup={WARMUP}, runs={RUNS}.
"""
    if results.exists():
        text = results.read_text()
        if "## B2 entropy bench" in text:
            text = text.split("## B2 entropy bench")[0].rstrip() + "\n" + block
        else:
            text = text.rstrip() + "\n" + block
    else:
        text = "# Project B results\n" + block
    results.write_text(text)


if __name__ == "__main__":
    main()
