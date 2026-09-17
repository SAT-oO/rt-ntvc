#!/usr/bin/env python3
"""B3 gate: fused Rust decode vs per-frame Python constriction in time loop."""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baseline_python_coder import FrameDecoder, FrameEncoder  # noqa: E402

from commavq_prod.codec import decode_clip, decode_clip_fused, encode_clip, gather_probs, load_codec  # noqa: E402
from commavq_prod.constants import FRAMES, S  # noqa: E402
from commavq_prod.data import synthetic_batch  # noqa: E402
from commavq_prod.train import configure_determinism  # noqa: E402

WARMUP = 1
RUNS = 5


def _baseline_encode(tokens: np.ndarray, probs: np.ndarray) -> bytes:
    enc = FrameEncoder()
    for f in range(FRAMES):
        enc.encode_frame(tokens[f], probs[f])
    return enc.to_bytes()


def bench_baseline_loop(blob: bytes, probs: np.ndarray) -> float:
    """Baseline: 1200x FrameDecoder.decode_frame with precomputed probs (entropy path)."""
    t0 = time.perf_counter()
    dec = FrameDecoder(blob)
    for t in range(FRAMES):
        _ = dec.decode_frame(probs[t])
    return time.perf_counter() - t0


def bench_fused(blob: bytes, model, global_probs, device: str) -> float:
    """Fused: causal predictor in Python + Rust ClipDecoder entropy."""
    t0 = time.perf_counter()
    decode_clip(blob, FRAMES, model, global_probs, device=device)
    return time.perf_counter() - t0


def bench_entropy_bound(blob: bytes, probs: np.ndarray) -> float:
    t0 = time.perf_counter()
    decode_clip_fused(blob, probs)
    return time.perf_counter() - t0


def main() -> None:
    configure_determinism()
    device = "cpu"
    try:
        model, global_probs = load_codec(device=device)
    except FileNotFoundError:
        print("No model weights; skipping e2e bench.")
        return

    tokens = synthetic_batch(1, FRAMES, seed=0)[0]
    probs = gather_probs(tokens, model, global_probs, device=device)
    blob_baseline = _baseline_encode(tokens, probs)
    blob_rust = encode_clip(tokens, model, global_probs, device=device)

    # Sanity: both lossless
    from commavq_prod.codec import decode_clip_fused as dcf

    got_r = dcf(blob_rust, probs)
    assert np.array_equal(got_r.astype(np.int16), tokens)

    for _ in range(WARMUP):
        bench_baseline_loop(blob_baseline, probs)
        bench_fused(blob_rust, model, global_probs, device)

    a_entropy = [bench_baseline_loop(blob_baseline, probs) for _ in range(RUNS)]
    b_entropy = [bench_entropy_bound(blob_rust, probs) for _ in range(RUNS)]
    b_full = [bench_fused(blob_rust, model, global_probs, device) for _ in range(RUNS)]

    a_med = statistics.median(a_entropy)
    b_med = statistics.median(b_entropy)
    speedup = a_med / b_med
    full_med = statistics.median(b_full)

    print(f"entropy_bound baseline_s={a_med:.3f} fused_s={b_med:.3f} speedup={speedup:.2f}x")
    print(f"full_causal_decode fused_s={full_med:.3f} (predictor+entropy)")

    results = ROOT / "RESULTS.md"
    block = f"""
## B3 e2e decode bench

| config | median_s | notes |
|--------|----------|-------|
| A baseline decode (precomputed probs) | {a_med:.3f} | 1200x constriction decode_frame |
| B fused decode (precomputed probs) | {b_med:.3f} | 1x rans.decode FFI |
| entropy-bound speedup B/A | {speedup:.2f}x | gate >= 1.5x |
| B full causal decode | {full_med:.3f} | predictor + ClipDecoder shipped |

Command: `python benches/bench_e2e_decode.py`
"""
    text = results.read_text() if results.exists() else "# Project B results\n"
    if "## B3 e2e decode bench" in text:
        text = text.split("## B3 e2e decode bench")[0].rstrip() + "\n" + block
    else:
        text = text.rstrip() + "\n" + block
    results.write_text(text)


if __name__ == "__main__":
    main()
