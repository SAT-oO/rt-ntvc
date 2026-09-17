#!/usr/bin/env python3
"""B4 gate: adaptive per-frame tau vs global tau."""

from __future__ import annotations

import gc
import os
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from commavq_prod import _rans  # noqa: E402
from commavq_prod.codec import gather_probs, load_codec  # noqa: E402
from commavq_prod.constants import FRAMES, S  # noqa: E402
from commavq_prod.data import synthetic_batch  # noqa: E402
from commavq_prod.rate import assign_adaptive_tau, header_bits, nll_bits, probs_with_tau  # noqa: E402

N_CLIPS = 8
GATE_BITS_RATIO = 0.92
# ponytail: 120 frames keeps RSS ~150 MB/clip; set BENCH_FRAMES=1200 for full clip
BENCH_FRAMES = int(os.environ.get("BENCH_FRAMES", "120"))


def bits_for_clip(
    tokens: np.ndarray,
    base_probs: np.ndarray,
    global_probs: np.ndarray,
    tau: np.ndarray,
) -> float:
    n_frames = tokens.shape[0]
    probs = np.empty_like(base_probs)
    global_tile = np.broadcast_to(global_probs[None, :], (S, len(global_probs)))
    for t in range(n_frames):
        probs[t] = probs_with_tau(base_probs[t : t + 1], float(tau[t]), global_tile)[0]
    blob = _rans.encode(tokens.astype(np.int32), probs)
    del probs
    return len(blob) * 8 + header_bits(n_frames)


def mean_nll(tokens: np.ndarray, base_probs: np.ndarray, global_probs: np.ndarray, tau: np.ndarray) -> float:
    n_frames = tokens.shape[0]
    global_tile = np.broadcast_to(global_probs[None, :], (S, len(global_probs)))
    total = 0.0
    for t in range(n_frames):
        p = probs_with_tau(base_probs[t : t + 1], float(tau[t]), global_tile)
        total += nll_bits(tokens[t : t + 1], p)
    return total / (n_frames * S)


def main() -> None:
    n_frames = min(BENCH_FRAMES, FRAMES)
    try:
        model, global_probs = load_codec(device="cpu")
    except FileNotFoundError:
        print("No weights; using uniform probs.")
        model = None
        global_probs = np.full(1024, 1 / 1024, dtype=np.float32)

    global_bits: list[float] = []
    adaptive_bits: list[float] = []
    nll_global: list[float] = []
    nll_adaptive: list[float] = []

    tau_g = np.ones(n_frames, dtype=np.float32)

    for i in range(N_CLIPS):
        tokens = synthetic_batch(1, n_frames, seed=i)[0]
        if model is not None:
            base = gather_probs(tokens, model, global_probs, device="cpu")
        else:
            base = np.full((n_frames, S, len(global_probs)), 1.0 / len(global_probs), np.float32)

        tau_a, _cb = assign_adaptive_tau(tokens, base, global_probs, global_tau=1.0)

        global_bits.append(bits_for_clip(tokens, base, global_probs, tau_g))
        adaptive_bits.append(bits_for_clip(tokens, base, global_probs, tau_a))
        nll_global.append(mean_nll(tokens, base, global_probs, tau_g))
        nll_adaptive.append(mean_nll(tokens, base, global_probs, tau_a))

        del base, tokens, tau_a
        gc.collect()

    g_med = statistics.median(global_bits)
    a_med = statistics.median(adaptive_bits)
    ratio = a_med / g_med
    nll_ratio = statistics.median(nll_adaptive) / statistics.median(nll_global)

    print(f"frames={n_frames} clips={N_CLIPS}")
    print(f"bits global={g_med:.0f} adaptive={a_med:.0f} ratio={ratio:.4f} gate<={GATE_BITS_RATIO}")
    print(f"nll_ratio={nll_ratio:.4f} (target [0.99,1.01])")

    results = ROOT / "RESULTS.md"
    block = f"""
## B4 adaptive rate ({N_CLIPS} clips, {n_frames} frames)

| metric | global tau | adaptive tau |
|--------|------------|--------------|
| median bits | {g_med:.0f} | {a_med:.0f} |
| bits ratio adaptive/global | {ratio:.4f} | gate <= {GATE_BITS_RATIO} |
| mean NLL ratio | {nll_ratio:.4f} | target [0.99, 1.01] |

Command: `python benches/bench_adaptive_rate.py` (BENCH_FRAMES={n_frames})
"""
    text = results.read_text() if results.exists() else "# Project B results\n"
    if "## B4 adaptive rate" in text:
        text = text.split("## B4 adaptive rate")[0].rstrip() + "\n" + block
    else:
        text = text.rstrip() + "\n" + block
    results.write_text(text)


if __name__ == "__main__":
    main()
