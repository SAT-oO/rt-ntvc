#!/usr/bin/env python3
"""Per-frame decode P50/P99 for the native session."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from commavq_prod.codec import decode_clip_native, encode_clip_native, load_engine  # noqa: E402
from commavq_prod.data import synthetic_batch  # noqa: E402
from commavq_prod.eval.metrics import latency_stats  # noqa: E402

OUT = ROOT / "bench_latency.json"


def main() -> None:
    n = int(__import__("os").environ.get("BENCH_FRAMES", "32"))
    load_engine()
    tok = synthetic_batch(1, n, seed=3)[0].astype(np.int32)
    blob = encode_clip_native(tok, adaptive=False)
    samples = []
    for _ in range(8):
        t0 = time.perf_counter()
        decode_clip_native(blob)
        samples.append((time.perf_counter() - t0) * 1000 / n)
    st = latency_stats(samples)
    out = {"per_frame_ms": samples, "p50": st.p50_ms, "p99": st.p99_ms, "n_frames": n}
    OUT.write_text(json.dumps(out, indent=2))
    print(out)


if __name__ == "__main__":
    main()
