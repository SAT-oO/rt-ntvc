#!/usr/bin/env python3
"""Production gates R1–R4. Writes prod/bench_report.json. Synthetic does not pass R3."""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from commavq_prod.codec import (  # noqa: E402
    decode_clip,
    decode_clip_native,
    encode_clip,
    encode_clip_native,
    load_codec,
    load_engine,
)
from commavq_prod.constants import FPS, H, S, W  # noqa: E402
from commavq_prod.data import load_heldout_tokens  # noqa: E402

REPORT = ROOT / "bench_report.json"
RESULTS = ROOT / "RESULTS.md"
RD = ROOT / "bench_rd.json"


def _write_results_block(title: str, body: str) -> None:
    text = RESULTS.read_text() if RESULTS.exists() else "# Project B results\n"
    if title in text:
        text = text.split(title)[0].rstrip() + "\n" + body
    else:
        text = text.rstrip() + "\n" + body
    RESULTS.write_text(text)


def _med(fn, n=3):
    xs = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        xs.append(time.perf_counter() - t0)
    return statistics.median(xs)


def main() -> None:
    report: dict = {"gates": {}, "notes": []}
    try:
        model, global_probs = load_codec(device="cpu")
        load_engine()
    except FileNotFoundError as e:
        print(f"weights missing: {e}")
        report["gates"]["r1"] = {"pass": False, "reason": "no weights"}
        REPORT.write_text(json.dumps(report, indent=2))
        return

    from commavq_prod import _rans

    n_frames = int(os.environ.get("BENCH_FRAMES", "32"))
    tokens, real = load_heldout_tokens(n_frames)
    if tokens.shape[0] < n_frames:
        n_frames = int(tokens.shape[0])
    tokens = tokens[:n_frames, :S].astype(np.int32)

    native_blob_cpu = encode_clip_native(tokens, adaptive=False, cpu=True)
    got = decode_clip_native(native_blob_cpu, cpu=True)
    hot = int(_rans.hot_allocs())
    r1_pass = (
        hasattr(_rans, "Engine")
        and (ROOT / "rust_rans" / "include" / "rtnv.h").exists()
        and (ROOT / "rust_rans" / "include" / "rtnv.hpp").exists()
        and hot == 0
        and bool(np.array_equal(got, tokens))
    )
    report["gates"]["r1"] = {
        "pass": r1_pass,
        "hot_allocs": hot,
        "engine": True,
        "roundtrip": bool(np.array_equal(got, tokens)),
        "c_header": True,
    }

    blob_py = encode_clip(tokens, model, global_probs, device="cpu")
    decode_clip(blob_py, n_frames, model, global_probs, device="cpu")
    native_blob = encode_clip_native(tokens, adaptive=False)
    decode_clip_native(native_blob)

    base = _med(lambda: decode_clip(blob_py, n_frames, model, global_probs, device="cpu"))
    neu = _med(lambda: decode_clip_native(native_blob))
    speed_native = base / max(neu, 1e-9)
    r2_pass = speed_native >= 1.5
    report["gates"]["r2"] = {
        "pass": r2_pass,
        "baseline_cpu_s": base,
        "native_1ffi_s": neu,
        "speedup": speed_native,
        "n_frames": n_frames,
        "ffi_native": 1,
        "ffi_baseline": n_frames,
    }

    b_g = encode_clip_native(tokens, adaptive=False, neural=False)
    b_a = encode_clip_native(tokens, adaptive=True, neural=True)
    ratio = len(b_a) / max(len(b_g), 1)
    got_a = decode_clip_native(b_a)
    lossless = bool(np.array_equal(got_a, tokens))
    r3_pass = real and lossless and ratio <= 0.88
    report["gates"]["r3"] = {
        "pass": r3_pass,
        "bits_ratio": ratio,
        "global_bytes": len(b_g),
        "adaptive_bytes": len(b_a),
        "lossless": lossless,
        "n_frames": n_frames,
        "synthetic": not real,
    }

    r4 = {"pass": False, "reason": "run benches/bench_rd.py"}
    if RD.exists():
        rd = json.loads(RD.read_text())
        n = rd.get("neural", {})
        cl = rd.get("classical", {})
        have = all(
            k in cl
            and cl[k]
            and sum(1 for p in cl[k] if "psnr" in p) >= 2
            for k in ("x264", "x265", "av1")
        )
        r4 = {
            "pass": bool(
                have
                and n.get("psnr") is not None
                and n.get("ssim") is not None
                and n.get("decode_p99_ms") is not None
                and rd.get("source") == "original_rgb"
            ),
            "neural_bpp": n.get("bpp"),
            "neural_psnr": n.get("psnr"),
            "neural_ssim": n.get("ssim"),
            "decode_p99_ms": n.get("decode_p99_ms"),
            "classical": {k: cl.get(k) for k in ("x264", "x265", "av1")},
        }
    report["gates"]["r4"] = r4

    per_frame_ms = (neu / n_frames) * 1000
    report["latency"] = {
        "native_mean_ms": per_frame_ms,
        "budget_ms": 1000.0 / FPS,
        "hw": H,
        "ww": W,
        "s": S,
        "real_tokens": real,
    }

    REPORT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["gates"], indent=2))

    _write_results_block(
        "## Gates R1–R4",
        f"""
## Gates R1–R4 ({n_frames} frames, real={real})

| gate | pass | metric |
|------|------|--------|
| R1 zero-alloc engine + C ABI | {r1_pass} | hot_allocs={hot} |
| R2 e2e decode >= 1.5x | {r2_pass} | {speed_native:.2f}x cpu={base:.3f}s native={neu:.3f}s |
| R3 adaptive bits <= 0.88 | {r3_pass} | ratio={ratio:.4f} lossless={lossless} |
| R4 PSNR/SSIM/P99 vs classical | {r4.get("pass")} | {json.dumps({k: r4.get(k) for k in ("neural_bpp","neural_psnr","decode_p99_ms")})} |

Command: `python benches/bench_gates.py`
""",
    )


if __name__ == "__main__":
    main()
