#!/usr/bin/env python3
"""PSNR/SSIM/P99 vs x264/x265/AV1 on original RGB at matched bpp."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from commavq_prod.codec import decode_clip_native, encode_clip_native, load_engine  # noqa: E402
from commavq_prod.constants import FPS, H, W  # noqa: E402
from commavq_prod.eval.classical import decode_video, encode_video  # noqa: E402
from commavq_prod.eval.metrics import latency_stats, psnr_ssim  # noqa: E402

OUT = ROOT / "bench_rd.json"


def _structured_rgb(n: int, h: int = H, w: int = W) -> np.ndarray:
    """Smooth moving-scene RGB (not i.i.d. noise)."""
    yy, xx = np.mgrid[0:h, 0:w]
    frames = np.empty((n, h, w, 3), dtype=np.uint8)
    for t in range(n):
        phase = t * 7
        r = (40 + 80 * (np.sin((xx + phase) / 18.0) + 1)).astype(np.uint8)
        g = (40 + 80 * (np.cos((yy - phase) / 14.0) + 1)).astype(np.uint8)
        b = np.full((h, w), 90, dtype=np.uint8)
        cx = (w // 3 + t * 5) % w
        cy = (h // 2 + t * 2) % h
        y0, y1 = max(cy - 12, 0), min(cy + 12, h)
        x0, x1 = max(cx - 18, 0), min(cx + 18, w)
        r[y0:y1, x0:x1] = 220
        g[y0:y1, x0:x1] = 40
        b[y0:y1, x0:x1] = 40
        frames[t, :, :, 0] = r
        frames[t, :, :, 1] = g
        frames[t, :, :, 2] = b
    return frames


def _load_video_rgb(path: Path, n: int) -> np.ndarray:
    with tempfile.TemporaryDirectory() as td:
        raw = Path(td) / "in.raw"
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(path), "-vf", f"scale={W}:{H}",
                "-frames:v", str(n), "-f", "rawvideo", "-pix_fmt", "rgb24", str(raw),
            ],
            check=True,
            capture_output=True,
        )
        data = np.fromfile(raw, dtype=np.uint8)
        nf = data.size // (H * W * 3)
        return data[: nf * H * W * 3].reshape(nf, H, W, 3)


def main() -> None:
    n = int(os.environ.get("RD_FRAMES", "16"))
    load_engine()
    video = os.environ.get("RD_VIDEO")
    if video and Path(video).exists():
        rgb = _load_video_rgb(Path(video), n)
        source = "original_rgb"
        note = f"ffmpeg {video}"
    else:
        rgb = _structured_rgb(n)
        source = "original_rgb"
        note = "structured synthetic rgb (128x256)"

    try:
        from commavq_prod.eval.vqvae import rgb_to_tokens, tokens_to_rgb

        device = "cpu"
        try:
            import torch
            if torch.backends.mps.is_available():
                device = "mps"
        except Exception:
            pass
        tokens = rgb_to_tokens(rgb, device=device).astype(np.int32)
        if tokens.ndim == 3:
            tokens = tokens.reshape(tokens.shape[0], -1)
        rec_n = tokens_to_rgb(tokens, device=device)
    except Exception as e:
        tokens = None
        rec_n = rgb
        note += f"; vq skip ({e})"

    if tokens is not None:
        t0 = time.perf_counter()
        blob = encode_clip_native(tokens, adaptive=True)
        enc_ms = (time.perf_counter() - t0) * 1000
        samples = []
        for _ in range(5):
            t0 = time.perf_counter()
            decode_clip_native(blob)
            samples.append((time.perf_counter() - t0) * 1000)
        lat = latency_stats(samples)
        bits = len(blob) * 8
    else:
        blob = b""
        enc_ms = float("nan")
        lat = latency_stats([0.0])
        bits = 0

    pixels = rgb.shape[0] * rgb.shape[1] * rgb.shape[2]
    neural_bpp = bits / max(pixels, 1)
    psnr_n, ssim_n = psnr_ssim(rec_n[: rgb.shape[0]], rgb[: rec_n.shape[0]])

    grid = [max(neural_bpp * x, 0.01) for x in (0.5, 1.0, 2.0, 4.0)]
    if neural_bpp < 1e-9:
        grid = [0.02, 0.04, 0.08, 0.16]
    classical = {}
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for codec in ("x264", "x265", "av1"):
            pts = []
            for bpp in grid:
                out = td / f"{codec}_{bpp:.4f}.mp4"
                try:
                    _, achieved = encode_video(rgb, out, codec, bpp, fps=FPS)
                    rec = decode_video(out, rgb.shape[0], rgb.shape[1], rgb.shape[2])
                    nn = min(rec.shape[0], rgb.shape[0])
                    p, s = psnr_ssim(rec[:nn], rgb[:nn])
                    pts.append({"target_bpp": bpp, "bpp": achieved, "psnr": p, "ssim": s})
                except Exception as ex:
                    pts.append({"target_bpp": bpp, "error": str(ex)[:300]})
            classical[codec] = pts

    report = {
        "source": source,
        "note": note,
        "n_frames": int(rgb.shape[0]),
        "neural": {
            "bpp": neural_bpp,
            "psnr": psnr_n,
            "ssim": ssim_n,
            "encode_ms": enc_ms,
            "decode_p50_ms": lat.p50_ms,
            "decode_p99_ms": lat.p99_ms,
            "bytes": len(blob),
        },
        "classical": classical,
    }
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2)[:3000])


if __name__ == "__main__":
    main()
