"""ffmpeg x264 / x265 / AV1 encode-decode at matched bpp."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np


def _run(cmd: list[str]) -> None:
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        err = (r.stderr or b"").decode("utf-8", "replace")[-800]
        raise RuntimeError(err)


def _kbps(target_bpp: float, h: int, w: int, n: int, fps: int) -> float:
    bits_per_sec = target_bpp * w * h * fps
    return max(bits_per_sec / 1000.0, 32.0)


def encode_video(
    frames: np.ndarray,
    out_path: Path,
    codec: str,
    target_bpp: float,
    fps: int = 20,
) -> tuple[Path, float]:
    """
    frames: (N, H, W, 3) uint8
    Returns (bitstream path, achieved bpp).
    """
    h, w = frames.shape[1], frames.shape[2]
    pixels = frames.shape[0] * h * w
    kbps = _kbps(target_bpp, h, w, frames.shape[0], fps)

    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "in.raw"
        frames.tofile(raw)

        def enc_cmd(rate_k: float) -> list[str]:
            rate = f"{rate_k:.3f}k"
            common_in = [
                "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{w}x{h}", "-r", str(fps), "-i", str(raw),
            ]
            if codec == "x264":
                return common_in + [
                    "-c:v", "libx264", "-preset", "medium", "-tune", "zerolatency",
                    "-pix_fmt", "yuv420p",
                    "-b:v", rate, "-maxrate", rate, "-bufsize", str(max(float(rate_k) * 2, 64)) + "k",
                    str(out_path),
                ]
            if codec == "x265":
                return common_in + [
                    "-c:v", "libx265", "-preset", "medium", "-x265-params", "bframes=0",
                    "-pix_fmt", "yuv420p",
                    "-b:v", rate, "-maxrate", rate, "-bufsize", str(max(float(rate_k) * 2, 64)) + "k",
                    str(out_path),
                ]
            return common_in + [
                "-c:v", "libsvtav1", "-preset", "6", "-b:v", rate,
                "-pix_fmt", "yuv420p",
                str(out_path),
            ]

        rate = kbps
        achieved = 0.0
        last_ok = None
        for _ in range(4):
            rate = float(min(max(rate, 32.0), 8000.0))
            try:
                _run(enc_cmd(rate))
            except RuntimeError:
                if last_ok is not None:
                    break
                raise
            bits = out_path.stat().st_size * 8
            achieved = bits / max(pixels, 1)
            last_ok = achieved
            if target_bpp <= 0:
                break
            err = abs(achieved - target_bpp) / target_bpp
            if err <= 0.15:
                break
            rate *= target_bpp / max(achieved, 1e-12)
        return out_path, achieved


def decode_video(path: Path, num_frames: int, h: int, w: int) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out.raw"
        _run([
            "ffmpeg", "-y", "-i", str(path),
            "-f", "rawvideo", "-pix_fmt", "rgb24", str(out),
        ])
        data = np.fromfile(out, dtype=np.uint8)
        n = min(num_frames, data.size // (h * w * 3))
        return data[: n * h * w * 3].reshape(n, h, w, 3)
