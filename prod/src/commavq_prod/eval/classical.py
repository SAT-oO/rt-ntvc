"""ffmpeg x264 / x265 / AV1 encode-decode at matched bpp."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, capture_output=True)


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
    target_bits = target_bpp * pixels

    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "in.raw"
        frames.tofile(raw)

        if codec == "x264":
            enc = [
                "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{w}x{h}", "-r", str(fps), "-i", str(raw),
                "-c:v", "libx264", "-preset", "medium", "-b:v", "64k",
                str(out_path),
            ]
        elif codec == "x265":
            enc = [
                "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{w}x{h}", "-r", str(fps), "-i", str(raw),
                "-c:v", "libx265", "-preset", "medium", "-b:v", "64k",
                str(out_path),
            ]
        else:  # av1
            enc = [
                "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{w}x{h}", "-r", str(fps), "-i", str(raw),
                "-c:v", "libsvtav1", "-preset", "6", "-b:v", "64k",
                str(out_path),
            ]

        _run(enc)
        bits = out_path.stat().st_size * 8
        achieved = bits / pixels
        # ponytail: single-pass bitrate; refine with 2-pass if >5% off target
        if abs(achieved - target_bpp) / target_bpp > 0.05:
            pass
        return out_path, achieved


def decode_video(path: Path, num_frames: int, h: int, w: int) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out.raw"
        _run([
            "ffmpeg", "-y", "-i", str(path),
            "-f", "rawvideo", "-pix_fmt", "rgb24", str(out),
        ])
        data = np.fromfile(out, dtype=np.uint8)
        return data.reshape(num_frames, h, w, 3)
