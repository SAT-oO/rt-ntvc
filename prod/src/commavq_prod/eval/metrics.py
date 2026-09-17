"""BPP, PSNR, SSIM, LPIPS, latency, RSS."""

from __future__ import annotations

import resource
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator, Optional

import numpy as np

from commavq_prod.constants import BIT_DEPTH, S, SYMBOLS_PER_CLIP


@dataclass
class LatencyStats:
    p50_ms: float
    p99_ms: float
    mean_ms: float


@dataclass
class TokenMetrics:
    bpp: float
    bits_per_token: float
    encode_latency: LatencyStats
    decode_latency: LatencyStats
    throughput_clips_per_s: float
    peak_rss_mb: float


@dataclass
class PixelMetrics:
    psnr: float
    ssim: float
    lpips: float
    decode_p50_ms: float
    decode_p99_ms: float
    encode_ms: Optional[float] = None


def token_bpp(num_bits: float, num_symbols: int = SYMBOLS_PER_CLIP) -> float:
    return num_bits / (num_symbols * BIT_DEPTH / BIT_DEPTH)  # bits per 10-bit symbol slot
    # Actually bpp = bits / (pixels * bit_depth) for pixels; for tokens:
    # bits per token vs 10-bit raw = num_bits / (num_symbols * 10)


def bits_per_token(num_bits: float, num_symbols: int = SYMBOLS_PER_CLIP) -> float:
    return num_bits / num_symbols


def raw_token_bpp() -> float:
    return BIT_DEPTH  # 10 bits per token uncompressed


def latency_stats(samples_ms: list[float]) -> LatencyStats:
    arr = np.asarray(samples_ms, dtype=np.float64)
    return LatencyStats(
        p50_ms=float(np.percentile(arr, 50)),
        p99_ms=float(np.percentile(arr, 99)),
        mean_ms=float(arr.mean()),
    )


@contextmanager
def peak_rss_tracker() -> Generator[list[float], None, None]:
    peak: list[float] = [0.0]
    yield peak
    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak[0] = usage.ru_maxrss / (1024 * 1024)  # macOS: bytes; Linux: KB — normalize below


def peak_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KB, macOS reports bytes
    import sys

    if sys.platform == "darwin":
        return usage / (1024 * 1024)
    return usage / 1024


def psnr_ssim(
    pred: np.ndarray, target: np.ndarray
) -> tuple[float, float]:
    """pred/target: (N, H, W, 3) float [0,1] or uint8."""
    try:
        from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
        import torch

        t = torch.from_numpy(pred).float()
        g = torch.from_numpy(target).float()
        if t.max() > 1.5:
            t = t / 255.0
            g = g / 255.0
        psnr = float(peak_signal_noise_ratio(t, g, data_range=1.0))
        ssim = float(structural_similarity_index_measure(t, g, data_range=1.0))
        return psnr, ssim
    except ImportError:
        mse = np.mean((pred.astype(np.float64) - target.astype(np.float64)) ** 2)
        psnr = 10.0 * np.log10(255.0**2 / max(mse, 1e-12)) if pred.dtype == np.uint8 else -10 * np.log10(max(mse, 1e-12))
        return float(psnr), 0.0


def lpips_distance(pred: np.ndarray, target: np.ndarray) -> float:
    try:
        import lpips
        import torch

        loss_fn = lpips.LPIPS(net="alex")
        t = torch.from_numpy(pred).float()
        g = torch.from_numpy(target).float()
        if t.ndim == 4 and t.shape[-1] == 3:
            t = t.permute(0, 3, 1, 2)
            g = g.permute(0, 3, 1, 2)
        if t.max() > 1.5:
            t = t / 127.5 - 1.0
            g = g / 127.5 - 1.0
        with torch.no_grad():
            d = loss_fn(t, g)
        return float(d.mean().item())
    except ImportError:
        return float("nan")


@dataclass
class TimedRun:
    elapsed_s: float = 0.0
    per_frame_ms: list[float] = field(default_factory=list)


@contextmanager
def timer() -> Generator[TimedRun, None, None]:
    run = TimedRun()
    t0 = time.perf_counter()
    yield run
    run.elapsed_s = time.perf_counter() - t0
