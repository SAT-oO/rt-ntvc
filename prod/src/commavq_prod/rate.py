"""Frame-level adaptive rate: per-frame temperature with matched mean NLL."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from commavq_prod.constants import S, VOCAB
from commavq_prod.types import FloatArray, TokenArray

TAU_GRID = np.array([0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15, 1.20], dtype=np.float64)
TAU_HEADER_BITS_PER_FRAME = 8  # codebook index 0..len(TAU_GRID)-1


def load_global_freq(path: Path | str) -> FloatArray:
    freq = np.load(path)
    p = freq.astype(np.float64)
    p = np.clip(p, 1e-12, None)
    return (p / p.sum()).astype(np.float32)


def apply_temperature(probs: FloatArray, tau: float) -> FloatArray:
    """probs (..., VOCAB) -> tempered softmax."""
    if tau == 1.0:
        return probs
    logits = np.log(np.clip(probs, 1e-12, None))
    scaled = logits / tau
    scaled -= scaled.max(axis=-1, keepdims=True)
    exp = np.exp(scaled)
    return (exp / exp.sum(axis=-1, keepdims=True)).astype(np.float32)


def nll_bits(tokens: TokenArray, probs: FloatArray) -> float:
    """Cross-entropy in bits for tokens (T,S) and probs (T,S,V)."""
    flat_tok = tokens.reshape(-1).astype(np.int64)
    flat_p = probs.reshape(-1, probs.shape[-1])
    p = np.clip(flat_p[np.arange(flat_tok.shape[0]), flat_tok], 1e-12, 1.0)
    return float(-np.log2(p).sum())


def probs_with_tau(
    base_probs: FloatArray,
    tau: float,
    global_tile: FloatArray,
    mix_alpha: float = 1.0,
) -> FloatArray:
    """Apply temperature and optional mix with global marginal."""
    out = apply_temperature(base_probs, tau)
    if mix_alpha < 1.0:
        out = mix_alpha * out + (1.0 - mix_alpha) * global_tile
        out /= out.sum(axis=-1, keepdims=True)
    return out.astype(np.float32)


def assign_adaptive_tau(
    tokens: TokenArray,
    base_probs: FloatArray,
    global_probs: FloatArray,
    global_tau: float = 1.0,
    nll_tolerance: float = 0.01,
) -> tuple[FloatArray, np.ndarray]:
    """
    Two-pass: pick per-frame tau from TAU_GRID minimizing bits s.t. mean NLL
    within +/- nll_tolerance of global-tau baseline.

    Returns (tau_per_frame (FRAMES,), codebook_ids (FRAMES,) uint8).
    """
    num_frames = tokens.shape[0]
    global_tile = np.broadcast_to(global_probs[None, :], (S, VOCAB)).astype(np.float32)

    baseline_probs = probs_with_tau(base_probs, global_tau, global_tile)
    baseline_nll = nll_bits(tokens, baseline_probs) / (num_frames * S)
    target_lo = baseline_nll * (1.0 - nll_tolerance)
    target_hi = baseline_nll * (1.0 + nll_tolerance)

    tau_out = np.full(num_frames, global_tau, dtype=np.float64)
    codebook = np.zeros(num_frames, dtype=np.uint8)

    for t in range(num_frames):
        best_tau = global_tau
        best_bits = float("inf")
        tok_f = tokens[t : t + 1]
        base_f = base_probs[t : t + 1]

        for gi, tau in enumerate(TAU_GRID):
            p = probs_with_tau(base_f, float(tau), global_tile)
            frame_nll = nll_bits(tok_f, p) / S
            if frame_nll < target_lo or frame_nll > target_hi:
                continue
            # ponytail: approximate bits via cross-entropy (exact rANS varies slightly)
            bits = frame_nll * S
            if bits < best_bits:
                best_bits = bits
                best_tau = float(tau)
                codebook[t] = gi

        tau_out[t] = best_tau

    return tau_out.astype(np.float32), codebook


def header_bits(num_frames: int) -> float:
    return num_frames * TAU_HEADER_BITS_PER_FRAME
