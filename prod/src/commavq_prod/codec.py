"""Python facade: causal predictor + batched rANS (one encode FFI per clip)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from commavq_prod import _rans
from commavq_prod.constants import FRAMES, S, T, VOCAB
from commavq_prod.model import NextFramePredictor, build_context_batch, load_model
from commavq_prod.rate import apply_temperature, load_global_freq
from commavq_prod.types import FloatArray, IntArray, TokenArray

# One-shot (T,S,V) float32 ≈ 629 MB; use one FFI when RAM allows.
CHUNK_FRAMES: int = FRAMES


def _softmax_probs(model: NextFramePredictor, ctx: TokenArray, device: str) -> FloatArray:
    x = torch.tensor(ctx, dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(x)
        return torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float32)


def gather_probs(
    tokens: TokenArray,
    model: NextFramePredictor,
    global_probs: FloatArray,
    device: str = "cpu",
    tau_per_frame: Optional[FloatArray] = None,
) -> FloatArray:
    """(num_frames, S, VOCAB) probs for entropy coding (encode / entropy-bound decode)."""
    num_frames = tokens.shape[0]
    probs = np.zeros((num_frames, S, VOCAB), dtype=np.float32)
    global_tile = np.broadcast_to(global_probs[None, :], (S, VOCAB)).astype(np.float32)
    probs[0] = global_tile

    batch = tokens[None, :, :]
    for t in range(1, num_frames):
        ctx = build_context_batch(batch, t)
        p = _softmax_probs(model, ctx, device)[0]
        if tau_per_frame is not None:
            p = apply_temperature(p, float(tau_per_frame[t]))
        probs[t] = p

    return probs


def encode_clip(
    tokens: TokenArray,
    model: NextFramePredictor,
    global_probs: FloatArray,
    device: str = "cpu",
    tau_per_frame: Optional[FloatArray] = None,
) -> bytes:
    """tokens (num_frames, S) -> bitstream. Single rANS FFI per clip."""
    probs = gather_probs(tokens, model, global_probs, device, tau_per_frame)
    return _rans.encode(tokens.astype(np.int32), probs)


def decode_clip(
    data: bytes,
    num_frames: int,
    model: NextFramePredictor,
    global_probs: FloatArray,
    device: str = "cpu",
    tau_per_frame: Optional[FloatArray] = None,
) -> IntArray:
    """
    Causal decode: ClipDecoder keeps rANS state; one init FFI + frame decodes in Rust.

    Predictor steps in Python; entropy stays in Rust (no per-frame Python constriction).
    """
    global_tile = np.broadcast_to(global_probs[None, :], (S, VOCAB)).astype(np.float32)
    dec = _rans.ClipDecoder(data, num_frames)
    tokens = np.zeros((num_frames, S), dtype=np.int32)
    batch = tokens[None, :, :]

    for t in range(num_frames):
        if t == 0:
            probs_t = global_tile
        else:
            ctx = build_context_batch(batch, t)
            probs_t = _softmax_probs(model, ctx, device)[0]
            if tau_per_frame is not None:
                probs_t = apply_temperature(probs_t, float(tau_per_frame[t]))
        tokens[t] = dec.decode_frame(probs_t)
        batch[0, t] = tokens[t]

    return tokens


def decode_clip_fused(
    data: bytes,
    probs: FloatArray,
) -> IntArray:
    """Entropy-bound path: precomputed probs, single rANS decode FFI."""
    return np.asarray(_rans.decode(data, probs), dtype=np.int32)


def default_resource_paths() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parents[3]
    for sub in ("commavq_compressor", "libs"):
        base = root / sub / "resource"
        model = base / "model.pt"
        freq = base / "global_freq.npy"
        if model.exists() and freq.exists():
            return model, freq
    raise FileNotFoundError("model.pt and global_freq.npy not found")


def load_codec(
    model_path: Optional[Path] = None,
    freq_path: Optional[Path] = None,
    device: str = "cpu",
) -> tuple[NextFramePredictor, FloatArray]:
    if model_path is None or freq_path is None:
        mp, fp = default_resource_paths()
        model_path = model_path or mp
        freq_path = freq_path or fp
    model = load_model(str(model_path), device=device)
    global_probs = load_global_freq(freq_path)
    return model, global_probs
