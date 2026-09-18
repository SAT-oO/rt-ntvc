"""Python facade: causal predictor + batched rANS, plus native Engine (one FFI/clip)."""

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

CHUNK_FRAMES: int = FRAMES

_ENGINE = None
_ENGINE_PATH: Optional[Path] = None
_PREDICTOR = None


class _TorchPredictor:
    """MPS/CPU PyTorch forward used as the Engine EP (one FFI owns the loop)."""

    def __init__(self, model, device: str) -> None:
        self.model = model
        self.device = device

    def __call__(self, ctx: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.ascontiguousarray(ctx).reshape(1, T, S).astype(np.int64)).to(
            self.device
        )
        with torch.inference_mode():
            logits = self.model(x)[0].float()
            if self.device == "mps":
                logits = logits.to("cpu")
            return logits.numpy()


def _weight_keys() -> list[str]:
    keys = [
        "token_embed.weight",
        "row_embed.weight",
        "col_embed.weight",
        "temporal_embed.weight",
    ]
    for i in range(6):
        p = f"transformer.layers.{i}"
        keys.extend(
            [
                f"{p}.self_attn.in_proj_weight",
                f"{p}.self_attn.in_proj_bias",
                f"{p}.self_attn.out_proj.weight",
                f"{p}.self_attn.out_proj.bias",
                f"{p}.linear1.weight",
                f"{p}.linear1.bias",
                f"{p}.linear2.weight",
                f"{p}.linear2.bias",
                f"{p}.norm1.weight",
                f"{p}.norm1.bias",
                f"{p}.norm2.weight",
                f"{p}.norm2.bias",
            ]
        )
    keys.extend(
        [
            "transformer.norm.weight",
            "transformer.norm.bias",
            "output_head.weight",
        ]
    )
    return keys


def export_runtime_weights(model_path: Path, out_path: Path) -> Path:
    """Pack f32 tensors for the Rust session (magic RTNVW001)."""
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    blobs: list[np.ndarray] = []
    for k in _weight_keys():
        t = state[k]
        if t.is_floating_point():
            t = t.float()
        blobs.append(t.detach().cpu().contiguous().numpy().astype(np.float32, copy=False).reshape(-1))
    data = np.concatenate(blobs)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(b"RTNVW001")
        f.write(np.uint32(data.size).tobytes())
        f.write(data.tobytes())
    return out_path


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
    """tokens (num_frames, S) -> bitstream. Single rANS FFI per clip (Python predictor)."""
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
    """Baseline causal decode: 1 init FFI + one decode_frame FFI per frame."""
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


def decode_clip_fused(data: bytes, probs: FloatArray) -> IntArray:
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


def runtime_weight_path(model_path: Path) -> Path:
    return model_path.with_suffix(".rtnv.bin")


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
    packed = runtime_weight_path(Path(model_path))
    if not packed.exists():
        export_runtime_weights(Path(model_path), packed)
    return model, global_probs


def load_engine(
    model_path: Optional[Path] = None,
    freq_path: Optional[Path] = None,
) -> object:
    """Native Rust Engine (neural + rANS). One FFI per clip."""
    global _ENGINE, _ENGINE_PATH, _PREDICTOR
    if model_path is None or freq_path is None:
        mp, fp = default_resource_paths()
        model_path = model_path or mp
        freq_path = freq_path or fp
    packed = runtime_weight_path(Path(model_path))
    if not packed.exists():
        export_runtime_weights(Path(model_path), packed)
    if _ENGINE is None or _ENGINE_PATH != packed:
        g = load_global_freq(freq_path)
        _ENGINE = _rans.Engine(str(packed), g)
        _ENGINE_PATH = packed
        device = "cpu"
        try:
            if torch.backends.mps.is_available():
                device = "mps"
        except Exception:
            device = "cpu"
        model = load_model(str(model_path), device=device)
        _PREDICTOR = _TorchPredictor(model, device)
        _ENGINE.set_predictor(_PREDICTOR)
        if device == "mps":
            dummy = np.zeros((T * S,), dtype=np.int32)
            _PREDICTOR(dummy)
            torch.mps.synchronize()
    return _ENGINE


def encode_clip_native(
    tokens: TokenArray, adaptive: bool = True, cpu: bool = False, neural: bool = True
) -> bytes:
    eng = load_engine()
    return bytes(eng.encode(tokens.astype(np.int32), adaptive=adaptive, cpu=cpu, neural=neural))


def decode_clip_native(data: bytes, cpu: bool = False) -> IntArray:
    eng = load_engine()
    out = np.asarray(eng.decode(data, cpu=cpu), dtype=np.int32)
    if not cpu:
        try:
            if torch.backends.mps.is_available():
                torch.mps.synchronize()
        except Exception:
            pass
    return out
