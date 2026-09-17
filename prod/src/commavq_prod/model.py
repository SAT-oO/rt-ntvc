"""NextFramePredictor — production rewrite (same I/O as challenge model.py)."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, Union

import numpy as np
import torch
import torch.nn as nn

from commavq_prod.types import TokenArray

from commavq_prod.constants import (
    D_MODEL,
    FFN_DIM,
    FRAME_COLS,
    FRAME_ROWS,
    N_HEADS,
    N_LAYERS,
    S,
    T,
    VOCAB,
)

CONTEXT_FRAMES = T
TOKENS_PER_FRAME = S


class NextFramePredictor(nn.Module):
    """Transformer encoder: (B, T, S) int tokens -> (B, S, VOCAB) logits."""

    def __init__(self) -> None:
        super().__init__()
        self.token_embed = nn.Embedding(VOCAB, D_MODEL)
        self.row_embed = nn.Embedding(FRAME_ROWS, D_MODEL)
        self.col_embed = nn.Embedding(FRAME_COLS, D_MODEL)
        self.temporal_embed = nn.Embedding(T, D_MODEL)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEADS,
            dim_feedforward=FFN_DIM,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=N_LAYERS,
            norm=nn.LayerNorm(D_MODEL),
        )
        self.output_head = nn.Linear(D_MODEL, VOCAB, bias=False)

        row_idx = torch.arange(FRAME_ROWS).repeat_interleave(FRAME_COLS)
        col_idx = torch.arange(FRAME_COLS).repeat(FRAME_ROWS)
        self.register_buffer("row_idx", row_idx)
        self.register_buffer("col_idx", col_idx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, S) int64 -> logits (B, S, VOCAB)."""
        b, t_len, _s = x.shape
        tok_emb = self.token_embed(x)
        sp_emb = self.row_embed(self.row_idx) + self.col_embed(self.col_idx)
        tp_emb = self.temporal_embed(torch.arange(t_len, device=x.device))
        x_emb = tok_emb + sp_emb[None, None] + tp_emb[None, :, None]
        x_flat = x_emb.reshape(b, t_len * _s, D_MODEL)
        out = self.transformer(x_flat)
        out_last = out[:, -_s:]
        return self.output_head(out_last)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_context(
    tokens: TokenArray, t: int, context_frames: int = T
) -> TokenArray:
    """Return (T, S) context for predicting frame t."""
    start = max(0, t - context_frames)
    avail = tokens[start:t]
    n = len(avail)
    if n == context_frames:
        return avail
    if t > 0:
        pad_frame = tokens[0:1]
    else:
        pad_frame = np.zeros((1, tokens.shape[1]), dtype=tokens.dtype)
    pad = np.repeat(pad_frame, context_frames - n, axis=0)
    return np.concatenate([pad, avail], axis=0)


def build_context_batch(
    tokens_batch: TokenArray, t: int, context_frames: int = T
) -> TokenArray:
    """tokens_batch: (B, num_frames, S) -> (B, T, S)."""
    start = max(0, t - context_frames)
    avail = tokens_batch[:, start:t, :]
    n = avail.shape[1]
    if n == context_frames:
        return avail.copy()
    pad = np.repeat(tokens_batch[:, 0:1, :], context_frames - n, axis=1)
    if n > 0:
        return np.concatenate([pad, avail], axis=1)
    return pad


PathLike = Union[str, Path]


def load_model(path: PathLike, device: str = "cpu") -> NextFramePredictor:
    model = NextFramePredictor().to(device)
    state = torch.load(path, map_location=device, weights_only=True)
    state = {k: v.float() if v.is_floating_point() else v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model


def save_model_f16(model: NextFramePredictor, path: Union[PathLike, BinaryIO]) -> None:
    state_f16 = {
        k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()
    }
    torch.save(state_f16, path)
