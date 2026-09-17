"""Minimal training step and tiny loop — no framework."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from commavq_prod.constants import T
from commavq_prod.types import TokenArray
from commavq_prod.model import NextFramePredictor, build_context_batch


def configure_determinism() -> None:
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def train_step(
    model: NextFramePredictor,
    batch_tokens: TokenArray,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
    frame_index: Optional[int] = None,
) -> float:
    """
    One step: predict frame `frame_index` (default random in [T, num_frames-1]).

    batch_tokens: (B, num_frames, S) int16/int64
    Returns scalar loss.
    """
    model.train()
    b, num_frames, _s = batch_tokens.shape
    t = frame_index if frame_index is not None else int(np.random.randint(T, num_frames))
    ctx = build_context_batch(batch_tokens, t)
    target = batch_tokens[:, t, :].astype(np.int64)

    x = torch.tensor(ctx, dtype=torch.long, device=device)
    y = torch.tensor(target, dtype=torch.long, device=device)

    logits = model(x)
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))

    optimizer.zero_grad(set_to_none=True)
    loss.backward()  # type: ignore[no-untyped-call]
    optimizer.step()
    return float(loss.item())


def train_smoke(
    steps: int = 2,
    batch_size: int = 2,
    device: str = "cpu",
    lr: float = 1e-4,
) -> list[float]:
    """Run `steps` on synthetic data; returns loss history."""
    from commavq_prod.data import synthetic_batch

    configure_determinism()
    model = NextFramePredictor().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses: list[float] = []
    for step in range(steps):
        batch = synthetic_batch(batch_size=batch_size, seed=step)
        losses.append(train_step(model, batch, opt, device=device, frame_index=T + step))
    return losses
