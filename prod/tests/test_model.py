"""Unit tests for NextFramePredictor."""

import numpy as np
import torch

from commavq_prod.constants import S, T, VOCAB
from commavq_prod.model import NextFramePredictor, build_context, build_context_batch
from commavq_prod.train import train_smoke


def test_forward_shapes() -> None:
    model = NextFramePredictor()
    x = torch.randint(0, VOCAB, (2, T, S))
    logits = model(x)
    assert logits.shape == (2, S, VOCAB)


def test_param_count() -> None:
    model = NextFramePredictor()
    n = model.param_count()
    assert 4.4e6 <= n <= 4.6e6


def test_build_context() -> None:
    tokens = np.zeros((20, S), dtype=np.int16)
    ctx = build_context(tokens, t=5)
    assert ctx.shape == (T, S)


def test_build_context_batch() -> None:
    batch = np.zeros((3, 20, S), dtype=np.int16)
    ctx = build_context_batch(batch, t=10)
    assert ctx.shape == (3, T, S)


def test_train_step_smoke() -> None:
    losses = train_smoke(steps=2, batch_size=2, device="cpu")
    assert len(losses) == 2
    assert all(np.isfinite(losses))
