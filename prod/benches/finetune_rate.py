#!/usr/bin/env python3
"""Path C: fine-tune predictor with CE (lossless rate). Eval bits on held-out clip."""

from __future__ import annotations

import tarfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from commavq_prod.codec import default_resource_paths, export_runtime_weights
from commavq_prod.constants import S, T
from commavq_prod.model import NextFramePredictor, build_context_batch, load_model

TAR = Path.home() / ".cache/huggingface/hub/datasets--commaai--commavq/snapshots"
CACHE = Path(__file__).resolve().parents[1] / ".cache"
HELDOUT = CACHE / "heldout_tokens.npy"
OUT_PT = CACHE / "model_ft.pt"


def _tar() -> Path:
    snaps = sorted(TAR.glob("*/data-0000.tar.gz"))
    if not snaps:
        raise FileNotFoundError("commaVQ shard not cached")
    return snaps[-1]


def load_train_clips(n: int = 8, skip_first: bool = True) -> list[np.ndarray]:
    clips = []
    with tarfile.open(_tar(), "r:gz") as tar:
        names = sorted(m.name for m in tar.getmembers() if m.name.endswith(".token.npy"))
        if skip_first:
            names = names[1:]
        for name in names[:n]:
            import io
            f = tar.extractfile(name)
            tok = np.load(io.BytesIO(f.read())).astype(np.int16)
            if tok.ndim == 3:
                tok = tok.reshape(tok.shape[0], -1)
            clips.append(tok[:1200, :S])
    return clips


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    mp, _ = default_resource_paths()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    clips = load_train_clips(8)
    print("clips", len(clips), "device", device)
    model = load_model(str(mp), device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5)
    model.train()
    rng = np.random.default_rng(0)
    last = 0.0
    for step in range(250):
        ci = int(rng.integers(0, len(clips)))
        tok = clips[ci]
        t = int(rng.integers(T, tok.shape[0]))
        batch = tok[None, :, :]
        ctx = build_context_batch(batch, t)
        y = tok[t].astype(np.int64)
        x = torch.tensor(ctx, dtype=torch.long, device=device)
        yt = torch.tensor(y[None, :], dtype=torch.long, device=device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), yt.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last = float(loss.item())
        if step % 50 == 0:
            print(f"step {step} loss {last:.4f}")
    model.eval()
    torch.save(model.cpu().state_dict(), OUT_PT)
    export_runtime_weights(OUT_PT, OUT_PT.with_suffix(".rtnv.bin"))
    print("wrote", OUT_PT, "last_loss", last)

    if HELDOUT.exists():
        held = np.load(HELDOUT)[:48].astype(np.int64)
        def nll(m, device):
            m = m.to(device).eval()
            tot = 0.0
            n = 0
            with torch.no_grad():
                for t in range(T, held.shape[0]):
                    ctx = build_context_batch(held[None], t)
                    x = torch.tensor(ctx, dtype=torch.long, device=device)
                    y = torch.tensor(held[t][None], dtype=torch.long, device=device)
                    logits = m(x)
                    tot += float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1)).item()) * S
                    n += S
            return tot / n
        base = load_model(str(mp), device=device)
        ft = NextFramePredictor()
        ft.load_state_dict(torch.load(OUT_PT, map_location="cpu", weights_only=True))
        print("heldout ce bits/tok base", nll(base, device) / np.log(2), "ft", nll(ft, device) / np.log(2))


if __name__ == "__main__":
    main()
