#!/usr/bin/env python3
"""Analytic + probed compute sheet. No board/ISA names."""

from __future__ import annotations

import json
from pathlib import Path

from commavq_prod.constants import D_MODEL, FFN_DIM, FPS, FRAMES, N_LAYERS, S, T, VOCAB

ROOT = Path(__file__).resolve().parents[1]
OUT_MD = ROOT / "COMPUTE_REQUIREMENTS.md"
OUT_JSON = ROOT / "compute_sheet.json"

L = T * S
D = D_MODEL
# mul+add
flops_qkv = 2 * L * (3 * D) * D
flops_attn = 2 * (2 * L * L * D)  # QK + AV
flops_o = 2 * L * D * D
flops_ffn = 2 * 2 * L * D * FFN_DIM
flops_layer = flops_qkv + flops_attn + flops_o + flops_ffn
flops_head = 2 * S * D * VOCAB
flops_frame = N_LAYERS * flops_layer + flops_head
gflop = flops_frame / 1e9
gflops_rt = gflop * FPS
weights_f32 = 4_483_584 * 4 / 1e6
probs_mb = FRAMES * S * VOCAB * 4 / 1e6
attn_mb = 4 * L * L * 4 / 1e6


def main() -> None:
    data = {
        "seq_len": L,
        "d_model": D,
        "layers": N_LAYERS,
        "gflop_per_predicted_frame": round(gflop, 2),
        "gflops_for_20fps": round(gflops_rt, 1),
        "weights_mb_f32": round(weights_f32, 2),
        "weights_mb_f16": round(weights_f32 / 2, 2),
        "one_shot_probs_mb": round(probs_mb, 1),
        "attention_act_mb": round(attn_mb, 1),
        "fps": FPS,
    }
    OUT_JSON.write_text(json.dumps(data, indent=2))
    OUT_MD.write_text(
        f"""# Minimum compute (token codec)

Parameters only. Token predictor is {N_LAYERS} × TransformerEncoder, L={L}, d={D}, FFN={FFN_DIM}, vocab={VOCAB}, {S} tokens/frame, {FPS} FPS.

## Work

- **~{data['gflop_per_predicted_frame']} GFLOP** per predicted frame (mul+add)
- **~{data['gflops_for_20fps']} GFLOP/s** sustained for {FPS} FPS token decode

## Storage

- Weights **~{data['weights_mb_f32']} MB** f32 (as trained). **~{data['weights_mb_f16']} MB** if inference stays f16
- `global_freq.npy` **8 KB**
- Pixel VQ encoder/decoder weights are extra (~119 MB encode, ~171 MB decode) and are not in the token crate

## RAM

- This repo one-shot: **≥ ~{data['one_shot_probs_mb']:.0f} MB** probability tensor plus interpreter
- Causal Python path: **~150 MB** RSS at 120 frames (measured)
- Native session (pre-sized): attention scores **~{data['attention_act_mb']:.0f} MB** plus weights if resident (~{data['weights_mb_f32']:.0f} MB f32)
- Tiled rewrite (not shipped): **~2–4 MB** working set plus one logits frame (512 KB)

## Numeric / platform (token session)

- f32 GEMM/attention; rate grid f64
- Hosted process with a heap for init; hot decode reuses session buffers
- Filesystem (or equivalent) to load weights
- No camera or display required for token I/O

## Pixel pipeline

Add VQ encode/decode weight bytes and **measured** VQ FLOPs. The {data['gflops_for_20fps']} GFLOP/s figure is the predictor only.
"""
    )
    print(OUT_MD)


if __name__ == "__main__":
    main()
