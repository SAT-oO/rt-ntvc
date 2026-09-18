# Minimum compute (token codec)

Parameters only. Token predictor is 6 × TransformerEncoder, L=1024, d=256, FFN=768, vocab=1024, 128 tokens/frame, 20 FPS.

## Work

- **~14.56 GFLOP** per predicted frame (mul+add)
- **~291.3 GFLOP/s** sustained for 20 FPS token decode

## Storage

- Weights **~17.93 MB** f32 (as trained). **~8.97 MB** if inference stays f16
- `global_freq.npy` **8 KB**
- Pixel VQ encoder/decoder weights are extra (~119 MB encode, ~171 MB decode) and are not in the token crate

## RAM

- This repo one-shot: **≥ ~629 MB** probability tensor plus interpreter
- Causal Python path: **~150 MB** RSS at 120 frames (measured)
- Native session (pre-sized): attention scores **~17 MB** plus weights if resident (~18 MB f32)
- Tiled rewrite (not shipped): **~2–4 MB** working set plus one logits frame (512 KB)

## Numeric / platform (token session)

- f32 GEMM/attention; rate grid f64
- Hosted process with a heap for init; hot decode reuses session buffers
- Filesystem (or equivalent) to load weights
- No camera or display required for token I/O

## Pixel pipeline

Add VQ encode/decode weight bytes and **measured** VQ FLOPs. The 291.3 GFLOP/s figure is the predictor only.
