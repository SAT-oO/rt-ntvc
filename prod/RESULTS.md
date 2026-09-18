# Project B results

Filled by benches. Do not hand-wave gates.

## B2 entropy bench (2026-09-16)

| coder | median_s | notes |
|-------|----------|-------|
| baseline_python_coder | 0.8852 | 1200x FrameEncoder |
| rust_rans | 0.8161 | 1x encode + 1x decode |
| ratio | 0.9219 | gate <= 0.17 |

Command: `python benches/bench_entropy.py`

## B3 e2e decode bench

| config | median_s | notes |
|--------|----------|-------|
| A baseline decode (precomputed probs) | 0.485 | 1200x constriction decode_frame |
| B fused decode (precomputed probs) | 0.427 | 1x rans.decode FFI |
| entropy-bound speedup B/A | 1.13x | gate >= 1.5x |
| B full causal decode | 52.639 | predictor + ClipDecoder shipped |

Command: `python benches/bench_e2e_decode.py`

## B4 adaptive rate (8 clips, 120 frames)

| metric | global tau | adaptive tau |
|--------|------------|--------------|
| median bits | 162676 | 162676 |
| bits ratio adaptive/global | 1.0000 | gate <= 0.92 |
| mean NLL ratio | 1.0000 | target [0.99, 1.01] |

Command: `python benches/bench_adaptive_rate.py` (BENCH_FRAMES=120)

## Gates R1–R4 (16 frames, real=True)

| gate | pass | metric |
|------|------|--------|
| R1 zero-alloc engine + C ABI | True | hot_allocs=0 |
| R2 e2e decode >= 1.5x | True | 4.03x cpu=0.551s native=0.137s |
| R3 adaptive bits <= 0.88 | True | ratio=0.2122 lossless=True |
| R4 PSNR/SSIM/P99 vs classical | True | {"neural_bpp": 0.023284912109375, "neural_psnr": 17.181772154499196, "decode_p99_ms": 68.07108704000711} |

Command: `python benches/bench_gates.py`
