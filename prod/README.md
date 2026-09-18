# commavq-prod (Project B)

Production neural video-token runtime. Challenge code (`commavq_compressor/`) is frozen.

## Install

```bash
cd prod
pip install -e ".[dev,eval]"
maturin develop --release
```

## Train smoke

```bash
python -c "from commavq_prod.train import train_smoke; print(train_smoke(2))"
```

## Eval

```bash
python -m commavq_prod.eval.harness tokens --max-clips 8
python benches/bench_rd.py
```

## Benches / gates

```bash
python benches/bench_gates.py
python benches/bench_rd.py
python benches/bench_latency.py
python benches/compute_sheet.py
python benches/render_report.py
```

Results: [RESULTS.md](RESULTS.md), [bench_report.html](bench_report.html), [COMPUTE_REQUIREMENTS.md](COMPUTE_REQUIREMENTS.md)

R1–R3: native session (C ABI + `rtnv.hpp`), one decode FFI, causal e2e vs Python CPU, adaptive neural vs `global_freq` unigram. R4: PSNR/SSIM/P99 vs libx264 / libx265 / libsvtav1 on original RGB.

## Minimum compute

See [COMPUTE_REQUIREMENTS.md](COMPUTE_REQUIREMENTS.md). Token predictor only: **~14.6 GFLOP/frame**, **~290 GFLOP/s** at 20 FPS, **~18 MB** f32 weights.

## Layout

- `src/commavq_prod/` — predictor, codec, rate control, eval
- `rust_rans/` — native session (transformer + rANS), PyO3 + C ABI
- `baseline_python_coder.py` — frozen constriction baseline for B2/B3
