# commavq-prod (Project B)

Production neural video-token runtime. Challenge code (`commavq_compressor/`) is frozen.

## Install

```bash
cd prod
pip install -e ".[dev,eval]"
maturin develop --release
```

Docker:

```bash
docker build -f prod/Dockerfile -t commavq-prod .
```

## Train smoke

```bash
python -c "from commavq_prod.train import train_smoke; print(train_smoke(2))"
```

## Eval

```bash
python -m commavq_prod.eval.harness tokens --max-clips 8
```

## Benches

```bash
python benches/bench_entropy.py
python benches/bench_e2e_decode.py
python benches/bench_adaptive_rate.py
```

Results: [RESULTS.md](RESULTS.md)

## Layout

- `src/commavq_prod/` — predictor, codec, rate control, eval
- `rust_rans/` — PyO3 rANS (one encode/decode FFI per clip)
- `baseline_python_coder.py` — frozen constriction baseline for B2/B3
