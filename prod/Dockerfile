FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl pkg-config ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app/prod
COPY prod/ /app/prod/

RUN pip install --no-cache-dir maturin \
    && pip install --no-cache-dir -e ".[dev,eval]" \
    && maturin develop --release

CMD ["sh", "-c", "pytest tests/ -q && python benches/bench_entropy.py"]
