"""Smoke test for eval harness imports."""

from commavq_prod.eval.metrics import latency_stats, raw_token_bpp


def test_metrics_smoke() -> None:
    assert raw_token_bpp() == 10
    st = latency_stats([1.0, 2.0, 3.0, 100.0])
    assert st.p50_ms == 2.5
