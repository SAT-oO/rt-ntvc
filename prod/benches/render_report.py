#!/usr/bin/env python3
"""HTML report from bench_report.json + optional bench_rd.json."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GATES = ROOT / "bench_report.json"
RD = ROOT / "bench_rd.json"
HTML = ROOT / "bench_report.html"


def _bar(label: str, ok: bool, detail: str) -> str:
    c = "#0a0" if ok else "#c00"
    return f'<div style="margin:8px 0;font-family:sans-serif"><b style="color:{c}">{label}: {"PASS" if ok else "FAIL"}</b> — {detail}</div>'


def main() -> None:
    gates = json.loads(GATES.read_text()) if GATES.exists() else {"gates": {}}
    g = gates.get("gates", {})
    parts = ["<html><body><h1>RT-NTVC gates</h1>"]
    for k in ("r1", "r2", "r3", "r4"):
        row = g.get(k, {"pass": False, "reason": "missing"})
        parts.append(_bar(k.upper(), bool(row.get("pass")), json.dumps(row)[:400]))
    if RD.exists():
        rd = json.loads(RD.read_text())
        n = rd.get("neural", {})
        parts.append("<h2>RD (pixel)</h2>")
        parts.append(
            f"<p>neural bpp={n.get('bpp')} PSNR={n.get('psnr')} SSIM={n.get('ssim')} "
            f"decode p50/p99 ms={n.get('decode_p50_ms')}/{n.get('decode_p99_ms')}</p>"
        )
        parts.append("<pre>" + json.dumps(rd.get("classical"), indent=2)[:4000] + "</pre>")
    parts.append("</body></html>")
    HTML.write_text("".join(parts))
    print(HTML)


if __name__ == "__main__":
    main()
