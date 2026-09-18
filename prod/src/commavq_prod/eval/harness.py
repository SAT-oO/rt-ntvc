"""CLI: token eval, pixel eval, classical compare."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from commavq_prod.codec import decode_clip, encode_clip, load_codec
from commavq_prod.constants import BIT_DEPTH, FRAMES, H, S, SYMBOLS_PER_CLIP, W
from commavq_prod.data import iter_clips_hf, iter_clips_from_tar, synthetic_batch
from commavq_prod.types import TokenArray
from commavq_prod.eval.metrics import (
    bits_per_token,
    latency_stats,
    peak_rss_mb,
    psnr_ssim,
)
from commavq_prod.train import configure_determinism


def eval_tokens(
    splits: list[str] | None,
    max_clips: int,
    device: str,
) -> None:  # noqa: complex
    configure_determinism()
    model, global_probs = load_codec(device=device)
    encode_ms: list[float] = []
    decode_ms: list[float] = []
    bits_list: list[float] = []

    clips = _iter_clips(splits, max_clips)
    for tokens, _name in clips:
        tokens = tokens[:FRAMES, :S].astype(np.int16)

        t0 = time.perf_counter()
        blob = encode_clip(tokens, model, global_probs, device=device)
        encode_ms.append((time.perf_counter() - t0) * 1000)

        bits_list.append(len(blob) * 8)

        t0 = time.perf_counter()
        got = decode_clip(blob, tokens.shape[0], model, global_probs, device=device)
        decode_ms.append((time.perf_counter() - t0) * 1000)

        assert np.array_equal(got.astype(np.int16), tokens), "lossless token identity failed"

    n = len(bits_list)
    if n == 0:
        print("No clips evaluated.")
        return

    med_bits = float(np.median(bits_list))
    print(f"clips={n} bpp={med_bits / (SYMBOLS_PER_CLIP * BIT_DEPTH):.6f}")
    print(f"bits/token={bits_per_token(med_bits):.4f}")
    print(f"encode p50/p99 ms={latency_stats(encode_ms).p50_ms:.2f}/{latency_stats(encode_ms).p99_ms:.2f}")
    print(f"decode p50/p99 ms={latency_stats(decode_ms).p50_ms:.2f}/{latency_stats(decode_ms).p99_ms:.2f}")
    print(f"peak_rss_mb={peak_rss_mb():.1f}")


def _iter_clips(
    splits: list[str] | None, max_clips: int
) -> object:  # Iterator[Tuple[TokenArray, str]] — synthetic fallback mixed
    if splits:
        for sp in splits:
            p = Path(sp)
            if p.exists():
                yield from iter_clips_from_tar(p, max_clips=max_clips)
                return
    try:
        yield from iter_clips_hf(max_clips=max_clips)
    except Exception as e:
        print(f"HF unavailable ({e}); using synthetic clip.", file=sys.stderr)
        for i in range(max_clips):
            yield synthetic_batch(1, FRAMES, seed=i)[0], f"synthetic_{i}"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="commavq_prod.eval.harness")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_tok = sub.add_parser("tokens")
    p_tok.add_argument("--splits", nargs="*", default=None)
    p_tok.add_argument("--max-clips", type=int, default=8)
    p_tok.add_argument("--device", default="cpu")

    p_pix = sub.add_parser("pixels")
    p_pix.add_argument("--video", type=Path, default=None)
    p_pix.add_argument("--bpp", type=float, default=0.013)
    p_pix.add_argument("--max-frames", type=int, default=FRAMES)

    args = parser.parse_args(argv)
    if args.cmd == "tokens":
        eval_tokens(args.splits, args.max_clips, args.device)
    elif args.cmd == "pixels":
        from commavq_prod.eval.classical import encode_video, decode_video
        from commavq_prod.eval.metrics import psnr_ssim
        from commavq_prod.eval.vqvae import rgb_to_tokens, tokens_to_rgb
        from commavq_prod.codec import encode_clip_native, decode_clip_native, load_engine
        import tempfile

        load_engine()
        n = args.max_frames
        if args.video and args.video.exists():
            import subprocess
            with tempfile.TemporaryDirectory() as td:
                raw = Path(td) / "in.raw"
                subprocess.run(
                    ["ffmpeg", "-y", "-i", str(args.video), "-vf", f"scale={W}:{H}",
                     "-frames:v", str(n), "-f", "rawvideo", "-pix_fmt", "rgb24", str(raw)],
                    check=True, capture_output=True,
                )
                rgb = np.fromfile(raw, dtype=np.uint8)
                nf = rgb.size // (H * W * 3)
                rgb = rgb[: nf * H * W * 3].reshape(nf, H, W, 3)
        else:
            print("no --video; run benches/bench_rd.py for structured RGB RD")
            return
        device = "cpu"
        try:
            import torch
            if torch.backends.mps.is_available():
                device = "mps"
        except Exception:
            pass
        tokens = rgb_to_tokens(rgb, device=device).astype(np.int32)
        blob = encode_clip_native(tokens, adaptive=True)
        rec = tokens_to_rgb(decode_clip_native(blob), device=device)
        p, s = psnr_ssim(rec[: rgb.shape[0]], rgb[: rec.shape[0]])
        bpp = len(blob) * 8 / (rgb.shape[0] * H * W)
        print(f"neural bpp={bpp:.4f} psnr={p:.2f} ssim={s:.4f} bytes={len(blob)}")


if __name__ == "__main__":
    main()
