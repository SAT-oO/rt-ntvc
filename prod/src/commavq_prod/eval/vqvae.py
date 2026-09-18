"""commaVQ encoder/decoder (no einops). Weights downloaded at runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEC_URL = "https://huggingface.co/commaai/commavq-gpt2m/resolve/main/decoder_pytorch_model.bin"
ENC_URL = "https://huggingface.co/commaai/commavq-gpt2m/resolve/main/encoder_pytorch_model.bin"


@dataclass
class CompressorConfig:
    in_channels: int = 3
    out_channels: int = 3
    ch_mult: tuple[int, ...] = (1, 1, 2, 2, 4)
    attn_resolutions: tuple[int, ...] = (16,)
    resolution: int = 256
    num_res_blocks: int = 2
    z_channels: int = 256
    vocab_size: int = 1024
    ch: int = 128
    dropout: float = 0.0

    @property
    def num_resolutions(self) -> int:
        return len(self.ch_mult)

    @property
    def quantized_resolution(self) -> int:
        return self.resolution // 2 ** (self.num_resolutions - 1)


def _swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def _norm(ch: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=32, num_channels=ch, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, 2, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (0, 1, 0, 1))
        return self.conv(x)


class ResnetBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float) -> None:
        super().__init__()
        self.in_channels = in_ch
        self.out_channels = out_ch
        self.norm1 = _norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.norm2 = _norm(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.nin_shortcut: Optional[nn.Conv2d]
        if in_ch != out_ch:
            self.nin_shortcut = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.nin_shortcut = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(_swish(self.norm1(x)))
        h = self.conv2(self.dropout(_swish(self.norm2(h))))
        if self.nin_shortcut is not None:
            x = self.nin_shortcut(x)
        return x + h


class AttnBlock(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.norm = _norm(ch)
        self.q = nn.Conv2d(ch, ch, 1)
        self.k = nn.Conv2d(ch, ch, 1)
        self.v = nn.Conv2d(ch, ch, 1)
        self.proj_out = nn.Conv2d(ch, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_ = self.norm(x)
        q, k, v = self.q(h_), self.k(h_), self.v(h_)
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w).permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        w_ = torch.bmm(q, k) * (c ** -0.5)
        w_ = torch.softmax(w_, dim=2).permute(0, 2, 1)
        v = v.reshape(b, c, h * w)
        h_ = torch.bmm(v, w_).reshape(b, c, h, w)
        return x + self.proj_out(h_)


class VectorQuantizer(nn.Module):
    def __init__(self, n_embed: int, dim: int) -> None:
        super().__init__()
        self._num_embeddings = n_embed
        self._embedding_dim = dim
        self._embedding = nn.Embedding(n_embed, dim)

    def decode(self, idx: torch.Tensor) -> torch.Tensor:
        b, s = idx.shape
        flat = idx.reshape(-1)
        q = self._embedding(flat)
        return q.reshape(b, s, self._embedding_dim)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, s, c = inputs.shape
        flat = inputs.reshape(b * s, c)
        dist = (
            (flat ** 2).sum(1, keepdim=True)
            + (self._embedding.weight ** 2).sum(1)
            - 2 * flat @ self._embedding.weight.t()
        )
        idx = dist.argmin(1)
        q = self._embedding(idx).reshape(b, s, c)
        return q, idx.reshape(b, s)


class Decoder(nn.Module):
    def __init__(self, config: CompressorConfig) -> None:
        super().__init__()
        self.config = config
        block_in = config.ch * config.ch_mult[config.num_resolutions - 1]
        curr_res = config.quantized_resolution
        self.post_quant_conv = nn.Conv2d(config.z_channels, config.z_channels, 1)
        self.quantize = VectorQuantizer(config.vocab_size, config.z_channels)
        self.conv_in = nn.Conv2d(config.z_channels, block_in, 3, 1, 1)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in, block_in, config.dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(block_in, block_in, config.dropout)
        self.up = nn.ModuleList()
        for i_level in reversed(range(config.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = config.ch * config.ch_mult[i_level]
            for _ in range(config.num_res_blocks + 1):
                block.append(ResnetBlock(block_in, block_out, config.dropout))
                block_in = block_out
            if curr_res in config.attn_resolutions:
                attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in)
                curr_res *= 2
            self.up.insert(0, up)
        self.norm_out = _norm(block_in)
        self.conv_out = nn.Conv2d(block_in, config.out_channels, 3, 1, 1)

    def forward(self, encoding_indices: torch.Tensor) -> torch.Tensor:
        z = self.quantize.decode(encoding_indices)
        w = self.config.quantized_resolution
        b, hw, c = z.shape
        h = hw // w
        z = z.reshape(b, h, w, c).permute(0, 3, 1, 2)
        z = self.post_quant_conv(z)
        x = self.conv_in(z)
        x = self.mid.block_2(self.mid.attn_1(self.mid.block_1(x)))
        for i_level in reversed(range(self.config.num_resolutions)):
            for i_block in range(self.config.num_res_blocks + 1):
                x = self.up[i_level].block[i_block](x)
                if len(self.up[i_level].attn) > 0:
                    x = self.up[i_level].attn[min(i_block, len(self.up[i_level].attn) - 1)](x)
            if i_level != 0:
                x = self.up[i_level].upsample(x)
        x = self.conv_out(_swish(self.norm_out(x)))
        return ((x + 1.0) / 2.0) * 255.0

    def load_from_url(self, url: str = DEC_URL) -> None:
        sd = torch.hub.load_state_dict_from_url(url, map_location="cpu", weights_only=True)
        self.load_state_dict(sd, strict=False)


class Encoder(nn.Module):
    def __init__(self, config: CompressorConfig) -> None:
        super().__init__()
        self.config = config
        self.conv_in = nn.Conv2d(config.in_channels, config.ch, 3, 1, 1)
        curr_res = config.resolution
        in_ch_mult = (1,) + tuple(config.ch_mult)
        self.down = nn.ModuleList()
        block_in = config.ch
        for i_level in range(config.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = config.ch * in_ch_mult[i_level]
            block_out = config.ch * config.ch_mult[i_level]
            for _ in range(config.num_res_blocks):
                block.append(ResnetBlock(block_in, block_out, config.dropout))
                block_in = block_out
                if curr_res in config.attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != config.num_resolutions - 1:
                down.downsample = Downsample(block_in)
                curr_res //= 2
            self.down.append(down)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in, block_in, config.dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(block_in, block_in, config.dropout)
        self.norm_out = _norm(block_in)
        self.conv_out = nn.Conv2d(block_in, config.z_channels, 3, 1, 1)
        self.quant_conv = nn.Conv2d(config.z_channels, config.z_channels, 1)
        self.quantize = VectorQuantizer(config.vocab_size, config.z_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hs = [self.conv_in(x)]
        for i_level in range(self.config.num_resolutions):
            for i_block in range(self.config.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.config.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(hs[-1])))
        h = self.conv_out(_swish(self.norm_out(h)))
        h = self.quant_conv(h)
        b, c, hh, ww = h.shape
        h = h.permute(0, 2, 3, 1).reshape(b, hh * ww, c)
        _, idx = self.quantize(h)
        return idx

    def load_from_url(self, url: str = ENC_URL) -> None:
        sd = torch.hub.load_state_dict_from_url(url, map_location="cpu", weights_only=True)
        self.load_state_dict(sd, strict=False)


_dec: Optional[Decoder] = None
_enc: Optional[Encoder] = None


def tokens_to_rgb(tokens: np.ndarray, device: str = "cpu") -> np.ndarray:
    """tokens (N, 128) int -> uint8 (N, 128, 256, 3)."""
    global _dec
    if _dec is None:
        cfg = CompressorConfig()
        _dec = Decoder(cfg)
        _dec.load_from_url()
        _dec.eval()
    _dec.to(device)
    out = []
    x = torch.from_numpy(tokens.astype(np.int64))
    with torch.no_grad():
        for i in range(x.shape[0]):
            y = _dec(x[i : i + 1].to(device)).cpu()
            out.append(y)
    rgb = torch.cat(out, 0).permute(0, 2, 3, 1).clamp(0, 255).byte().numpy()
    return rgb


def rgb_to_tokens(frames: np.ndarray, device: str = "cpu") -> np.ndarray:
    """uint8 (N, H, W, 3) -> (N, 128) int64. Resizes to 128x256 if needed."""
    global _enc
    if _enc is None:
        cfg = CompressorConfig()
        _enc = Encoder(cfg)
        _enc.load_from_url()
        _enc.eval()
    _enc.to(device)
    t = torch.from_numpy(frames.astype(np.float32))
    if t.ndim == 4 and t.shape[-1] == 3:
        t = t.permute(0, 3, 1, 2)
    if t.shape[-2] != 128 or t.shape[-1] != 256:
        t = F.interpolate(t, size=(128, 256), mode="bilinear", align_corners=False)
    # encoder is trained on 0..255 RGB floats (not [-1,1])
    # encoder expects 256-square-ish via config.resolution=256; feed 128x256 as-is
    idx = []
    with torch.no_grad():
        for i in range(t.shape[0]):
            idx.append(_enc(t[i : i + 1].to(device)).cpu())
    return torch.cat(idx, 0).numpy().astype(np.int64)
