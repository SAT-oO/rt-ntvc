"""HuggingFace commavq loader + synthetic batches for offline smoke."""

from __future__ import annotations

import tarfile
from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np

from commavq_prod.constants import FRAMES, S
from commavq_prod.types import TokenArray

HF_REPO = "commaai/commavq"
HF_DATA_GLOB = "data/data-*.tar.gz"


def synthetic_batch(
    batch_size: int = 2,
    num_frames: int = FRAMES,
    seed: int = 0,
) -> TokenArray:
    """(B, num_frames, S) int16 tokens for train smoke without HF."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 1024, size=(batch_size, num_frames, S), dtype=np.int16)


def _tokens_from_tar_member(tar: tarfile.TarFile, member_name: str) -> TokenArray:
    f = tar.extractfile(member_name)
    if f is None:
        raise ValueError(f"Cannot extract {member_name}")
    data = np.load(f)
    tokens = data["tokens"] if isinstance(data, np.lib.npyio.NpzFile) else data
    return np.asarray(tokens, dtype=np.int16)


def iter_clips_from_tar(
    tar_path: Path,
    max_clips: Optional[int] = None,
) -> Iterator[Tuple[TokenArray, str]]:
    """Yield (tokens (1200, 128), file_name) from a local shard tar.gz."""
    count = 0
    with tarfile.open(tar_path, "r:gz") as tar:
        members = sorted(m.name for m in tar.getmembers() if m.name.endswith(".npz"))
        for name in members:
            tokens = _tokens_from_tar_member(tar, name)
            yield tokens, Path(name).stem
            count += 1
            if max_clips is not None and count >= max_clips:
                return


def iter_clips_hf(
    max_clips: Optional[int] = None,
    split_glob: Optional[str] = None,
) -> Iterator[Tuple[TokenArray, str]]:
    """Stream clips from commaai/commavq on HuggingFace."""
    from datasets import load_dataset

    pattern = split_glob or HF_DATA_GLOB
    ds = load_dataset(HF_REPO, data_files=pattern, split="train", streaming=True)
    count = 0
    for row in ds:
        tokens = np.asarray(row["tokens"], dtype=np.int16)
        name = str(row.get("file_name", row.get("__key__", f"clip_{count}")))
        yield tokens, name
        count += 1
        if max_clips is not None and count >= max_clips:
            return
