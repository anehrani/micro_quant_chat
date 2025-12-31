from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


class TokenizedDataset(Dataset):
    """Contiguous token stream -> sliding windows for next-token prediction."""

    def __init__(
        self,
        *,
        tokens: List[int],
        seq_len: int,
        stride: int | None = None,
    ):
        self.seq_len = int(seq_len)
        if stride is None:
            stride = max(1, self.seq_len // 2)
        self.stride = int(stride)

        self.tokens = tokens
        self.starts: List[int] = []
        for i in range(0, len(self.tokens) - self.seq_len - 1, self.stride):
            self.starts.append(i)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = self.starts[idx]
        end = start + self.seq_len + 1
        seq = torch.tensor(self.tokens[start:end], dtype=torch.long)
        return seq[:-1], seq[1:]


@dataclass(frozen=True)
class DataSplit:
    train_tokens: List[int]
    val_tokens: List[int]


def load_tokens(path: str) -> List[int]:
    with open(path, "r") as f:
        raw = f.read().strip()
    if not raw:
        return []
    return list(map(int, raw.split()))


def time_split_tokens(tokens: List[int], train_ratio: float = 0.8) -> DataSplit:
    split_idx = int(train_ratio * len(tokens))
    return DataSplit(train_tokens=tokens[:split_idx], val_tokens=tokens[split_idx:])


def create_dataloaders(
    *,
    tokens: List[int],
    seq_len: int,
    batch_size: int,
    num_workers: int = 0,
    train_ratio: float = 0.8,
) -> tuple[DataLoader, DataLoader]:
    """Create time-split train/val loaders to avoid leakage with overlapping windows."""
    split = time_split_tokens(tokens, train_ratio=train_ratio)

    train_ds = TokenizedDataset(tokens=split.train_tokens, seq_len=seq_len, stride=max(1, seq_len // 2))
    val_ds = TokenizedDataset(tokens=split.val_tokens, seq_len=seq_len, stride=seq_len)
    
    # Only drop_last if we have enough samples
    drop_last = len(train_ds) > batch_size

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=drop_last,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader
