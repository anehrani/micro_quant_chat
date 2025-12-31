from __future__ import annotations

import math
from typing import Tuple

import torch

from .ckpt import load_checkpoint
from .data import load_tokens


@torch.no_grad()
def _eval_loss(model, tokens: list[int], seq_len: int, device: torch.device) -> float:
    if len(tokens) < seq_len + 1:
        raise ValueError("Not enough tokens for evaluation")

    total_loss = 0.0
    n = 0
    for i in range(0, len(tokens) - seq_len - 1, seq_len):
        seq = tokens[i : i + seq_len + 1]
        x = torch.tensor([seq[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([seq[1:]], dtype=torch.long, device=device)
        loss = model(x, y)
        total_loss += float(loss.item())
        n += 1

    return total_loss / max(1, n)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="nanochat-style base_eval")
    p.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    p.add_argument("--data", "--data_file", dest="data", default="data/all_tokens.txt")
    p.add_argument("--device", default="auto")
    p.add_argument("--seq-len", "--seq_len", dest="seq_len", type=int, default=256)
    args = p.parse_args()

    model, cfg, dev, _ = load_checkpoint(args.checkpoint, device=args.device)
    tokens = load_tokens(args.data)

    loss = _eval_loss(model, tokens, args.seq_len, dev)
    ppl = math.exp(loss)

    total_params = sum(p.numel() for p in model.parameters())

    print(f"loss: {loss:.4f}")
    print(f"ppl:  {ppl:.2f}")
    print(f"params: {total_params:,}")
    print(f"config: {cfg}")


if __name__ == "__main__":
    main()
