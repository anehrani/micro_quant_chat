from __future__ import annotations

import os
import time
from dataclasses import asdict

import torch
from torch.utils.data import DataLoader

from .asset_tokens import asset_tokens_from_csv, load_tokenizer
from .ckpt import load_checkpoint, save_checkpoint
from .data import create_dataloaders
from .device import resolve_device


def _train_epoch(model, train_loader: DataLoader, optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += float(loss.item())
        n += 1
    return total_loss / max(1, n)


@torch.no_grad()
def _eval(model, val_loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    n = 0
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        loss = model(x, y)
        total_loss += float(loss.item())
        n += 1
    return total_loss / max(1, n)


def midtrain_asset(
    *,
    checkpoint_in: str,
    checkpoint_out: str,
    asset_csv: str,
    epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 1e-5,
    seq_len: int = 256,
    device: str = "auto",
) -> None:
    """Stage 2: Midtraining - specialized training on target asset.
    
    Like nanochat's midtraining with curated data, this fine-tunes the
    pretrained model on a specific asset's historical data using
    teacher forcing (next-token prediction).
    """
    dev = resolve_device(device)

    model, cfg, dev, ckpt = load_checkpoint(checkpoint_in, device=str(dev))
    model.train()

    tok = load_tokenizer(device=str(dev))
    asset = asset_tokens_from_csv(csv_path=asset_csv, tokenizer=tok, device=str(dev), normalize=True)

    # Reuse the existing GPT context length.
    seq_len = min(int(seq_len), int(cfg.sequence_len))
    
    # Adjust batch size for smaller datasets
    effective_batch_size = min(batch_size, max(1, len(asset.tokens) // (seq_len * 4)))

    train_loader, val_loader = create_dataloaders(tokens=asset.tokens, seq_len=seq_len, batch_size=effective_batch_size)
    
    if len(train_loader) == 0:
        raise ValueError(f"No training batches! tokens={len(asset.tokens)}, seq_len={seq_len}, batch={effective_batch_size}")

    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    print("Stage 2 (Midtrain) - Specialized training on target asset")
    print(f"  checkpoint_in:  {checkpoint_in}")
    print(f"  checkpoint_out: {checkpoint_out}")
    print(f"  asset_tokens:   {len(asset.tokens)}")
    print(f"  seq_len:        {seq_len}")
    print(f"  epochs:         {epochs}")
    print(f"  batch_size:     {effective_batch_size} (requested: {batch_size})")
    print(f"  train_batches:  {len(train_loader)}")
    print(f"  lr:             {learning_rate}")
    print(f"  config:         {cfg}")

    best = float("inf")
    start = time.time()
    for e in range(epochs):
        tr = _train_epoch(model, train_loader, opt, dev)
        va = _eval(model, val_loader, dev)
        sched.step()
        print(f"epoch {e + 1}/{epochs} | train {tr:.4f} | val {va:.4f}")

        if va < best:
            best = va
            save_checkpoint(
                checkpoint_out,
                model=model,
                optimizer=opt,
                epoch=int(ckpt.get("epoch", 0)) + e,
                train_loss=tr,
                val_loss=va,
            )
            print(f"  saved best midtrain checkpoint (val={best:.4f})")

    total = time.time() - start
    print(f"Midtrain complete in {total:.2f}s | best val {best:.4f}")


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Stage 2: Midtraining on target asset")
    p.add_argument("--in", dest="checkpoint_in", default="checkpoints/best_model.pt")
    p.add_argument("--out", dest="checkpoint_out", default="checkpoints/midtrain_btc.pt")
    p.add_argument(
        "--csv",
        "--asset-csv",
        dest="csv",
        default="data/BTC-USD_2014-09-17_2025-12-31_1d.csv",
    )

    p.add_argument("--device", default="auto")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--seq-len", type=int, default=256)

    args = p.parse_args()

    midtrain_asset(
        checkpoint_in=args.checkpoint_in,
        checkpoint_out=args.checkpoint_out,
        asset_csv=args.csv,
        epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=args.lr,
        seq_len=args.seq_len,
        device=args.device,
    )


if __name__ == "__main__":
    main()
