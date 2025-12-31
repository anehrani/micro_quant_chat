from __future__ import annotations

import os
import time
from dataclasses import asdict

import torch
from torch.utils.data import DataLoader

from .ckpt import save_checkpoint
from .data import create_dataloaders, load_tokens
from .device import resolve_device
from .gpt import GPT, GPTConfig


def _train_epoch(model: GPT, train_loader: DataLoader, optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        loss = model(inputs, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += float(loss.item())
        num_batches += 1

    return total_loss / max(1, num_batches)


@torch.no_grad()
def _eval(model: GPT, val_loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for inputs, targets in val_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        loss = model(inputs, targets)
        total_loss += float(loss.item())
        num_batches += 1

    return total_loss / max(1, num_batches)


def train(
    *,
    config: GPTConfig,
    data_file: str,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    save_dir: str,
    device: str = "auto",
) -> GPT:
    dev = resolve_device(device)
    os.makedirs(save_dir, exist_ok=True)

    tokens = load_tokens(data_file)
    if not tokens:
        raise ValueError(f"No tokens found in {data_file}")

    train_loader, val_loader = create_dataloaders(tokens=tokens, seq_len=config.sequence_len, batch_size=batch_size)

    model = GPT(config)
    model.init_weights()
    model = model.to(dev)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(num_epochs):
        train_loss = _train_epoch(model, train_loader, optimizer, dev)
        val_loss = _eval(model, val_loader, dev)
        scheduler.step()

        print(f"Epoch {epoch + 1}/{num_epochs} | train {train_loss:.4f} | val {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                os.path.join(save_dir, "best_model.pt"),
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
            )
            print(f"  Saved best model (val={best_val_loss:.4f})")

    total = time.time() - start_time
    print(f"Training complete in {total:.2f}s | best val {best_val_loss:.4f}")
    return model


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="nanochat-style base_train")
    p.add_argument("--data", "--data_file", dest="data", default="data/all_tokens.txt")
    p.add_argument("--out", "--save_dir", dest="out", default="checkpoints")
    p.add_argument("--device", default="auto")

    # Backwards-compatible flags (src/train.py used these names)
    p.add_argument("--epochs", "--num_epochs", dest="epochs", type=int, default=10)
    p.add_argument("--batch", "--batch_size", dest="batch", type=int, default=32)
    p.add_argument("--lr", "--learning_rate", dest="lr", type=float, default=5e-4)

    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--vocab", type=int, default=512)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--kv-heads", type=int, default=4)
    p.add_argument("--embd", type=int, default=256)

    args = p.parse_args()

    cfg = GPTConfig(
        sequence_len=args.seq_len,
        vocab_size=args.vocab,
        n_layer=args.layers,
        n_head=args.heads,
        n_kv_head=args.kv_heads,
        n_embd=args.embd,
    )

    print(f"Config: {asdict(cfg)}")
    train(
        config=cfg,
        data_file=args.data,
        num_epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=args.lr,
        save_dir=args.out,
        device=args.device,
    )


if __name__ == "__main__":
    main()
