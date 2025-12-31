"""
Training script for GPT model on tokenized price series data.
"""

import os
import time
import math
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.gpt_model import GPT, GPTConfig


class TokenizedDataset(Dataset):
    """Load tokenized price series data"""

    def __init__(
        self,
        tokens_file: str | None = None,
        seq_len: int = 512,
        stride: int | None = None,
        tokens: list[int] | None = None,
    ):
        """
        Args:
            tokens_file: Path to file with space-separated token IDs
            seq_len: Sequence length for training
            stride: Step size between sequence starts (defaults to seq_len//2)
            tokens: Optional in-memory tokens (overrides tokens_file)
        """
        self.seq_len = seq_len

        if stride is None:
            stride = max(1, seq_len // 2)
        self.stride = stride

        # Load tokens
        if tokens is not None:
            self.tokens = tokens
            src = "<in-memory>"
        else:
            if tokens_file is None:
                raise ValueError("Either tokens_file or tokens must be provided")
            with open(tokens_file, "r") as f:
                token_str = f.read().strip()
                self.tokens = list(map(int, token_str.split()))
            src = tokens_file

        print(f"Loaded {len(self.tokens)} tokens from {src}")

        # Create sequences
        self.data = []
        for i in range(0, len(self.tokens) - seq_len - 1, self.stride):
            self.data.append(i)

        print(f"Created {len(self.data)} sequences of length {seq_len}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = self.data[idx]
        end = start + self.seq_len + 1
        seq = torch.tensor(self.tokens[start:end], dtype=torch.long)
        inputs = seq[:-1]
        targets = seq[1:]
        return inputs, targets


def create_dataloaders(
    data_file: str, seq_len: int = 512, batch_size: int = 32, num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """Create train/val dataloaders"""
    # IMPORTANT: Use a contiguous time split to avoid leakage.
    # With overlapping windows, random_split can put almost-identical sequences in train & val.
    with open(data_file, "r") as f:
        tokens = list(map(int, f.read().strip().split()))

    split_idx = int(0.8 * len(tokens))
    train_tokens = tokens[:split_idx]
    val_tokens = tokens[split_idx:]

    # Train uses overlap (more samples); Val uses non-overlapping windows for cleaner estimates.
    train_dataset = TokenizedDataset(tokens_file=None, tokens=train_tokens, seq_len=seq_len, stride=max(1, seq_len // 2))
    val_dataset = TokenizedDataset(tokens_file=None, tokens=val_tokens, seq_len=seq_len, stride=seq_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader


def train_epoch(model: GPT, train_loader: DataLoader, optimizer, device: torch.device) -> float:
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        loss = model(inputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if (batch_idx + 1) % max(1, len(train_loader) // 5) == 0:
            print(f"  Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")

    return total_loss / num_batches


@torch.no_grad()
def evaluate(model: GPT, val_loader: DataLoader, device: torch.device) -> float:
    """Evaluate on validation set"""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for inputs, targets in val_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        loss = model(inputs, targets)
        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def train(
    config: GPTConfig,
    data_file: str,
    num_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    save_dir: str = "checkpoints",
    device: str = "auto",
):
    """
    Train GPT model on tokenized price series
    
    Args:
        config: GPTConfig object
        data_file: Path to tokenized data file
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        save_dir: Directory to save checkpoints
        device: Device to use ('cuda', 'cpu', or 'auto')
    """
    # Auto-detect device
    device_str = device
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device_str}")
    dev = torch.device(device_str)

    # Create checkpoint directory
    os.makedirs(save_dir, exist_ok=True)

    # Create dataloaders
    print("\nLoading data...")
    train_loader, val_loader = create_dataloaders(data_file, config.sequence_len, batch_size)

    # Create model
    print(f"\nCreating model with config:")
    print(f"  Layers: {config.n_layer}")
    print(f"  Embedding dim: {config.n_embd}")
    print(f"  Heads: {config.n_head}")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Sequence length: {config.sequence_len}")

    model = GPT(config)
    model.init_weights()
    model = model.to(dev)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

    print("\nStarting training...\n")
    start_time = time.time()

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, dev)

        # Evaluate
        val_loss = evaluate(model, val_loader, dev)

        # Learning rate step
        scheduler.step()

        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"  Train loss: {train_loss:.4f}")
        print(f"  Val loss: {val_loss:.4f}")
        print(f"  Time: {epoch_time:.2f}s")

        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
            checkpoint_path = os.path.join(save_dir, "best_model.pt")
            torch.save(checkpoint, checkpoint_path)
            print(f"  Saved best model to {checkpoint_path}")

    total_time = time.time() - start_time
    print(f"\n\nTraining complete!")
    print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"Best validation loss: {best_val_loss:.4f}")

    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train GPT model")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate")
    args = parser.parse_args()
    
    # Configuration
    config = GPTConfig(
        sequence_len=256,
        vocab_size=128,
        n_layer=6,
        n_head=8,
        n_kv_head=8,
        n_embd=512,
    )

    # Train
    model = train(
        config=config,
        data_file="data/all_tokens.txt",
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_dir="checkpoints",
        device="auto",
    )
