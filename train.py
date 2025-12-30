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

    def __init__(self, tokens_file: str, seq_len: int = 512):
        """
        Args:
            tokens_file: Path to file with space-separated token IDs
            seq_len: Sequence length for training
        """
        self.seq_len = seq_len

        # Load tokens
        with open(tokens_file, "r") as f:
            token_str = f.read().strip()
            self.tokens = list(map(int, token_str.split()))

        print(f"Loaded {len(self.tokens)} tokens from {tokens_file}")

        # Create sequences
        self.data = []
        for i in range(0, len(self.tokens) - seq_len, seq_len // 2):  # Stride by half for more samples
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
    dataset = TokenizedDataset(data_file, seq_len)

    # Split into train/val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
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
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    device = torch.device(device)

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
    model = model.to(device)

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
        train_loss = train_epoch(model, train_loader, optimizer, device)

        # Evaluate
        val_loss = evaluate(model, val_loader, device)

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
    # Configuration
    config = GPTConfig(
        sequence_len=256,
        vocab_size=128,
        n_layer=2,
        n_head=8,
        n_kv_head=8,
        n_embd=512,
    )

    # Train
    model = train(
        config=config,
        data_file="data/all_tokens.txt",
        num_epochs=10,
        batch_size=32,
        learning_rate=5e-4,
        save_dir="checkpoints",
        device="auto",
    )
