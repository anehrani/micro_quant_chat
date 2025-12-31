"""
Stage 3: Supervised Fine-Tuning for Multi-Day Prediction

Like multi-turn conversation in chat models, this stage trains the model
to predict multiple future days given a context window.

Training format:
  Context: [day_1, day_2, ..., day_N]  (prompt)
  Target:  [day_N+1, day_N+2, ..., day_N+H]  (multi-day forecast)

The model learns to generate coherent multi-day trajectories, not just
single next-token predictions.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .asset_tokens import asset_tokens_from_csv, load_tokenizer
from .ckpt import load_checkpoint, save_checkpoint
from .device import resolve_device


@dataclass
class MultiDayPredictionSample:
    """A training sample for multi-day prediction."""
    context: List[int]  # Input tokens (prompt)
    target: List[int]   # Output tokens to predict (multi-day forecast)


class MultiDayDataset(Dataset):
    """
    Dataset for multi-day prediction training.
    
    Each sample consists of:
    - context: N tokens representing past days
    - target: H tokens representing future days to predict
    
    This is like having a "conversation" where:
    - User asks: "Given these past N days..."
    - Model responds: "The next H days will be..."
    """
    
    def __init__(
        self,
        tokens: List[int],
        context_len: int = 128,
        horizon: int = 32,
        stride: int = 16,
    ):
        """
        Args:
            tokens: Full token sequence
            context_len: Number of past tokens as context
            horizon: Number of future tokens to predict
            stride: Step between samples
        """
        self.tokens = tokens
        self.context_len = context_len
        self.horizon = horizon
        
        # Generate sample start positions
        self.starts = []
        max_start = len(tokens) - context_len - horizon
        for i in range(0, max(1, max_start), stride):
            self.starts.append(i)
    
    def __len__(self) -> int:
        return len(self.starts)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = self.starts[idx]
        
        # Context (past) and target (future)
        context = self.tokens[start : start + self.context_len]
        target = self.tokens[start + self.context_len : start + self.context_len + self.horizon]
        
        # For autoregressive training, we need the full sequence
        # Input: [context, target[:-1]]
        # Target: [context[1:], target]
        full_seq = context + target
        
        x = torch.tensor(full_seq[:-1], dtype=torch.long)
        y = torch.tensor(full_seq[1:], dtype=torch.long)
        
        return x, y


def _compute_multi_day_loss(
    model,
    context: torch.Tensor,
    target: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, float]:
    """
    Compute loss focusing on the multi-day prediction portion.
    
    Args:
        model: GPT model
        context: (B, context_len) - past tokens
        target: (B, horizon) - future tokens to predict
        device: compute device
        
    Returns:
        loss: scalar loss tensor
        accuracy: token accuracy on target portion
    """
    B, context_len = context.shape
    horizon = target.shape[1]
    
    # Concatenate for full sequence
    full_input = torch.cat([context, target[:, :-1]], dim=1)  # (B, context_len + horizon - 1)
    full_target = torch.cat([context[:, 1:], target], dim=1)  # (B, context_len + horizon - 1)
    
    # Forward pass
    logits = model(full_input)  # (B, T, vocab_size)
    
    # Compute loss only on the prediction portion (last `horizon` tokens)
    pred_logits = logits[:, context_len - 1:, :]  # (B, horizon, vocab_size)
    pred_targets = full_target[:, context_len - 1:]  # (B, horizon)
    
    loss = F.cross_entropy(
        pred_logits.reshape(-1, pred_logits.size(-1)),
        pred_targets.reshape(-1),
    )
    
    # Compute accuracy
    with torch.no_grad():
        pred_tokens = pred_logits.argmax(dim=-1)  # (B, horizon)
        correct = (pred_tokens == pred_targets).float().mean().item()
    
    return loss, correct


def _train_sft_epoch(
    model,
    train_loader: DataLoader,
    optimizer,
    device: torch.device,
    context_len: int,
) -> Tuple[float, float]:
    """Train one epoch of multi-day SFT."""
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        
        # Standard next-token prediction loss
        loss = model(x, y)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += float(loss.item())
        n += 1
    
    return total_loss / max(1, n), total_acc / max(1, n)


@torch.no_grad()
def _eval_sft(
    model,
    val_loader: DataLoader,
    device: torch.device,
    context_len: int,
    horizon: int,
) -> Tuple[float, float]:
    """Evaluate multi-day prediction performance."""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        
        # Standard loss
        loss = model(x, y)
        total_loss += float(loss.item())
        
        # Also compute accuracy on the prediction horizon
        # Get context (first context_len tokens) and target (rest)
        context = x[:, :context_len]
        
        # Generate predictions autoregressively
        generated = context.clone()
        for _ in range(min(horizon, x.size(1) - context_len)):
            logits = model(generated)[:, -1, :]
            next_token = logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
        
        # Compare with ground truth
        if generated.size(1) > context_len:
            pred_tokens = generated[:, context_len:]
            true_tokens = y[:, context_len - 1 : context_len - 1 + pred_tokens.size(1)]
            if true_tokens.size(1) == pred_tokens.size(1):
                acc = (pred_tokens == true_tokens).float().mean().item()
                total_acc += acc
        
        n += 1
    
    return total_loss / max(1, n), total_acc / max(1, n)


def sft_multiday(
    *,
    checkpoint_in: str,
    checkpoint_out: str,
    asset_csv: str,
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 5e-6,
    context_len: int = 128,
    horizon: int = 32,
    device: str = "auto",
) -> None:
    """
    Stage 3: Supervised Fine-Tuning for Multi-Day Prediction.
    
    Trains the model to predict multiple future days given a context,
    similar to how chat models learn multi-turn conversations.
    
    Args:
        checkpoint_in: Input checkpoint (from midtrain)
        checkpoint_out: Output checkpoint path
        asset_csv: Path to asset CSV file
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate (lower than midtrain)
        context_len: Number of past days as context
        horizon: Number of future days to predict
        device: Compute device
    """
    dev = resolve_device(device)
    
    model, cfg, dev, ckpt = load_checkpoint(checkpoint_in, device=str(dev))
    model.train()
    
    tok = load_tokenizer(device=str(dev))
    asset = asset_tokens_from_csv(csv_path=asset_csv, tokenizer=tok, device=str(dev), normalize=True)
    tokens = asset.tokens
    
    if len(tokens) < context_len + horizon + 10:
        raise ValueError(f"Not enough tokens: {len(tokens)} < {context_len + horizon + 10}")
    
    # Time-based split (80% train, 20% val)
    split_idx = int(0.8 * len(tokens))
    train_tokens = tokens[:split_idx]
    val_tokens = tokens[split_idx:]
    
    train_ds = MultiDayDataset(train_tokens, context_len, horizon, stride=horizon // 2)
    val_ds = MultiDayDataset(val_tokens, context_len, horizon, stride=horizon)
    
    effective_batch = min(batch_size, max(1, len(train_ds) // 4))
    
    train_loader = DataLoader(train_ds, batch_size=effective_batch, shuffle=True, drop_last=len(train_ds) > effective_batch)
    val_loader = DataLoader(val_ds, batch_size=effective_batch, shuffle=False)
    
    if len(train_loader) == 0:
        raise ValueError(f"No training batches! samples={len(train_ds)}, batch={effective_batch}")
    
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    
    print("Stage 3 (SFT) - Multi-Day Prediction Training")
    print(f"  checkpoint_in:  {checkpoint_in}")
    print(f"  checkpoint_out: {checkpoint_out}")
    print(f"  asset_tokens:   {len(tokens)}")
    print(f"  context_len:    {context_len} days")
    print(f"  horizon:        {horizon} days to predict")
    print(f"  epochs:         {epochs}")
    print(f"  batch_size:     {effective_batch} (requested: {batch_size})")
    print(f"  train_samples:  {len(train_ds)}")
    print(f"  val_samples:    {len(val_ds)}")
    print(f"  lr:             {learning_rate}")
    print()
    
    best_loss = float("inf")
    best_acc = 0.0
    start = time.time()
    
    for e in range(epochs):
        train_loss, train_acc = _train_sft_epoch(model, train_loader, opt, dev, context_len)
        val_loss, val_acc = _eval_sft(model, val_loader, dev, context_len, horizon)
        sched.step()
        
        print(f"epoch {e + 1:2d}/{epochs} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | val_acc {val_acc:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_acc = val_acc
            save_checkpoint(
                checkpoint_out,
                model=model,
                optimizer=opt,
                epoch=int(ckpt.get("epoch", 0)) + e,
                train_loss=train_loss,
                val_loss=val_loss,
            )
            print(f"  saved best SFT checkpoint (val_loss={best_loss:.4f}, val_acc={best_acc:.4f})")
    
    total = time.time() - start
    print(f"\nSFT complete in {total:.2f}s | best val_loss {best_loss:.4f} | best val_acc {best_acc:.4f}")


def main() -> None:
    import argparse
    
    p = argparse.ArgumentParser(description="Stage 3: SFT for Multi-Day Prediction")
    p.add_argument("--in", dest="checkpoint_in", default="checkpoints/midtrain_btc.pt")
    p.add_argument("--out", dest="checkpoint_out", default="checkpoints/sft_btc.pt")
    p.add_argument(
        "--csv",
        "--asset-csv",
        dest="csv",
        default="data/BTC-USD_2014-09-17_2025-12-31_1d.csv",
    )
    
    p.add_argument("--device", default="auto")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--context-len", type=int, default=128)
    p.add_argument("--horizon", type=int, default=32)
    
    args = p.parse_args()
    
    sft_multiday(
        checkpoint_in=args.checkpoint_in,
        checkpoint_out=args.checkpoint_out,
        asset_csv=args.csv,
        epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=args.lr,
        context_len=args.context_len,
        horizon=args.horizon,
        device=args.device,
    )


if __name__ == "__main__":
    main()
