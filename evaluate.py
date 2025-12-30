"""
Evaluation script for GPT model.
Compute metrics and analyze model performance.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

from src.gpt_model import GPT, GPTConfig


def load_checkpoint(checkpoint_path: str, device: str = "auto"):
    """Load model from checkpoint"""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model = GPT(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, config, device


def compute_perplexity(model: GPT, inputs: torch.Tensor, targets: torch.Tensor, device: torch.device) -> float:
    """Compute perplexity (lower is better)"""
    with torch.no_grad():
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), reduction="mean"
        )
        perplexity = torch.exp(loss)
    return perplexity.item()


def compute_accuracy(model: GPT, inputs: torch.Tensor, targets: torch.Tensor, device: torch.device) -> float:
    """Compute next-token prediction accuracy"""
    with torch.no_grad():
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        predictions = torch.argmax(logits, dim=-1)
        accuracy = (predictions == targets).float().mean()
    return accuracy.item()


def compute_top_k_accuracy(
    model: GPT, inputs: torch.Tensor, targets: torch.Tensor, device: torch.device, k: int = 5
) -> float:
    """Compute top-k accuracy"""
    with torch.no_grad():
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        _, top_k_indices = torch.topk(logits, k, dim=-1)
        correct = torch.sum(top_k_indices == targets.unsqueeze(-1))
        accuracy = correct.float() / targets.numel()
    return accuracy.item()


def analyze_token_distribution(tokens: List[int]) -> Dict[str, float]:
    """Analyze distribution of tokens"""
    tokens = np.array(tokens)
    return {
        "mean": float(tokens.mean()),
        "std": float(tokens.std()),
        "min": int(tokens.min()),
        "max": int(tokens.max()),
        "median": float(np.median(tokens)),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate GPT model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to checkpoint")
    parser.add_argument("--data_file", type=str, default="data/all_tokens.txt", help="Path to token data")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")
    parser.add_argument("--seq_len", type=int, default=256, help="Sequence length for evaluation")

    args = parser.parse_args()

    # Load model
    print("Loading model...")
    model, config, device = load_checkpoint(args.checkpoint, args.device)
    print(f"Model config:\n{config}\n")

    # Load data
    print("Loading data...")
    with open(args.data_file, "r") as f:
        tokens = list(map(int, f.read().strip().split()))
    print(f"Loaded {len(tokens)} tokens\n")

    # Analyze token distribution
    print("Token distribution statistics:")
    dist_stats = analyze_token_distribution(tokens)
    for key, val in dist_stats.items():
        print(f"  {key}: {val}")
    print()

    # Create test sequences
    print("Preparing evaluation sequences...")
    test_sequences = []
    for i in range(0, len(tokens) - args.seq_len - 1, args.seq_len):
        seq = tokens[i : i + args.seq_len + 1]
        if len(seq) == args.seq_len + 1:
            test_sequences.append(seq)

    print(f"Created {len(test_sequences)} test sequences\n")

    # Compute metrics
    print("Computing metrics...")
    total_loss = 0.0
    total_perplexity = 0.0
    total_acc = 0.0
    total_top5_acc = 0.0
    num_batches = 0

    with torch.no_grad():
        for idx, seq in enumerate(test_sequences):
            inputs = torch.tensor([seq[:-1]], dtype=torch.long)
            targets = torch.tensor([seq[1:]], dtype=torch.long)

            # Compute loss
            inputs, targets_dev = inputs.to(device), targets.to(device)
            loss = model(inputs, targets_dev)
            total_loss += loss.item()

            # Compute other metrics
            perp = compute_perplexity(model, inputs, targets, device)
            acc = compute_accuracy(model, inputs, targets, device)
            top5_acc = compute_top_k_accuracy(model, inputs, targets, device, k=5)

            total_perplexity += perp
            total_acc += acc
            total_top5_acc += top5_acc
            num_batches += 1

            if (idx + 1) % max(1, len(test_sequences) // 5) == 0:
                print(f"  Processed {idx + 1}/{len(test_sequences)} sequences")

    # Average metrics
    avg_loss = total_loss / num_batches
    avg_perplexity = total_perplexity / num_batches
    avg_acc = total_acc / num_batches
    avg_top5_acc = total_top5_acc / num_batches

    print(f"\n{'='*50}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"Average loss: {avg_loss:.4f}")
    print(f"Average perplexity: {avg_perplexity:.4f}")
    print(f"Next-token accuracy: {avg_acc*100:.2f}%")
    print(f"Top-5 accuracy: {avg_top5_acc*100:.2f}%")
    print(f"{'='*50}\n")

    # Model size
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model size: {total_params:,} parameters ({total_params/1e6:.2f}M)")


if __name__ == "__main__":
    main()
