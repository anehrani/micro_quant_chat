"""
Inference script for GPT model on price series.
Generate future price movements from token sequences.
"""

import torch
import argparse
from pathlib import Path

from src.gpt_model import GPT, GPTConfig


def load_checkpoint(checkpoint_path: str, device: str = "auto") -> tuple:
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

    print(f"Loaded model from {checkpoint_path}")
    print(f"Config: {config}")

    return model, config, device


def generate_sequence(
    model: GPT,
    seed_tokens: list,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 50,
    device: str = "cpu",
) -> list:
    """
    Generate a sequence of tokens
    
    Args:
        model: Trained GPT model
        seed_tokens: Starting tokens
        max_new_tokens: How many tokens to generate
        temperature: Sampling temperature (0 = greedy, higher = more random)
        top_k: Top-k sampling
        device: Device to use
    
    Returns:
        Generated token sequence
    """
    model.eval()
    generated = seed_tokens.copy()

    with torch.no_grad():
        for token in model.generate(
            seed_tokens, max_new_tokens, temperature=temperature, top_k=top_k
        ):
            generated.append(token)

    return generated


def main():
    parser = argparse.ArgumentParser(description="Generate price series predictions")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to checkpoint")
    parser.add_argument("--seed_tokens", type=str, default="80 81 83 89 66", help="Starting tokens (space-separated)")
    parser.add_argument("--num_generate", type=int, default=50, help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of different samples to generate")

    args = parser.parse_args()

    # Check checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return

    # Load model
    model, config, device = load_checkpoint(args.checkpoint, args.device)

    # Parse seed tokens
    seed_tokens = list(map(int, args.seed_tokens.split()))
    print(f"\nSeed tokens: {seed_tokens}")
    print(f"Generating {args.num_generate} new tokens ({args.num_samples} samples)\n")

    # Generate samples
    for sample_idx in range(args.num_samples):
        generated = generate_sequence(
            model,
            seed_tokens,
            max_new_tokens=args.num_generate,
            temperature=args.temperature,
            top_k=args.top_k,
            device=str(device),
        )

        print(f"Sample {sample_idx + 1}:")
        print(f"  Full sequence: {generated}")
        print(f"  New tokens: {generated[len(seed_tokens) :]}")
        print()


if __name__ == "__main__":
    main()
