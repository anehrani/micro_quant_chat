"""
Example usage of the GPT model for price series prediction.
This script demonstrates the complete workflow from data loading to generation.
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.gpt_model import GPT, GPTConfig
from src.token_analysis import TokenAnalyzer, load_tokens


def example_1_token_analysis():
    """Example 1: Analyze the token data"""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Token Data Analysis")
    print("=" * 60)

    # Load tokens
    tokens = load_tokens("data/all_tokens.txt")
    print(f"\nLoaded {len(tokens)} tokens")

    # Analyze
    analyzer = TokenAnalyzer(tokens)
    print(analyzer.summary())

    # Vocabulary distribution
    vocab_stats = analyzer.vocabulary_stats()
    print(f"\nTop token coverage: {vocab_stats['coverage_top_10']*100:.1f}% of sequences")
    print(f"Entropy: {vocab_stats['entropy']:.4f}")

    return tokens


def example_2_create_model():
    """Example 2: Create and initialize model"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Creating GPT Model")
    print("=" * 60)

    # Define configuration
    config = GPTConfig(
        sequence_len=256,
        vocab_size=128,
        n_layer=6,
        n_head=8,
        n_kv_head=8,
        n_embd=512,
    )

    print(f"\nModel Configuration:")
    print(f"  Layers: {config.n_layer}")
    print(f"  Embedding dimension: {config.n_embd}")
    print(f"  Attention heads: {config.n_head}")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Context length: {config.sequence_len}")

    # Create model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT(config)
    model.init_weights()
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Device: {device}")

    return model, config, device


def example_3_forward_pass(model, device):
    """Example 3: Run a forward pass"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Forward Pass (Inference)")
    print("=" * 60)

    # Create random input
    batch_size = 2
    seq_len = 256
    inputs = torch.randint(0, 128, (batch_size, seq_len), device=device)

    print(f"\nInput shape: {inputs.shape}")

    # Forward pass (inference)
    with torch.no_grad():
        logits = model(inputs)

    print(f"Output logits shape: {logits.shape}")
    print(f"Logits min: {logits.min():.4f}, max: {logits.max():.4f}")

    # Get next token probabilities
    next_token_logits = logits[:, -1, :]
    next_token_probs = torch.softmax(next_token_logits, dim=-1)
    next_tokens = torch.argmax(next_token_probs, dim=-1)

    print(f"\nPredicted next tokens: {next_tokens.tolist()}")
    print(f"Confidence: {next_token_probs.max(dim=-1)[0].tolist()}")


def example_4_generation(model, config, device):
    """Example 4: Generate sequence"""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Token Generation")
    print("=" * 60)

    # Start with seed tokens
    seed_tokens = [80, 81, 83, 89, 66]
    print(f"\nSeed tokens: {seed_tokens}")

    # Generate
    print("Generating 30 tokens...")
    generated = seed_tokens.copy()

    with torch.no_grad():
        for token in model.generate(
            seed_tokens, max_new_tokens=30, temperature=0.8, top_k=50
        ):
            generated.append(token)

    print(f"Generated sequence: {generated}")
    print(f"New tokens: {generated[len(seed_tokens):]}")


def example_5_batch_training(model, config, device):
    """Example 5: Single training step"""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Training Step")
    print("=" * 60)

    # Create dummy training data
    batch_size = 4
    seq_len = 256
    inputs = torch.randint(0, 128, (batch_size, seq_len), device=device)
    targets = torch.randint(0, 128, (batch_size, seq_len), device=device)

    # Forward and backward
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    print(f"\nBatch shape: {inputs.shape}")

    # Forward pass
    loss = model(inputs, targets)
    print(f"Loss: {loss.item():.6f}")

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    print("✓ Training step complete")


def example_6_model_analysis(model):
    """Example 6: Analyze model"""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Model Analysis")
    print("=" * 60)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Layer breakdown
    print(f"\nLayer breakdown:")
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            params = module.weight.numel()
            if hasattr(module, "bias") and module.bias is not None:
                params += module.bias.numel()
            print(f"  {name}: {params:,} params")


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("MICRO QUANT CHAT - USAGE EXAMPLES")
    print("=" * 60)

    # Example 1: Token analysis
    tokens = example_1_token_analysis()

    # Example 2: Model creation
    model, config, device = example_2_create_model()

    # Example 3: Forward pass
    example_3_forward_pass(model, device)

    # Example 4: Generation
    example_4_generation(model, config, device)

    # Example 5: Training step
    example_5_batch_training(model, config, device)

    # Example 6: Model analysis
    example_6_model_analysis(model)

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Run: python train.py")
    print("  2. Run: python evaluate.py")
    print("  3. Run: python generate.py --seed_tokens '80 81 83 89 66'")
    print()


if __name__ == "__main__":
    main()
