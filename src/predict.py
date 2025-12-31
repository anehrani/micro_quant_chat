#!/usr/bin/env python3
"""
Predict next close price from a CSV file using the trained model.

Usage:
    python predict.py --csv data/AAPL_2024-01-01_2025-12-31_1h.csv --checkpoint checkpoints/best_model.pt
"""

import argparse
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import json

from src.gpt_model import GPT, GPTConfig
from src.tokenizer import VQVAETwitterizerOC, consecutive_log_returns


def load_checkpoint(checkpoint_path: str, device: str = "auto"):
    """Load GPT model from checkpoint"""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    config = checkpoint["config"]
    model = GPT(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    return model, config, device


def load_tokenizer(model_path: str, device: str = "cpu"):
    """Load the trained tokenizer model"""
    tokenizer = VQVAETwitterizerOC(
        patch_size=16,
        emb_dim=32,
        num_codes=128,
        hidden=64,
        beta=0.25,
        ema_decay=0.95,
    )
    
    tokenizer.load_state_dict(torch.load(model_path, map_location=device))
    tokenizer.eval()
    tokenizer = tokenizer.to(device)
    
    return tokenizer


def _load_tokenizer_stats(stats_path: str) -> dict | None:
    p = Path(stats_path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _compute_training_log_return_stats_from_repo_data(data_dir: str = "data") -> dict | None:
    """Best-effort fallback: recompute mean/std from the repo's CSVs.

    This is only used if tokenizer_stats.json is missing.
    """
    p = Path(data_dir)
    if not p.exists():
        return None

    csv_files = sorted(p.glob("*.csv"))
    if not csv_files:
        return None

    all_r = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            r = consecutive_log_returns(df, c="Close")
            if r.size:
                all_r.append(r.astype(np.float64))
        except Exception:
            continue

    if not all_r:
        return None

    r = np.concatenate(all_r)
    r = r[np.isfinite(r)]
    if r.size < 10:
        return None

    return {"mean": float(r.mean()), "std": float(r.std() + 1e-8)}


def csv_to_tokens(csv_path: str, tokenizer, device: str = "cpu", context_length: int = 100):
    """
    Read CSV and convert to tokens using the tokenizer.
    
    Args:
        csv_path: Path to CSV file with OHLC data
        tokenizer: VQVAETwitterizerOC model
        device: Device to use
        context_length: Number of recent data points to use
    
    Returns:
        token_ids: List of token IDs
        last_close: The last close price in the data
        log_returns: The log returns from the data
    """
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Skip the first two rows (headers) and get Close column
    # The CSV has "Price" in first row and "Ticker" in second row
    if df.iloc[0]['Close'] == 'AAPL' or 'Close' in str(df.iloc[0]['Close']):
        df = df.iloc[2:].reset_index(drop=True)  # Skip header rows
    
    # Convert Close column to numeric
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna(subset=['Close'])
    
    # Get the last N close prices for context
    close_prices = df['Close'].tail(context_length + 1).values
    
    # Calculate log returns
    log_returns = consecutive_log_returns(
        pd.DataFrame({'Close': close_prices})
    )

    # IMPORTANT: Tokenizer was trained on normalized log returns.
    # Use saved stats when available; otherwise fall back to recomputing from repo data.
    stats = _load_tokenizer_stats("checkpoints/tokenizer_stats.json")
    if stats is None:
        stats = _compute_training_log_return_stats_from_repo_data("data")

    if stats is not None and "mean" in stats and "std" in stats:
        log_returns = (log_returns - float(stats["mean"])) / float(stats["std"])  # type: ignore[assignment]
    
    # Convert to tensor and encode with tokenizer
    log_returns_tensor = torch.from_numpy(log_returns).float().unsqueeze(0).unsqueeze(-1)  # (1, T, 1)
    log_returns_tensor = log_returns_tensor.to(device)
    
    with torch.no_grad():
        token_ids = tokenizer.encode(log_returns_tensor)  # (1, T')
    
    token_list = token_ids[0].cpu().tolist()
    last_close = close_prices[-1]
    
    return token_list, last_close, log_returns


def predict_next_token(model, seed_tokens: list, temperature: float = 0.8, top_k: int = 50):
    """
    Predict the next token given seed tokens.
    
    Args:
        model: Trained GPT model
        seed_tokens: List of token IDs
        temperature: Sampling temperature
        top_k: Top-k sampling
    
    Returns:
        next_token: The predicted next token ID
    """
    model.eval()
    
    with torch.no_grad():
        # Generate 1 token
        generated = seed_tokens.copy()
        for token in model.generate(seed_tokens, max_new_tokens=1, temperature=temperature, top_k=top_k):
            generated.append(token)
            break
    
    return generated[-1]


def predict_expected_log_return(
    model: GPT,
    seed_tokens: list[int],
    tokenizer: VQVAETwitterizerOC,
    temperature: float = 0.0,
    top_k: int | None = None,
    device: str | torch.device = "cpu",
) -> float:
    """Predict E[log_return] from the model's next-token distribution.

    This is usually more stable than sampling a single next token.
    """
    model.eval()
    device = torch.device(device) if isinstance(device, str) else device

    with torch.no_grad():
        ids = torch.tensor([seed_tokens], dtype=torch.long, device=device)
        logits = model(ids)[:, -1, :]  # (1, vocab)

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = logits.clone()
            logits[logits < v[:, [-1]]] = -float("inf")

        if temperature and temperature > 0:
            logits = logits / float(temperature)

        probs = torch.softmax(logits, dim=-1)[0]  # (vocab,)

        # Precompute token->mean(log_return) using the tokenizer decoder/codebook
        token_ids = torch.arange(0, model.config.vocab_size, dtype=torch.long, device=device).unsqueeze(0)
        recon = tokenizer.decode(token_ids)  # (1, T, 1) where T ~= patch_size * vocab_size? Actually decoder upsamples.

        # Decoder output length depends on token sequence length; here it's (1, patch_size * vocab, 1)
        # We chunk into per-token patches and average each patch.
        patch = tokenizer.patch_size
        recon = recon[0, :, 0]
        if recon.numel() < patch * model.config.vocab_size:
            # Fallback: average entire recon if shape isn't as expected.
            token_vals = recon.mean().repeat(model.config.vocab_size)
        else:
            token_vals = recon[: patch * model.config.vocab_size].view(model.config.vocab_size, patch).mean(dim=1)

        expected = (probs * token_vals.float()).sum().item()
        return float(expected)


def token_to_log_return(token_id: int, tokenizer, device: str = "cpu"):
    """
    Convert a single token ID back to its log return value.
    
    Args:
        token_id: Token ID to decode
        tokenizer: VQVAETwitterizerOC model
        device: Device to use
    
    Returns:
        log_return: Approximate log return value
    """
    # Create a token tensor
    token_tensor = torch.tensor([[token_id]], dtype=torch.long).to(device)
    
    with torch.no_grad():
        # Decode the token
        recon = tokenizer.decode(token_tensor)  # (1, patch_size, 1)
    
    # Take the mean of the reconstructed patch as the representative value
    log_return = recon[0, :, 0].mean().cpu().item()
    
    return log_return


def predict_next_close(csv_path: str, checkpoint_path: str, tokenizer_path: str = "models/tokenizer_model.pt", 
                       context_length: int = 500, temperature: float = 0.8, top_k: int = 50):
    """
    Complete pipeline: Load CSV, predict next close price.
    
    Args:
        csv_path: Path to CSV file
        checkpoint_path: Path to model checkpoint
        tokenizer_path: Path to tokenizer model
        context_length: Number of recent data points to use (default 500 to get ~30 tokens with patch_size=16)
        temperature: Sampling temperature
        top_k: Top-k sampling
    
    Returns:
        Dictionary with prediction results
    """
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load models
    print(f"\nLoading GPT model from {checkpoint_path}...")
    gpt_model, config, device = load_checkpoint(checkpoint_path, device=device)
    print(f"Model config: vocab_size={config.vocab_size}, n_layer={config.n_layer}, n_embd={config.n_embd}")
    
    print(f"\nLoading tokenizer from {tokenizer_path}...")
    tokenizer = load_tokenizer(tokenizer_path, device=device)
    
    # Process CSV
    print(f"\nReading CSV: {csv_path}")
    token_list, last_close, log_returns = csv_to_tokens(csv_path, tokenizer, device=device, context_length=context_length)
    print(f"Extracted {len(token_list)} tokens from {len(log_returns)} log returns")
    print(f"Last close price: ${last_close:.2f}")
    
    # Use the last tokens as context (limit to model's context window)
    max_context = min(len(token_list), config.sequence_len - 1)
    seed_tokens = token_list[-max_context:]
    print(f"\nUsing {len(seed_tokens)} tokens as context")
    print(f"Context tokens: {seed_tokens[:10]}..." if len(seed_tokens) > 10 else f"Context tokens: {seed_tokens}")
    
    # Predict expected log return (more stable than sampling)
    print(f"\nPredicting expected next log return (temperature={temperature}, top_k={top_k})...")
    predicted_log_return = predict_expected_log_return(
        gpt_model,
        seed_tokens,
        tokenizer,
        temperature=temperature,
        top_k=top_k,
        device=device,
    )
    print(f"Predicted log return (expected): {predicted_log_return:.6f}")
    
    # Calculate predicted close price
    # log_return ≈ log(next_close / last_close)
    # next_close = last_close * exp(log_return)
    predicted_close = last_close * np.exp(predicted_log_return)
    price_change = predicted_close - last_close
    price_change_pct = (price_change / last_close) * 100
    
    print(f"\n{'='*60}")
    print(f"PREDICTION RESULTS")
    print(f"{'='*60}")
    print(f"Current close price:   ${last_close:.2f}")
    print(f"Predicted close price: ${predicted_close:.2f}")
    print(f"Expected change:       ${price_change:+.2f} ({price_change_pct:+.2f}%)")
    print(f"{'='*60}")
    
    return {
        "last_close": last_close,
        "predicted_close": predicted_close,
        "price_change": price_change,
        "price_change_pct": price_change_pct,
        "predicted_token": next_token,
        "predicted_log_return": predicted_log_return,
        "num_context_tokens": len(seed_tokens),
    }


def main():
    parser = argparse.ArgumentParser(description="Predict next close price from CSV data")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV file with OHLC data")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to model checkpoint")
    parser.add_argument("--tokenizer", type=str, default="checkpoints/tokenizer_model.pt", help="Path to tokenizer model")
    parser.add_argument("--context_length", type=int, default=500, help="Number of recent data points to use (500 gives ~30 tokens)")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    
    args = parser.parse_args()
    
    # Check files exist
    if not Path(args.csv).exists():
        print(f"Error: CSV file not found: {args.csv}")
        return
    
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return
    
    if not Path(args.tokenizer).exists():
        print(f"Error: Tokenizer not found: {args.tokenizer}")
        return
    
    # Run prediction
    result = predict_next_close(
        csv_path=args.csv,
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        context_length=args.context_length,
        temperature=args.temperature,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
