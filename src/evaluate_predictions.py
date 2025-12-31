#!/usr/bin/env python3
"""
Evaluate prediction accuracy by testing on multiple data points.

Usage:
    python evaluate_predictions.py --csv data/AAPL_2024-01-01_2025-12-31_1h.csv --checkpoint checkpoints/best_model.pt --num_predictions 100
"""

import argparse
import pandas as pd
import numpy as np
import torch
from pathlib import Path

from src.gpt_model import GPT, GPTConfig
from src.tokenizer import VQVAETwitterizerOC, consecutive_log_returns


def load_checkpoint(checkpoint_path: str, device: str = "auto"):
    """Load GPT model from checkpoint"""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    config = checkpoint["config"]
    if isinstance(config, dict):
        config = GPTConfig(**config)
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


def prepare_data(csv_path: str):
    """Read and prepare CSV data"""
    df = pd.read_csv(csv_path)
    
    # Skip header rows if present
    if df.iloc[0]['Close'] == 'AAPL' or 'Close' in str(df.iloc[0]['Close']):
        df = df.iloc[2:].reset_index(drop=True)
    
    # Convert Close column to numeric
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna(subset=['Close'])
    
    return df['Close'].values


def encode_window(close_prices: np.ndarray, tokenizer, device: str = "cpu"):
    """Convert price window to tokens"""
    log_returns = consecutive_log_returns(
        pd.DataFrame({'Close': close_prices})
    )
    
    log_returns_tensor = torch.from_numpy(log_returns).float().unsqueeze(0).unsqueeze(-1)
    log_returns_tensor = log_returns_tensor.to(device)
    
    with torch.no_grad():
        token_ids = tokenizer.encode(log_returns_tensor)
    
    return token_ids[0].cpu().tolist()


def predict_next_token(model, seed_tokens: list, temperature: float = 0.8, top_k: int = 50):
    """Predict the next token given seed tokens"""
    model.eval()
    
    with torch.no_grad():
        generated = seed_tokens.copy()
        for token in model.generate(seed_tokens, max_new_tokens=1, temperature=temperature, top_k=top_k):
            generated.append(token)
            break
    
    return generated[-1]


def token_to_log_return(token_id: int, tokenizer, device: str = "cpu"):
    """Convert token ID to log return value"""
    token_tensor = torch.tensor([[token_id]], dtype=torch.long).to(device)
    
    with torch.no_grad():
        recon = tokenizer.decode(token_tensor)
    
    log_return = recon[0, :, 0].mean().cpu().item()
    return log_return


def evaluate_predictions(csv_path: str, checkpoint_path: str, tokenizer_path: str,
                        num_predictions: int = 100, context_length: int = 500,
                        temperature: float = 0.8, top_k: int = 50):
    """
    Evaluate model accuracy by predicting multiple time steps.
    
    Args:
        csv_path: Path to CSV file
        checkpoint_path: Path to model checkpoint
        tokenizer_path: Path to tokenizer model
        num_predictions: Number of predictions to make
        context_length: Number of data points to use as context
        temperature: Sampling temperature
        top_k: Top-k sampling
    
    Returns:
        Dictionary with evaluation metrics
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load models
    print(f"\nLoading GPT model from {checkpoint_path}...")
    gpt_model, config, device = load_checkpoint(checkpoint_path, device=device)
    
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = load_tokenizer(tokenizer_path, device=device)
    
    # Prepare data
    print(f"\nReading CSV: {csv_path}")
    close_prices = prepare_data(csv_path)
    print(f"Total data points: {len(close_prices)}")
    
    # Ensure we have enough data
    min_required = context_length + num_predictions + 1
    if len(close_prices) < min_required:
        print(f"Error: Need at least {min_required} data points, but only have {len(close_prices)}")
        return None
    
    # Storage for predictions and actuals
    predictions = []
    actuals = []
    predicted_prices = []
    actual_prices = []
    
    print(f"\nMaking {num_predictions} predictions...")
    print(f"Context length: {context_length} data points")
    
    # Use data points before the last num_predictions for testing
    # Reserve last num_predictions+1 points (we need the +1 for the actual next values)
    test_start_idx = len(close_prices) - num_predictions - 1
    
    for i in range(num_predictions):
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{num_predictions}")
        
        end_idx = test_start_idx + i
        start_idx = max(0, end_idx - context_length)
        
        context_prices = close_prices[start_idx:end_idx + 1]
        
        # Encode to tokens
        token_list = encode_window(context_prices, tokenizer, device=device)
        
        # Use last tokens as context (limit to model's sequence length)
        max_context = min(len(token_list), config.sequence_len - 1)
        seed_tokens = token_list[-max_context:]
        
        # Predict next token
        next_token = predict_next_token(gpt_model, seed_tokens, temperature=temperature, top_k=top_k)
        
        # Decode to log return
        predicted_log_return = token_to_log_return(next_token, tokenizer, device=device)
        
        # Calculate predicted price
        current_price = context_prices[-1]
        predicted_price = current_price * np.exp(predicted_log_return)
        
        # Get actual next price
        actual_price = close_prices[end_idx + 1]
        actual_log_return = np.log(actual_price / current_price)
        
        # Store results
        predictions.append(predicted_log_return)
        actuals.append(actual_log_return)
        predicted_prices.append(predicted_price)
        actual_prices.append(actual_price)
    
    # Convert to numpy arrays
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    predicted_prices = np.array(predicted_prices)
    actual_prices = np.array(actual_prices)
    
    # Calculate metrics
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    # Log return metrics
    mae_log_return = np.mean(np.abs(predictions - actuals))
    rmse_log_return = np.sqrt(np.mean((predictions - actuals)**2))
    
    print(f"\nLog Return Metrics:")
    print(f"  MAE:  {mae_log_return:.6f}")
    print(f"  RMSE: {rmse_log_return:.6f}")
    
    # Price metrics
    mae_price = np.mean(np.abs(predicted_prices - actual_prices))
    rmse_price = np.sqrt(np.mean((predicted_prices - actual_prices)**2))
    mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100
    
    print(f"\nPrice Metrics:")
    print(f"  MAE:  ${mae_price:.2f}")
    print(f"  RMSE: ${rmse_price:.2f}")
    print(f"  MAPE: {mape:.2f}%")
    
    # Directional accuracy
    predicted_direction = np.sign(predictions)
    actual_direction = np.sign(actuals)
    directional_accuracy = np.mean(predicted_direction == actual_direction) * 100
    
    print(f"\nDirectional Accuracy: {directional_accuracy:.2f}%")
    print(f"  (Correct prediction of price movement direction)")
    
    # Profit simulation (simple strategy)
    # If we predict up, we buy; if down, we sell
    returns_if_followed = actual_direction * predicted_direction  # 1 if correct, -1 if wrong
    cumulative_return = np.sum(actuals * (predicted_direction > 0).astype(float) - 
                               np.abs(actuals) * (predicted_direction < 0).astype(float))
    
    print(f"\nSimple Trading Strategy:")
    print(f"  Cumulative return if following predictions: {cumulative_return:.4f}")
    print(f"  Average return per trade: {cumulative_return/num_predictions:.6f}")
    
    # Statistics
    print(f"\nPrediction Statistics:")
    print(f"  Mean predicted log return: {np.mean(predictions):.6f}")
    print(f"  Mean actual log return:    {np.mean(actuals):.6f}")
    print(f"  Std predicted log return:  {np.std(predictions):.6f}")
    print(f"  Std actual log return:     {np.std(actuals):.6f}")
    
    print("="*70)
    
    return {
        "num_predictions": num_predictions,
        "mae_log_return": mae_log_return,
        "rmse_log_return": rmse_log_return,
        "mae_price": mae_price,
        "rmse_price": rmse_price,
        "mape": mape,
        "directional_accuracy": directional_accuracy,
        "cumulative_return": cumulative_return,
        "predictions": predictions,
        "actuals": actuals,
        "predicted_prices": predicted_prices,
        "actual_prices": actual_prices,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate prediction accuracy")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV file with OHLC data")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to model checkpoint")
    parser.add_argument("--tokenizer", type=str, default="models/tokenizer_model.pt", help="Path to tokenizer model")
    parser.add_argument("--num_predictions", type=int, default=100, help="Number of predictions to evaluate")
    parser.add_argument("--context_length", type=int, default=500, help="Number of data points to use as context")
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
    
    # Run evaluation
    results = evaluate_predictions(
        csv_path=args.csv,
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        num_predictions=args.num_predictions,
        context_length=args.context_length,
        temperature=args.temperature,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
