"""
Decode token IDs back to log returns using the trained tokenizer.
"""
import numpy as np
import torch
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from tokenizer import VQVAETwitterizerOC


def load_model(model_path):
    """Load the trained tokenizer model."""
    tokenizer = VQVAETwitterizerOC(
        patch_size=16,
        emb_dim=32,
        num_codes=128,
        hidden=64,
        beta=0.25,
        ema_decay=0.95,
    )
    
    tokenizer.load_state_dict(torch.load(model_path, map_location='cpu'))
    tokenizer.eval()
    return tokenizer


def decode_tokens(tokens, tokenizer, device='cpu'):
    """
    Decode token IDs back to log returns.
    
    Args:
        tokens: 1D numpy array of token IDs
        tokenizer: VQVAETwitterizerOC model
        device: 'cpu' or 'cuda'
    
    Returns:
        reconstructed log returns: 1D numpy array
    """
    # Convert to tensor
    token_tensor = torch.from_numpy(tokens.astype(np.int64)).unsqueeze(0)  # (1, T')
    token_tensor = token_tensor.to(device)
    
    # Decode
    with torch.no_grad():
        recon = tokenizer.decode(token_tensor)  # (1, T, 1)
    
    # Convert back to numpy and squeeze
    recon_np = recon[0, :, 0].cpu().numpy()
    return recon_np


if __name__ == "__main__":
    # Paths
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    
    model_path = os.path.join(models_dir, "tokenizer_model.pt")
    tokens_path = os.path.join(data_dir, "all_tokens.txt")
    
    print(f"Loading model from {model_path}")
    tokenizer = load_model(model_path)
    
    print(f"Loading tokens from {tokens_path}")
    tokens = np.loadtxt(tokens_path, dtype=np.int32)
    print(f"Loaded {len(tokens)} tokens")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    tokenizer = tokenizer.to(device)
    
    print("\nDecoding tokens back to log returns...")
    log_returns = decode_tokens(tokens, tokenizer, device=device)
    print(f"Reconstructed {len(log_returns)} log returns")
    
    # Save reconstructed log returns
    output_file = os.path.join(data_dir, "reconstructed_log_returns.txt")
    np.savetxt(output_file, log_returns, fmt='%.8f')
    print(f"\nSaved reconstructed log returns to {output_file}")
    
    # Print statistics
    print(f"\nStatistics:")
    print(f"  Min: {log_returns.min():.8f}")
    print(f"  Max: {log_returns.max():.8f}")
    print(f"  Mean: {log_returns.mean():.8f}")
    print(f"  Std: {log_returns.std():.8f}")
