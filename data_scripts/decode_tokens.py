#!/usr/bin/env python3
"""
Decode tokenized data back to Close-Open differences.
"""

import argparse
import json
import pickle
from pathlib import Path
import sys

try:
    import numpy as np
    import pandas as pd
except ImportError as e:
    print(f"Error: Required package not found. Please install: uv pip install numpy pandas")
    sys.exit(1)


def decode_tokens(
    token_file: Path,
    tokenizer_path: Path,
    output_path: Path = None
):
    """
    Decode token IDs back to Close-Open differences.
    
    Args:
        token_file: Path to the *_tokens.npy file
        tokenizer_path: Path to tokenizer.pkl file
        output_path: Optional output path for decoded CSV
    """
    try:
        # Load tokenizer
        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
        
        # Load token IDs
        token_ids = np.load(token_file)
        
        # Get the base name to find corresponding files
        base_name = token_file.stem.replace('_tokens', '')
        token_dir = token_file.parent
        
        # Load open prices
        open_prices_file = token_dir / f"{base_name}_open_prices.npy"
        if not open_prices_file.exists():
            print(f"Error: Open prices file not found: {open_prices_file}")
            return None
        open_prices = np.load(open_prices_file)
        
        # Load timestamps if available
        timestamps_file = token_dir / f"{base_name}_timestamps.npy"
        if timestamps_file.exists():
            timestamps = np.load(timestamps_file, allow_pickle=True)
        else:
            timestamps = None
        
        # Load metadata
        metadata_file = token_dir / f"{base_name}_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        # Decode tokens back to close prices
        close_hat, signal_hat = tokenizer.decode(token_ids, open_prices)
        
        # Calculate C - O
        co_diff = close_hat - open_prices
        
        # Create DataFrame
        result_df = pd.DataFrame({
            'Open': open_prices,
            'Close_Reconstructed': close_hat,
            'C_minus_O': co_diff,
            'Signal_Reconstructed': signal_hat,
            'Token_ID': token_ids
        })
        
        if timestamps is not None:
            result_df.index = pd.to_datetime(timestamps)
            result_df.index.name = 'Datetime'
        
        # Filter out invalid tokens
        valid_mask = token_ids >= 0
        result_df_valid = result_df[valid_mask]
        
        # Display statistics
        print(f"✓ Decoded {token_file.name}")
        print(f"  Total records: {len(result_df)}")
        print(f"  Valid records: {len(result_df_valid)}")
        print(f"  C-O Statistics:")
        print(f"    Mean: {result_df_valid['C_minus_O'].mean():.6f}")
        print(f"    Std:  {result_df_valid['C_minus_O'].std():.6f}")
        print(f"    Min:  {result_df_valid['C_minus_O'].min():.6f}")
        print(f"    Max:  {result_df_valid['C_minus_O'].max():.6f}")
        
        if metadata:
            print(f"  Original date range: {metadata['date_range']['start']} to {metadata['date_range']['end']}")
        
        # Save to CSV if output path provided
        if output_path:
            result_df_valid.to_csv(output_path)
            print(f"  Saved to: {output_path}")
        
        return result_df_valid
        
    except Exception as e:
        print(f"Error decoding {token_file.name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def decode_all_tokens(
    tokenized_dir: Path,
    output_dir: Path = None
):
    """
    Decode all tokenized files in a directory.
    """
    # Find tokenizer
    tokenizer_path = tokenized_dir / "tokenizer.pkl"
    if not tokenizer_path.exists():
        print(f"Error: Tokenizer not found: {tokenizer_path}")
        return
    
    # Find all token files
    token_files = sorted(tokenized_dir.glob("*_tokens.npy"))
    
    if not token_files:
        print(f"No token files found in {tokenized_dir}")
        return
    
    print(f"Found {len(token_files)} token file(s)")
    print("="*60)
    
    # Create output directory if specified
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Decode each file
    for token_file in token_files:
        base_name = token_file.stem.replace('_tokens', '')
        
        if output_dir:
            output_path = output_dir / f"{base_name}_decoded.csv"
        else:
            output_path = None
        
        decode_tokens(token_file, tokenizer_path, output_path)
        print()
    
    print("="*60)
    print("Decoding complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Decode tokenized OHLC data back to C-O differences",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Decode all tokenized files
  python decode_tokens.py
  
  # Decode and save to CSV
  python decode_tokens.py --output ../data/decoded
  
  # Decode specific token file
  python decode_tokens.py --file ../data/tokenized/AAPL_2020-01-01_2023-12-31_1d_tokens.npy
        """
    )
    
    parser.add_argument(
        "--tokenized-dir",
        "-t",
        default="../data/tokenized",
        help="Directory containing tokenized data (default: ../data/tokenized)"
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output directory for decoded CSV files (optional)"
    )
    parser.add_argument(
        "--file",
        "-f",
        help="Decode a specific token file instead of all files"
    )
    
    args = parser.parse_args()
    
    # Setup paths
    script_dir = Path(__file__).parent
    tokenized_dir = script_dir / args.tokenized_dir
    output_dir = Path(script_dir / args.output) if args.output else None
    
    if not tokenized_dir.exists():
        print(f"Error: Tokenized directory not found: {tokenized_dir}")
        sys.exit(1)
    
    if args.file:
        # Decode single file
        token_file = Path(args.file)
        if not token_file.exists():
            print(f"Error: Token file not found: {token_file}")
            sys.exit(1)
        
        tokenizer_path = tokenized_dir / "tokenizer.pkl"
        if not tokenizer_path.exists():
            print(f"Error: Tokenizer not found: {tokenizer_path}")
            sys.exit(1)
        
        base_name = token_file.stem.replace('_tokens', '')
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{base_name}_decoded.csv"
        else:
            output_path = None
        
        decode_tokens(token_file, tokenizer_path, output_path)
    else:
        # Decode all files
        decode_all_tokens(tokenized_dir, output_dir)


if __name__ == "__main__":
    main()
