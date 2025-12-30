#!/usr/bin/env python3
"""
Tokenize OHLC data using TextCandleTokenizer and save for later decoding.
"""

import argparse
import json
import pickle
from pathlib import Path
import sys

try:
    import numpy as np
    import pandas as pd
    from preprocess_ohlc import TextCandleTokenizer
except ImportError as e:
    print(f"Error: Required package not found. Please install: uv pip install numpy pandas")
    sys.exit(1)


def tokenize_csv_file(
    csv_path: Path,
    tokenizer: TextCandleTokenizer,
    output_dir: Path,
    is_first: bool = False
) -> dict:
    """
    Tokenize a single CSV file and save results.
    
    Args:
        csv_path: Path to the CSV file
        tokenizer: TextCandleTokenizer instance (already fitted if not first file)
        output_dir: Directory to save tokenized data
        is_first: If True, fit the tokenizer on this file
        
    Returns:
        Dictionary with statistics about the tokenization
    """
    try:
        # Read CSV
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        
        # Standardize column names
        df.columns = df.columns.str.lower()
        
        if 'open' not in df.columns or 'close' not in df.columns:
            print(f"Warning: {csv_path.name} missing Open/Close columns. Skipping.")
            return None
        
        # Fit tokenizer on first file only
        if is_first:
            print(f"Fitting tokenizer on {csv_path.name}...")
            tokenizer.fit(df, o='open', c='close')
        
        # Encode the data
        token_ids, text, signal = tokenizer.encode(df, o='open', c='close')
        
        # Prepare output filename base
        output_base = csv_path.stem
        
        # Save token IDs (numpy array)
        token_ids_path = output_dir / f"{output_base}_tokens.npy"
        np.save(token_ids_path, token_ids)
        
        # Save token text
        text_path = output_dir / f"{output_base}_tokens.txt"
        with open(text_path, 'w') as f:
            f.write(text)
        
        # Save metadata for decoding
        metadata = {
            'original_file': str(csv_path.name),
            'num_tokens': len(token_ids),
            'num_valid_tokens': int(np.sum(token_ids >= 0)),
            'num_invalid_tokens': int(np.sum(token_ids < 0)),
            'date_range': {
                'start': str(df.index[0]),
                'end': str(df.index[-1])
            },
            'signal_stats': {
                'mean': float(signal.mean()),
                'std': float(signal.std()),
                'min': float(signal.min()),
                'max': float(signal.max())
            }
        }
        
        # Save open prices (needed for decoding)
        open_prices_path = output_dir / f"{output_base}_open_prices.npy"
        np.save(open_prices_path, df['open'].to_numpy())
        
        # Save timestamps
        timestamps_path = output_dir / f"{output_base}_timestamps.npy"
        np.save(timestamps_path, df.index.to_numpy())
        
        # Save metadata
        metadata_path = output_dir / f"{output_base}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Tokenized {csv_path.name}")
        print(f"  Total tokens: {metadata['num_tokens']}")
        print(f"  Valid tokens: {metadata['num_valid_tokens']}")
        print(f"  Invalid tokens: {metadata['num_invalid_tokens']}")
        print(f"  Saved to: {output_dir / output_base}_*")
        
        return metadata
        
    except Exception as e:
        print(f"Error tokenizing {csv_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_all_files(
    data_dir: Path,
    output_dir: Path,
    vocab_size: int = 256,
    signal: str = "log_oc",
    method: str = "quantile",
    rolling_z: int | None = None,
    pattern: str = "*.csv"
):
    """
    Process all CSV files in the data directory.
    """
    # Find all CSV files
    csv_files = sorted(data_dir.glob(pattern))
    
    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV file(s)")
    print(f"Vocab size: {vocab_size}, Signal: {signal}, Method: {method}")
    print("="*60)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create tokenizer
    tokenizer = TextCandleTokenizer(
        vocab_size=vocab_size,
        method=method,
        signal=signal,
        rolling_z=rolling_z
    )
    
    # Process all files
    all_metadata = []
    for i, csv_file in enumerate(csv_files):
        metadata = tokenize_csv_file(
            csv_file,
            tokenizer,
            output_dir,
            is_first=(i == 0)
        )
        if metadata:
            all_metadata.append(metadata)
        print()
    
    # Save the fitted tokenizer
    tokenizer_path = output_dir / "tokenizer.pkl"
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f"✓ Saved tokenizer to {tokenizer_path}")
    
    # Save overall summary
    summary = {
        'num_files': len(all_metadata),
        'vocab_size': vocab_size,
        'signal': signal,
        'method': method,
        'rolling_z': rolling_z,
        'files': all_metadata
    }
    
    summary_path = output_dir / "tokenization_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved summary to {summary_path}")
    
    print("="*60)
    print(f"Tokenization complete! Output in {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Tokenize OHLC data for ML training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Tokenize all CSV files with default settings
  python tokenize_data.py
  
  # Custom vocab size and signal type
  python tokenize_data.py --vocab-size 512 --signal pct_oc
  
  # With rolling z-score normalization
  python tokenize_data.py --rolling-z 256
  
  # Process specific pattern
  python tokenize_data.py --pattern "AAPL*.csv"
        """
    )
    
    parser.add_argument(
        "--data-dir",
        "-d",
        default="../data",
        help="Directory containing CSV files (default: ../data)"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="../data/tokenized",
        help="Output directory for tokenized data (default: ../data/tokenized)"
    )
    parser.add_argument(
        "--vocab-size",
        "-v",
        type=int,
        default=256,
        help="Vocabulary size (default: 256)"
    )
    parser.add_argument(
        "--signal",
        "-s",
        choices=["delta", "pct_oc", "log_oc"],
        default="log_oc",
        help="Signal type (default: log_oc)"
    )
    parser.add_argument(
        "--method",
        "-m",
        choices=["quantile", "fixed"],
        default="quantile",
        help="Binning method (default: quantile)"
    )
    parser.add_argument(
        "--rolling-z",
        "-r",
        type=int,
        default=None,
        help="Rolling window for z-score normalization (default: None)"
    )
    parser.add_argument(
        "--pattern",
        "-p",
        default="*.csv",
        help="Glob pattern to match CSV files (default: *.csv)"
    )
    
    args = parser.parse_args()
    
    # Setup paths
    script_dir = Path(__file__).parent
    data_dir = script_dir / args.data_dir
    output_dir = script_dir / args.output
    
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)
    
    # Process files
    process_all_files(
        data_dir=data_dir,
        output_dir=output_dir,
        vocab_size=args.vocab_size,
        signal=args.signal,
        method=args.method,
        rolling_z=args.rolling_z,
        pattern=args.pattern
    )


if __name__ == "__main__":
    main()
