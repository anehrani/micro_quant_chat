#!/usr/bin/env python3
"""
Download historic OHLC market data from Yahoo Finance.

This script downloads Open, High, Low, Close, Adjusted Close, and Volume data
for specified tickers and saves them to CSV files.
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path
import sys

try:
    import yfinance as yf
    import pandas as pd
except ImportError as e:
    print(f"Error: Required package not found. Please install: uv pip install yfinance pandas")
    sys.exit(1)


def download_ohlc_data(
    ticker: str,
    start_date: str = None,
    end_date: str = None,
    interval: str = "1d",
    output_dir: str = "../data"
) -> None:
    """
    Download OHLC data from Yahoo Finance and save to CSV.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        start_date: Start date in YYYY-MM-DD format (default: 5 years ago)
        end_date: End date in YYYY-MM-DD format (default: today)
        interval: Data interval - 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
        output_dir: Directory to save the CSV file
    """
    # Set default dates if not provided
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=5*365)).strftime("%Y-%m-%d")
    
    print(f"Downloading {ticker} data from {start_date} to {end_date}...")
    
    try:
        # Download data
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval=interval,
            progress=False
        )
        
        if data.empty:
            print(f"Warning: No data found for {ticker}")
            return
        
        # Create output directory if it doesn't exist
        output_path = Path(__file__).parent / output_dir
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        filename = f"{ticker}_{start_date}_{end_date}_{interval}.csv"
        filepath = output_path / filename
        
        # Save to CSV
        data.to_csv(filepath)
        
        print(f"✓ Successfully saved {len(data)} records to {filepath}")
        print(f"  Date range: {data.index[0]} to {data.index[-1]}")
        print(f"  Columns: {', '.join([str(col) for col in data.columns])}")
        
    except Exception as e:
        print(f"Error downloading data for {ticker}: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Download historic OHLC market data from Yahoo Finance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download Apple stock data for the last 5 years
  python download_yahoo_data.py AAPL
  
  # Download Microsoft data for a specific date range
  python download_yahoo_data.py MSFT --start 2020-01-01 --end 2023-12-31
  
  # Download Bitcoin data with hourly intervals
  python download_yahoo_data.py BTC-USD --interval 1h --start 2023-01-01
  
  # Download multiple tickers
  python download_yahoo_data.py AAPL MSFT GOOGL
        """
    )
    
    parser.add_argument(
        "tickers",
        nargs="+",
        help="Stock ticker symbol(s) (e.g., AAPL, MSFT, BTC-USD)"
    )
    parser.add_argument(
        "--start",
        "-s",
        help="Start date (YYYY-MM-DD format, default: 5 years ago)"
    )
    parser.add_argument(
        "--end",
        "-e",
        help="End date (YYYY-MM-DD format, default: today)"
    )
    parser.add_argument(
        "--interval",
        "-i",
        default="1d",
        choices=["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"],
        help="Data interval (default: 1d)"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="../data",
        help="Output directory for CSV files (default: ../data)"
    )
    
    args = parser.parse_args()
    
    # Download data for each ticker
    for ticker in args.tickers:
        download_ohlc_data(
            ticker=ticker,
            start_date=args.start,
            end_date=args.end,
            interval=args.interval,
            output_dir=args.output
        )
        print()  # Empty line between tickers


if __name__ == "__main__":
    main()
