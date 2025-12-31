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
import time
from typing import Iterable

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
    output_dir: str = "../data",
    max_history: bool = False,
    max_retries: int = 3,
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

    # Yahoo Finance imposes limits for intraday intervals. When max_history is requested,
    # we pick the largest allowed lookback window for the interval.
    intraday_lookback_days = {
        "1m": 7,
        "2m": 60,
        "5m": 60,
        "15m": 60,
        "30m": 60,
        "60m": 730,
        "90m": 60,
        "1h": 730,
    }
    is_intraday = interval in intraday_lookback_days

    if max_history and is_intraday:
        # This is a Yahoo limitation (not this script): intraday history is capped.
        # If the user wants 10+ years, they must use 1d/1wk/1mo.
        days = intraday_lookback_days[interval]
        print(
            f"Note: Yahoo limits intraday interval '{interval}' to about {days} days. "
            "For 10+ years, use --interval 1d (or 1wk/1mo) with --max."
        )

    yf_kwargs = {
        "interval": interval,
        "progress": False,
        "auto_adjust": False,
        "actions": False,
    }

    if max_history and not is_intraday:
        # For daily+ bars, yfinance supports period="max".
        yf_kwargs["period"] = "max"
        start_str_for_log = "MAX"
        end_str_for_log = "MAX"
    else:
        # Use explicit start/end.
        if start_date is None:
            if max_history and is_intraday:
                days = intraday_lookback_days[interval]
                start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            else:
                start_date = (datetime.now() - timedelta(days=5 * 365)).strftime("%Y-%m-%d")

        # yfinance uses an exclusive 'end' date. Make it inclusive by adding 1 day.
        # (Important when user passes --end or expects "up to today").
        end_dt = datetime.fromisoformat(end_date) + timedelta(days=1)
        yf_kwargs["start"] = start_date
        yf_kwargs["end"] = end_dt.strftime("%Y-%m-%d")
        start_str_for_log = start_date
        end_str_for_log = end_date

    print(f"Downloading {ticker} data from {start_str_for_log} to {end_str_for_log} (interval={interval})...")
    
    try:
        last_err = None
        data = None
        for attempt in range(1, max_retries + 1):
            try:
                data = yf.download(ticker, **yf_kwargs)
                break
            except Exception as e:
                last_err = e
                wait_s = min(30, 2 ** attempt)
                print(f"  Download failed (attempt {attempt}/{max_retries}): {e}")
                if attempt < max_retries:
                    print(f"  Retrying in {wait_s}s...")
                    time.sleep(wait_s)
        if data is None:
            raise RuntimeError(f"Download failed for {ticker}: {last_err}")
        
        if data.empty:
            print(f"Warning: No data found for {ticker}")
            return
        
        # Create output directory if it doesn't exist
        output_path = Path(__file__).parent / output_dir
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate filename based on actual returned range (more informative than requested dates)
        actual_start = str(data.index[0]).split(" ")[0]
        actual_end = str(data.index[-1]).split(" ")[0]
        filename = f"{ticker}_{actual_start}_{actual_end}_{interval}.csv"
        filepath = output_path / filename
        
        # Save to CSV
        data.to_csv(filepath)
        
        print(f"✓ Successfully saved {len(data)} records to {filepath}")
        print(f"  Date range: {data.index[0]} to {data.index[-1]}")
        print(f"  Columns: {', '.join([str(col) for col in data.columns])}")
        
    except Exception as e:
        print(f"Error downloading data for {ticker}: {e}")
        return


def _parse_tickers_file(path: str) -> list[str]:
    p = Path(path)
    raw = p.read_text().strip().replace(",", " ")
    if not raw:
        return []
    return [t.strip() for t in raw.split() if t.strip()]


def _dedupe_preserve(items: Iterable[str]) -> list[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


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
        nargs="*",
        help="Stock ticker symbol(s) (e.g., AAPL, MSFT, BTC-USD)"
    )
    parser.add_argument(
        "--tickers-file",
        type=str,
        default=None,
        help="Path to a file containing tickers (whitespace/comma separated)"
    )
    parser.add_argument(
        "--add-popular",
        action="store_true",
        help="Append a preset list of popular tickers (SPY/QQQ + large caps + crypto)"
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
        "--max",
        dest="max_history",
        action="store_true",
        help="Download the maximum available history (intraday auto-limited by Yahoo)"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="../data",
        help="Output directory for CSV files (default: ../data)"
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Max retries per ticker (default: 3)"
    )
    
    args = parser.parse_args()
    
    popular = [
        # Broad market ETFs
        "SPY",
        "QQQ",
        "IWM",
        "DIA",
        "TLT",
        "GLD",
        # Mega caps / liquid names
        "AAPL",
        "MSFT",
        "NVDA",
        "AMZN",
        "GOOGL",
        "META",
        "TSLA",
        "BRK-B",
        "JPM",
        "V",
        "MA",
        # Crypto
        "BTC-USD",
        "ETH-USD",
        "SOL-USD",
    ]

    tickers = list(args.tickers)
    if args.tickers_file:
        tickers.extend(_parse_tickers_file(args.tickers_file))
    if args.add_popular:
        tickers.extend(popular)
    tickers = _dedupe_preserve([t.strip() for t in tickers if t.strip()])

    if not tickers:
        print("Error: No tickers provided. Pass tickers as args, --tickers-file, or --add-popular.")
        sys.exit(2)

    # Download data for each ticker
    for ticker in tickers:
        download_ohlc_data(
            ticker=ticker,
            start_date=args.start,
            end_date=args.end,
            interval=args.interval,
            output_dir=args.output,
            max_history=args.max_history,
            max_retries=args.retries,
        )
        print()  # Empty line between tickers


if __name__ == "__main__":
    main()
