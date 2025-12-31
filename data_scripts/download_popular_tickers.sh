#!/bin/bash

# Download OHLC data for popular tickers
# This script uses the download_yahoo_data.py script to fetch historic market data

# Change to the script directory
cd "$(dirname "$0")"

echo "=========================================="
echo "Downloading OHLC data for popular tickers"
echo "=========================================="
echo ""

# Popular tickers (mix of mega caps, ETFs, crypto)
# AAPL - Apple Inc.
# MSFT - Microsoft Corporation
# GOOGL - Alphabet Inc. (Google)
# AMZN - Amazon.com Inc.
# TSLA - Tesla Inc.
# NVDA - NVIDIA Corporation
# META - Meta Platforms Inc. (Facebook)
# JPM - JPMorgan Chase & Co.
# V - Visa Inc.
# BTC-USD - Bitcoin

TICKERS=(
    "AAPL"
    "MSFT"
    "GOOGL"
    "AMZN"
    "TSLA"
    "NVDA"
    "META"
    "JPM"
    "V"
    "BTC-USD"
    "SPY"
    "QQQ"
    "IWM"
    "DIA"
    "TLT"
    "GLD"
    "BRK-B"
    "MA"
    "ETH-USD"
    "SOL-USD"
)

# Use the project's venv runner so dependencies are consistent.
# With --max, the downloader will fetch period=max for daily+.
python download_yahoo_data.py "${TICKERS[@]}" --interval 1d --max

echo ""
echo "=========================================="
echo "Download complete! Check the ../data folder"
echo "=========================================="
