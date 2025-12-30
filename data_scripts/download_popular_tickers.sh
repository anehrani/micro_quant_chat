#!/bin/bash

# Download 4-hour OHLC data for 10 popular tickers
# This script uses the download_yahoo_data.py script to fetch historic market data

# Change to the script directory
cd "$(dirname "$0")"

echo "=========================================="
echo "Downloading Hourly (1H) OHLC data for popular tickers"
echo "=========================================="
echo ""

# List of 10 famous tickers
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
)

# Download data for all tickers with hourly interval
# Yahoo Finance limits intraday data to ~730 days, so we use 2 years
python3 download_yahoo_data.py "${TICKERS[@]}" --interval 1h --start $(date -v-730d +%Y-%m-%d)

echo ""
echo "=========================================="
echo "Download complete! Check the ../data folder"
echo "=========================================="
