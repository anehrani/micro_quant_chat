#!/bin/bash
# Helper script to run Python commands using the virtual environment

# Check if .venv exists
if [ ! -d ".venv" ]; then
    echo "Error: .venv directory not found"
    echo "Creating virtual environment..."
    python3 -m venv .venv
    echo "Installing dependencies (uv)..."
    if ! command -v uv >/dev/null 2>&1; then
        echo "Error: 'uv' is not installed. Install it first: https://docs.astral.sh/uv/"
        exit 1
    fi
    uv pip install --python .venv/bin/python torch numpy pandas yfinance
fi

# Ensure dependencies are available even if .venv already existed
if ! command -v uv >/dev/null 2>&1; then
    echo "Error: 'uv' is not installed. Install it first: https://docs.astral.sh/uv/"
    exit 1
fi
uv pip install --python .venv/bin/python -q torch numpy pandas yfinance >/dev/null 2>&1 || true

# Run the Python command with any arguments passed to this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

.venv/bin/python "$@"
