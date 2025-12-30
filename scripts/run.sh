#!/bin/bash
# Helper script to run Python commands using the virtual environment

# Check if .venv exists
if [ ! -d ".venv" ]; then
    echo "Error: .venv directory not found"
    echo "Creating virtual environment..."
    python3 -m venv .venv
    echo "Installing dependencies..."
    .venv/bin/python -m pip install torch numpy pandas
fi

# Run the Python command with any arguments passed to this script
.venv/bin/python "$@"
