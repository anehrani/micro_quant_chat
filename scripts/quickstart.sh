#!/bin/bash
# Quick start script for training and evaluating the GPT model

set -e

# Get the directory where this script lives
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Get the project root (parent of scripts/)
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Export PYTHONPATH to include project root
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Change to project root directory
cd "$PROJECT_ROOT"

echo "======================================"
echo "Micro Quant Chat - Quick Start"
echo "======================================"
echo ""

# Activate virtual environment
if [ -f .venv/bin/activate ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
    echo "✓ Virtual environment activated"
else
    echo "Error: .venv/bin/activate not found"
    echo "Please create a virtual environment first with: python3 -m venv .venv"
    exit 1
fi

echo ""
echo "Step 1: Installing dependencies..."
python -m pip install -q torch numpy pandas
echo "✓ Dependencies installed"
echo ""

echo "Step 2: Creating checkpoint directory..."
mkdir -p checkpoints
echo "✓ Checkpoint directory created"
echo ""

echo "Step 3: Starting training..."
echo "This will train an 8-layer GPT model on your tokenized price data"
echo "Training will take 2-10 minutes depending on your hardware"
echo ""
python src/train.py \
    --num_epochs 10 \
    --batch_size 32 \
    --learning_rate 5e-4

echo ""
echo "✓ Training complete!"
echo ""

echo "Step 4: Evaluating model..."
python src/evaluate.py

echo ""
echo "Step 5: Generating predictions..."
echo "Generating 50 tokens starting from seed tokens..."
python src/generate.py \
    --seed_tokens "80 81 83 89 66" \
    --num_generate 50 \
    --temperature 0.8 \
    --num_samples 3

echo ""
echo "======================================"
echo "✓ Quick start complete!"
echo "======================================"
echo ""
echo "Next steps:"
echo "  1. Check checkpoints/best_model.pt for saved model"
echo "  2. Review the generated token sequences"
echo "  3. Run python src/generate.py with different prompts"
echo "  4. Modify hyperparameters in src/train.py for better results"
echo ""
