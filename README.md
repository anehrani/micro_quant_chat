# Micro Quant Chat - GPT for Financial Time Series

A GPT-like transformer model for predicting financial price movements from tokenized price series data. Built using a modern architecture inspired by [nanochat](https://github.com/karpathy/nanochat).

## Quick Start

Using the convenient `mqc` command:

```bash
# Run examples to verify everything works
./scripts/mqc examples

# Train the model
./scripts/mqc train

# Evaluate performance
./scripts/mqc eval

# Generate predictions
./scripts/mqc generate

# View all commands
./scripts/mqc help
```

Or use individual scripts with the run helper:
```bash
./scripts/run.sh src/examples.py
./scripts/run.sh src/train.py
./scripts/run.sh src/evaluate.py
./scripts/run.sh src/generate.py --seed_tokens "80 81 83 89 66"
```

Or use the automated quickstart:
```bash
./scripts/quickstart.sh
```

## 🎯 Four-Stage Training Pipeline

Micro Quant Chat uses a **4-stage training pipeline** inspired by modern LLM training practices. Each stage builds on the previous one to progressively refine the model.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TRAINING PIPELINE OVERVIEW                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐          │
│   │ STAGE 1  │ ──▶ │ STAGE 2  │ ──▶ │ STAGE 3  │ ──▶ │ STAGE 4  │          │
│   │ Pretrain │     │ Midtrain │     │   SFT    │     │    RL    │          │
│   └──────────┘     └──────────┘     └──────────┘     └──────────┘          │
│        │                │                │                │                 │
│        ▼                ▼                ▼                ▼                 │
│   All Assets       Single Asset    Multi-Day        Reward-Based           │
│   Next-Token       Fine-Tuning     Prediction       Optimization           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Stage 1: Pretrain — Learn General Market Patterns

> **Goal**: Train the model to predict the next token across *all* assets.

| Aspect | Details |
|--------|---------|
| **Input** | Combined token stream from all assets (`data/all_tokens.txt`) |
| **Task** | Next-token prediction with cross-entropy loss |
| **Learning** | General market patterns, volatility regimes, trend structures |
| **Output** | `checkpoints/best_model.pt` |

```bash
./scripts/mqc stage1 --epochs 10 --batch 32 --lr 5e-4
```

**What the model learns:**
- Common price movement patterns across different assets
- Market microstructure (mean reversion, momentum, volatility clustering)
- Token distribution and transition probabilities

---

### Stage 2: Midtrain — Specialize on a Single Asset

> **Goal**: Fine-tune the pretrained model on a specific asset (e.g., BTC).

| Aspect | Details |
|--------|---------|
| **Input** | Single asset CSV (e.g., `BTC-USD_*.csv`) |
| **Task** | Teacher-forcing next-token prediction |
| **Learning** | Asset-specific patterns, unique volatility profile |
| **Output** | `checkpoints/midtrain_btc.pt` |

```bash
./scripts/mqc stage2 \
  --in checkpoints/best_model.pt \
  --out checkpoints/midtrain_btc.pt \
  --csv data/BTC-USD_2014-09-17_2025-12-31_1d.csv \
  --epochs 3 --lr 1e-5
```

**Why this matters:**
- BTC has different dynamics than stocks (24/7 trading, higher volatility)
- Specialization helps the model capture asset-specific patterns
- Lower learning rate prevents catastrophic forgetting of Stage 1 knowledge

---

### Stage 3: SFT — Multi-Day Prediction Training

> **Goal**: Train the model to predict *sequences* of future tokens, not just the next one.

| Aspect | Details |
|--------|---------|
| **Input** | Context window (e.g., 128 days of history) |
| **Task** | Predict next H days (horizon, e.g., 32 days) |
| **Learning** | Multi-step trajectory forecasting |
| **Output** | `checkpoints/sft_btc.pt` |

```bash
./scripts/mqc stage3 \
  --in checkpoints/midtrain_btc.pt \
  --out checkpoints/sft_btc.pt \
  --csv data/BTC-USD_2014-09-17_2025-12-31_1d.csv \
  --epochs 10 --context-len 128 --horizon 32
```

**Key difference from Midtrain:**

| Midtrain (Stage 2) | SFT (Stage 3) |
|--------------------|---------------|
| Predict next **1** token | Predict next **H** tokens |
| Teacher forcing only | Autoregressive generation |
| Single-step objective | Trajectory-level objective |

**Analogy to LLMs:**
- Stage 2 is like training on individual word predictions
- Stage 3 is like training on full conversation responses

---

### Stage 4: RL — Reward-Based Fine-Tuning

> **Goal**: Optimize the model using reinforcement learning to maximize prediction quality.

| Aspect | Details |
|--------|---------|
| **Algorithm** | REINFORCE (policy gradient) |
| **Reward Options** | `token_acc` (accuracy) or `return_mse` (price similarity) |
| **Learning** | Optimize for actual prediction quality, not just likelihood |
| **Output** | `checkpoints/rl_btc.pt` |

```bash
./scripts/mqc stage4 \
  --in checkpoints/sft_btc.pt \
  --out checkpoints/rl_btc.pt \
  --csv data/BTC-USD_2014-09-17_2025-12-31_1d.csv \
  --steps 200 --batch-episodes 8 \
  --reward token_acc
```

**Reward functions:**

| Reward | Formula | Best For |
|--------|---------|----------|
| `token_acc` | % of tokens matching ground truth | Discrete accuracy |
| `return_mse` | `-MSE(decoded_returns, actual_returns)` | Financial accuracy |

**Why RL helps:**
- Cross-entropy loss optimizes token likelihood, not trading performance
- RL directly optimizes what we care about (prediction accuracy)
- Can incorporate custom reward signals (e.g., Sharpe ratio)

---

### 🚀 Quick Start: Full Pipeline

Run all four stages in sequence:

```bash
# Stage 1: Pretrain on all assets (5 epochs)
./scripts/mqc stage1 --epochs 5

# Stage 2: Specialize on BTC (3 epochs)
./scripts/mqc stage2 --epochs 3

# Stage 3: Multi-day prediction training (10 epochs)
./scripts/mqc stage3 --epochs 10

# Stage 4: RL fine-tuning (200 steps)
./scripts/mqc stage4 --steps 200

# Generate predictions
./scripts/mqc generate --checkpoint checkpoints/rl_btc.pt
```

---

### 📊 Stage Comparison Summary

| Stage | Objective | Data | LR | Epochs/Steps |
|-------|-----------|------|-----|--------------|
| **1. Pretrain** | Cross-entropy (next token) | All assets | `5e-4` | 5-10 epochs |
| **2. Midtrain** | Cross-entropy (single asset) | BTC only | `1e-5` | 3-5 epochs |
| **3. SFT** | Cross-entropy (trajectory) | BTC only | `5e-6` | 10-20 epochs |
| **4. RL** | REINFORCE (reward) | BTC only | `1e-6` | 100-500 steps |

---

### 🔧 Tokenizer

The tokenizer converts OHLC candles into discrete tokens:

| Feature | Description |
|---------|-------------|
| **Architecture** | VQ-VAE with EMA codebook updates |
| **Input** | 5 features per candle: `log_return`, `body`, `upper_wick`, `lower_wick`, `range` |
| **Output** | Single token ID (0-511) per candle |
| **Codebook Size** | 512 tokens |

Each day of price data → 1 token. Simple and interpretable.

---

> 📚 **Detailed Documentation**: See [docs/FOUR_STAGE_TRAINING.md](docs/FOUR_STAGE_TRAINING.md) for advanced usage.

---

## Project Structure

```
micro_quant_chat/
├── microchat/
│   ├── gpt.py                 # Nanochat-style GPT implementation
│   ├── data.py                # Token stream dataset + loaders
│   ├── train.py               # Train loop + CLI
│   ├── eval.py                # Eval loop + CLI
│   ├── sample.py              # Sampling/generation + CLI
│   └── ckpt.py                # Checkpoint save/load (dict config)
├── src/
│   ├── gpt_model.py           # GPT transformer implementation
│   ├── tokenizer.py           # Price tokenization (VQ-VAE)
│   ├── token_analysis.py      # Analysis tools
│   ├── train.py               # Training script
│   ├── generate.py            # Generation script
│   ├── evaluate.py            # Evaluation script
│   ├── examples.py            # Usage examples
│   └── decode_tokens.py       # Token decoding
├── data/
│   ├── all_tokens.txt         # Tokenized price series (3050 tokens)
│   ├── AAPL_*.csv             # Raw OHLC data (10 assets)
│   └── ...
├── checkpoints/               # Saved models
├── models/                    # Tokenizer models
├── scripts/
│   ├── mqc                    # Main command helper
│   ├── base_train.py           # Nanochat-style entrypoint
│   ├── base_eval.py            # Nanochat-style entrypoint
│   ├── base_sample.py          # Nanochat-style entrypoint
│   ├── run.sh                 # Helper script (uses .venv Python)
│   └── quickstart.sh          # Automated setup and training
└── data_scripts/              # Data processing utilities
```

## What This Model Does

This project trains a GPT-style language model on **tokenized financial price movements**:

1. **Input**: Historical price data (OHLC) from 10 stocks/crypto
2. **Tokenization**: Prices → Log returns → Quantized tokens (0-127)
3. **Training**: GPT learns patterns in token sequences
4. **Prediction**: Generate future token sequences → Price movement forecasts

## Model Architecture

Based on modern GPT design (nanochat-inspired):

- **19M parameters** (6 layers × 512 dim × 8 heads)
- Rotary positional embeddings (RoPE)
- RMSNorm (no learnable params)
- QK normalization for stable attention
- Group Query Attention (GQA) support
- No bias in linear layers
- ReLU² activation in FFN

## Usage Examples

### Training

```bash
# Basic training (10 epochs)
./scripts/mqc train

# Custom configuration
./scripts/mqc train --num_epochs 20 --batch_size 64 --learning_rate 1e-3
```

The training script:
- Loads tokenized data from `data/all_tokens.txt`
- Splits into 80/20 train/val
- Uses AdamW optimizer with cosine annealing
- Saves best model to `checkpoints/best_model.pt`

### Generating Predictions

```bash
# Generate from seed tokens
./scripts/mqc generate --seed_tokens "80 81 83 89 66" --num_generate 50

# Multiple samples with temperature
./scripts/mqc generate --temperature 0.8 --top_k 50 --num_samples 5

# Greedy decoding (temperature=0)
./scripts/mqc generate --temperature 0 --num_generate 100
```

### Evaluation

```bash
# Compute metrics on test set
./scripts/mqc eval

# Analyze token patterns
./scripts/mqc analyze --patterns --outliers
```

Metrics computed:
- Next-token prediction accuracy
- Top-5 accuracy
- Perplexity
- Cross-entropy loss

## Data

### Token Statistics
- **Total tokens**: 3,050
- **Vocabulary**: 128 tokens (0-127)
- **Coverage**: Top 10 tokens = 26.2% of sequences
- **Entropy**: 4.53 (diverse distribution)

### Source Data
- 10 assets: AAPL, AMZN, GOOGL, JPM, META, MSFT, NVDA, TSLA, V, BTC-USD
- Timeframe: 2024-01-01 to 2025-12-31 (hourly data)
- Features: Open, High, Low, Close, Volume

### Tokenization Process

1. **Log Returns**: `log(price_t+1 / price_t)`
2. **VQ-VAE Encoder**: Quantizes returns into discrete tokens
3. **Token IDs**: 0-127 representing different price movements

## Running the Examples

```bash
# See all usage examples
./scripts/mqc examples
```

This demonstrates:
1. Token data analysis
2. Model creation and initialization
3. Forward pass (inference)
4. Token generation
5. Training step
6. Model architecture analysis

## Model Performance

After training:
- **Loss**: ~3-5 (depends on epochs)
- **Perplexity**: ~5-10
- **Next-token accuracy**: 15-25%
- **Top-5 accuracy**: 40-60%

These metrics are reasonable for:
- Small dataset (3K tokens)
- Complex financial data
- 19M parameter model

## Environment Setup

The project uses Python from `.venv`:

```bash
# If you need to recreate the environment:
python3 -m venv .venv
source .venv/bin/activate
pip install torch numpy pandas
```

All scripts use `./scripts/mqc` or `./scripts/run.sh` which automatically uses the venv Python.

## Technical Details

### Rotary Embeddings

```python
def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)
```

### Causal Attention

- Training: Standard causal masking
- Inference: KV cache for efficiency
- Supports GQA (fewer KV heads than Q heads)

### Training Loop

```python
for epoch in epochs:
    for inputs, targets in dataloader:
        loss = model(inputs, targets)
        loss.backward()
        clip_grad_norm(1.0)
        optimizer.step()
```

## Comparison to nanochat

### Similarities ✓
- Rotary embeddings
- QK normalization  
- RMSNorm
- No bias in linear layers
- KV caching

### Differences
- Simpler (single file vs modular)
- Smaller scale (19M vs 2.2B params)
- No distributed training
- Financial domain vs language

## Next Steps

To improve the model:

1. **More data**: Train on longer sequences (more historical data)
2. **Larger model**: Increase layers/width (e.g., 12 layers × 768 dim)
3. **Better tokenization**: Try different quantization schemes
4. **Multi-task**: Predict volatility, volume, etc.
5. **Conditioning**: Add market context (VIX, sector, etc.)
6. **Ensemble**: Train multiple models and average predictions

## Files

- `src/train.py` - Training script with data loading and optimization
- `src/generate.py` - Generate token sequences from trained model
- `src/evaluate.py` - Compute metrics on test set
- `src/examples.py` - Usage demonstrations
- `src/gpt_model.py` - GPT architecture implementation
- `src/tokenizer.py` - Price tokenization using VQ-VAE
- `src/token_analysis.py` - Token statistics and patterns
- `scripts/mqc` - Main command helper script
- `scripts/run.sh` - Helper to run Python from venv
- `MODEL_README.md` - Detailed architecture documentation

## References

1. **nanochat**: https://github.com/karpathy/nanochat
2. **Rotary Embeddings (RoPE)**: https://arxiv.org/abs/2104.09864
3. **Group Query Attention**: https://arxiv.org/abs/2305.13245
4. **QK Normalization**: https://arxiv.org/abs/2405.06899

## License

MIT
