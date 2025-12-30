# Micro Quant Chat - GPT for Financial Time Series

A GPT-like transformer model trained on tokenized financial price series data. Inspired by [nanochat](https://github.com/karpathy/nanochat) by Karpathy, this project implements a minimal, hackable GPT architecture for predicting cryptocurrency and stock price movements.

## Architecture Overview

The model implements a modern transformer architecture with the following features:

### Core Components

- **Rotary Embeddings**: No positional embeddings - uses efficient rotary positional embeddings (RoPE)
- **QK Normalization**: Normalizes query and key vectors for stable attention
- **RMSNorm**: Lightweight normalization without learnable parameters
- **Group Query Attention (GQA)**: Efficient attention for inference
- **No Bias Terms**: Linear layers have no bias for reduced parameters
- **SwiGLU-style MLP**: Uses ReLU² activation function

### Model Configuration

```python
GPTConfig(
    sequence_len=512,      # Context window
    vocab_size=128,        # Token vocabulary from quantized price movements
    n_layer=8,            # Number of transformer blocks
    n_head=8,             # Number of attention heads
    n_kv_head=8,          # KV heads (GQA when < n_head)
    n_embd=512,           # Embedding dimension
)
```

## Data

### Tokenization

Price data is preprocessed into:
- **Log returns**: Consecutive price movements as log(price_t+1 / price_t)
- **Quantization**: Returns are discretized into token IDs (0-127)
- **Vocabulary**: 128 tokens representing different price movement magnitudes

### Data Files

- `data/all_tokens.txt`: Space-separated token IDs (48.8k tokens across 10 stocks)
- `data/AAPL_2024-01-01_2025-12-31_1h.csv` and others: Raw OHLC data
- `data/reconstructed_log_returns.txt`: Actual float log returns for reference

### Supported Assets

- AAPL, AMZN, GOOGL, JPM, META, MSFT, NVDA, TSLA, V (stocks)
- BTC-USD (cryptocurrency)

## Usage

### Training

```bash
# Train model with default configuration
python train.py

# Custom configuration
python train.py --num_epochs 20 --batch_size 64 --learning_rate 1e-3
```

**Training Features:**
- Cosine annealing learning rate scheduler
- Gradient clipping (max norm = 1.0)
- 80/20 train/val split
- Saves best model based on validation loss
- Automatic checkpoint saving

### Evaluation

```bash
# Evaluate on test sequences
python evaluate.py --checkpoint checkpoints/best_model.pt

# Custom sequence length
python evaluate.py --seq_len 512
```

**Metrics computed:**
- Next-token prediction accuracy
- Top-5 accuracy
- Perplexity
- Cross-entropy loss

### Generation

```bash
# Generate price movement predictions from seed tokens
python generate.py --seed_tokens "80 81 83 89 66" --num_generate 100

# Multiple samples with different temperatures
python generate.py --temperature 0.8 --top_k 50 --num_samples 5
```

## Project Structure

```
micro_quant_chat/
├── src/
│   ├── gpt_model.py          # GPT model implementation
│   ├── tokenizer.py          # Tokenization utilities
│   └── decode_tokens.py      # Token decoding
├── data/
│   ├── all_tokens.txt        # Tokenized price series
│   ├── AAPL_*.csv            # Raw price data
│   └── ...
├── data_scripts/
│   ├── download_yahoo_data.py
│   ├── tokenize_data.py
│   └── preprocess_ohlc.py
├── checkpoints/              # Saved models
├── train.py                  # Training script
├── generate.py               # Generation script
├── evaluate.py               # Evaluation script
├── main.py                   # Entry point
└── README.md
```

## Implementation Details

### Rotary Embeddings

Rather than learning positional embeddings, the model uses rotary embeddings that encode position through matrix rotations:

```python
def apply_rotary_emb(x, cos, sin):
    # Split into pairs and apply 2D rotation
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)
```

### Causal Self-Attention

Supports:
- Training: Standard causal masking
- Inference: KV cache for efficient generation
- GQA: Group Query Attention for parameter efficiency

### Training Loop

```python
for epoch in epochs:
    for batch in train_loader:
        loss = model(inputs, targets)
        loss.backward()
        clip_grad_norm(1.0)
        optimizer.step()
    
    val_loss = evaluate(model, val_loader)
    if val_loss < best_loss:
        save_checkpoint()
```

## Model Parameters

Example configuration (8-layer model):
- Embedding dim: 512
- Layers: 8
- Heads: 8
- FFN hidden: 2048 (4x embedding)
- **Total parameters**: ~33M

Larger models can be trained by increasing `n_layer`, `n_embd`, or `n_head`.

## Performance Metrics

Training is efficient due to:
- No positional embeddings (no learnable position parameters)
- RMSNorm (no learnable parameters)
- No bias in linear layers
- Efficient attention implementation

Typical training time:
- 10 epochs: 2-5 minutes (GPU) / 20-50 minutes (CPU)
- Perplexity: ~5-10 (depends on data and model size)

## Inference

### Generation Modes

1. **Greedy**: Pick highest probability token (`temperature=0`)
2. **Sampling**: Sample from distribution (`temperature=1.0`)
3. **Top-K**: Sample from top-k most likely tokens
4. **Top-P**: Nucleus sampling (not yet implemented)

### KV Cache

For efficient inference, the model maintains a key-value cache:
- Reduces redundant computation during token generation
- Enables fast streaming inference
- Memory grows linearly with sequence length

## Extending the Model

### Custom Tokenization

Modify `src/tokenizer.py` to:
- Change quantization scheme (more/fewer tokens)
- Add special tokens (market open/close, volatility, etc.)
- Implement BPE or other tokenization

### Model Architecture

Hyperparameters to tune:
- `n_layer`: Depth (8-24 typical)
- `n_embd`: Width (256-1024 typical)
- `n_head`: Attention heads (8-16 typical)
- `sequence_len`: Context window (256-2048)

### Training Data

Options:
- More assets (add more CSV files)
- Different timeframes (daily, weekly)
- Multi-asset sequences (interleaved prices)
- Market microstructure data (order flow, spreads)

## Comparison to nanochat

### Similarities
- Rotary embeddings
- QK normalization
- RMSNorm
- No bias in linear layers
- Efficient KV cache

### Differences
- Simpler code (single file vs modular)
- No distributed training (yet)
- No specialized optimizers (AdamW vs Muon/AdamW combo)
- Task-agnostic (just language modeling)
- Smaller scale (33M params vs 2.2B params)

## References

1. **Nanochat**: [github.com/karpathy/nanochat](https://github.com/karpathy/nanochat)
2. **Rotary Embeddings**: [RoPE paper](https://arxiv.org/abs/2104.09864)
3. **GQA**: [Group Query Attention](https://arxiv.org/abs/2305.13245)
4. **QK Norm**: [Stabilizing Transformers](https://arxiv.org/abs/2405.06899)

## Future Work

- [ ] Multi-GPU training with DDP
- [ ] Specialized optimizers (Muon for efficiency)
- [ ] KV cache quantization
- [ ] Speculative decoding
- [ ] Fine-tuning on specific assets
- [ ] Conditional generation (e.g., given volatility)
- [ ] Uncertainty estimation via ensemble
- [ ] Real trading backtest integration

## License

MIT

## Author

Created as an exercise in building modern language models for financial data.
