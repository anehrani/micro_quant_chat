# Four-Stage Training (nanochat-style)

This repo supports a **4-stage training pipeline** inspired by nanochat's architecture.

## Overview

| Stage | Name | Purpose | Output |
|-------|------|---------|--------|
| 1 | **Pretrain** | Next-token prediction on all assets | `best_model.pt` |
| 2 | **Midtrain** | Asset-specific training (e.g., BTC) | `midtrain_btc.pt` |
| 3 | **SFT** | Multi-day prediction (like conversation) | `sft_btc.pt` |
| 4 | **RL** | REINFORCE fine-tuning on ground truth | `rl_btc.pt` |

---

## Stage 1 — Pretrain

Entry point: `scripts/stage1_pretrain.py`

```bash
./scripts/mqc stage1 --epochs 10 --batch 32 --lr 5e-4
```

What it does:
- Trains on combined token stream from all assets (`data/all_tokens.txt`)
- Standard next-token prediction with cross-entropy loss
- Learns general market patterns across all assets

---

## Stage 2 — Midtrain (Asset-Specific)

Entry point: `scripts/stage2_sft.py` → `microchat/midtrain.py`

```bash
./scripts/mqc stage2 \
  --in checkpoints/best_model.pt \
  --out checkpoints/midtrain_btc.pt \
  --csv data/BTC-USD_2014-09-17_2025-12-31_1d.csv \
  --epochs 3 --batch 32 --lr 1e-5
```

What it does:
- Loads Stage-1 checkpoint
- Specializes the model on a single asset (BTC)
- Teacher-forcing on target asset tokens only

---

## Stage 3 — SFT (Multi-Day Prediction)

Entry point: `scripts/stage3_sft.py` → `microchat/sft.py`

```bash
./scripts/mqc stage3 \
  --in checkpoints/midtrain_btc.pt \
  --out checkpoints/sft_btc.pt \
  --csv data/BTC-USD_2014-09-17_2025-12-31_1d.csv \
  --epochs 10 --batch 16 --lr 5e-6 \
  --context-len 128 --horizon 32
```

What it does:
- Trains model to predict **multiple future days** given a context window
- Like multi-turn conversation in chat models
- Context (prompt) + Horizon (response) training format

This is the key difference from midtrain:
- **Midtrain**: Single next-token prediction
- **SFT**: Multi-step trajectory prediction

---

## Stage 4 — RL (REINFORCE)

Entry point: `scripts/stage4_rl.py` → `microchat/rl_finetune.py`

```bash
./scripts/mqc stage4 \
  --in checkpoints/sft_btc.pt \
  --out checkpoints/rl_btc.pt \
  --csv data/BTC-USD_2014-09-17_2025-12-31_1d.csv \
  --steps 200 --batch-episodes 8 \
  --context-len 128 --horizon 32 \
  --reward token_acc
```

Reward options:
- `token_acc`: Token-level accuracy vs ground truth
- `return_mse`: Decoded return similarity (requires tokenizer decode)

---

## Quick Start

```bash
# Full pipeline
./scripts/mqc stage1 --epochs 5
./scripts/mqc stage2 --epochs 3
./scripts/mqc stage3 --epochs 10
./scripts/mqc stage4 --steps 200
```

---

## Tokenizer

Uses OHLC VQ-VAE tokenizer (`models/tokenizer_model.pt`):
- Each candle → 1 token
- 5 features: log_return, body, upper_wick, lower_wick, range
- 512 codebook size
