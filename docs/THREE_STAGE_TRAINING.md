# Three-Stage Training (nanochat-style)

This repo now supports a simple **3-stage training pipeline** inspired by nanochat’s “scripts as entrypoints” style.

Stages:
1. **Stage 1 (Pretrain)**: next-token prediction on a large token stream (`data/all_tokens.txt`).
2. **Stage 2 (SFT)**: supervised fine-tuning on a *single asset* (BTC) using teacher-forcing on BTC-only tokens.
3. **Stage 3 (RL)**: REINFORCE fine-tuning on BTC ground truth to improve similarity to the target.

> Practical note
> - Stage 2 and Stage 3 are BTC-focused because you requested “choose one asset like BTC.”
> - The same pattern can be extended to other assets by swapping the CSV path.

---

## Stage 1 — Pretrain

Entry point:
- `scripts/stage1_pretrain.py` (same as `scripts/base_train.py`, but named as Stage 1)

Command:
```bash
./scripts/mqc train --num_epochs 10 --batch_size 32 --learning_rate 5e-4
```
Or explicitly:
```bash
./scripts/run.sh scripts/stage1_pretrain.py --epochs 10 --batch 32 --lr 5e-4 --data data/all_tokens.txt
```

Output:
- `checkpoints/best_model.pt`

---

## Stage 2 — SFT (BTC)

Entry point:
- `scripts/stage2_sft.py`

What it does:
- Loads Stage-1 checkpoint
- Converts BTC CSV → log returns → tokens using the repo tokenizer
- Runs teacher-forcing fine-tuning on BTC tokens only

Command:
```bash
./scripts/run.sh scripts/stage2_sft.py \
  --in checkpoints/best_model.pt \
  --out checkpoints/sft_btc.pt \
  --csv data/BTC-USD_2014-09-17_2025-12-31_1d.csv \
  --epochs 3 --batch 32 --lr 1e-5
```

Output:
- `checkpoints/sft_btc.pt`

---

## Stage 3 — RL (BTC, REINFORCE)

Entry point:
- `scripts/stage3_rl.py`

Command (token accuracy reward):
```bash
./scripts/run.sh scripts/stage3_rl.py \
  --in checkpoints/sft_btc.pt \
  --out checkpoints/rl_btc.pt \
  --csv data/BTC-USD_2014-09-17_2025-12-31_1d.csv \
  --steps 200 --batch-episodes 8 --context-len 128 --horizon 32 \
  --reward token_acc
```

Alternative reward (decoded return similarity):
```bash
./scripts/run.sh scripts/stage3_rl.py --reward return_mse
```

Output:
- `checkpoints/rl_btc.pt`

---

## Recommended order

1. Stage 1: produce `checkpoints/best_model.pt`
2. Stage 2: produce `checkpoints/sft_btc.pt`
3. Stage 3: produce `checkpoints/rl_btc.pt`

---

## Tokenizer compatibility

BTC tokenization uses `models/tokenizer_model.pt`.

The loader is designed to infer tokenizer hyperparameters (notably `patch_size`) directly from the checkpoint so it doesn’t break if the tokenizer was trained with different settings.
