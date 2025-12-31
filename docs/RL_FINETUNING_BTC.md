# Stage 3 — RL Fine-tuning on BTC Ground Truth (REINFORCE)

This repo now includes an optional **Stage 3** fine-tuning step using **REINFORCE** on **BTC** data.

The idea:
- You already have **ground-truth BTC prices**.
- Convert BTC prices → log returns → tokens using the repo’s VQ-VAE tokenizer.
- Treat the GPT as a **policy** that samples the next tokens.
- Score the sampled future tokens against the **ground-truth future tokens**, and update with REINFORCE so sampled sequences become more similar.

This is not RLHF; it’s **RL on a well-defined ground-truth sequence objective**.

---

## What gets optimized

We sample a future token sequence (actions) of length `horizon` conditioned on a past context of length `context_len`.

Let:
- context tokens: $c = (t_1, ..., t_{context\_len})$
- ground-truth future tokens: $y = (t_{context\_len+1}, ..., t_{context\_len+horizon})$
- sampled future tokens: $\hat{y}$

We compute a reward $R(\hat{y}, y)$ and optimize the policy with:

$$
\nabla_\theta \; \mathbb{E}[R] \approx (R - b) \cdot \nabla_\theta \sum_{i=1}^{horizon} \log \pi_\theta(\hat{y}_i | c, \hat{y}_{<i})
$$

where $b$ is a baseline (we use the batch mean reward).

---

## Reward types

The script supports two reward choices:

### 1) `token_acc` (default)
Reward is token-level accuracy:

$$
R = \frac{1}{horizon}\sum_i \mathbf{1}[\hat{y}_i = y_i]
$$

This is simple and stable; it pushes the policy toward matching ground truth.

### 2) `return_mse`
Reward is negative MSE between the tokenizer-decoded return sequences of predicted vs target tokens:

$$
R = -\text{MSE}(decode(\hat{y}), decode(y))
$$

This aligns more directly with “similarity to the target price path,” because it compares reconstructed return sequences.

---

## How BTC data is prepared

Implementation: `microchat/asset_tokens.py`

- Loads BTC CSV: `data/BTC-USD_2014-09-17_2025-12-31_1d.csv`
- Computes close-to-close log returns with `src.tokenizer.consecutive_log_returns`
- Optionally normalizes returns by mean/std estimated across repo CSVs
- Encodes returns into tokens using the trained VQ-VAE tokenizer:
  - tokenizer weights: `models/tokenizer_model.pt`

---

## Run it

### Minimal smoke run
```bash
./scripts/run.sh scripts/stage3_rl.py --steps 20 --batch-episodes 4 --context-len 128 --horizon 16
```

### Use decoded-return similarity reward
```bash
./scripts/run.sh scripts/stage3_rl.py --reward return_mse --steps 50 --batch-episodes 4
```

### Outputs
- Writes a new checkpoint (default): `checkpoints/rl_btc.pt`

---

## Notes / Caveats

- This is a **high-variance** estimator. Expect noisy rewards.
- With small data, RL can overfit quickly.
- `token_acc` reward is effectively another way to push toward ground truth; for many problems, Stage-2 SFT (teacher-forcing) is more sample-efficient.
- If you want “closer to prices,” you may eventually prefer a reward computed from a simple trading simulator (PnL/Sharpe). This script deliberately stays purely “similarity to ground-truth sequence.”
