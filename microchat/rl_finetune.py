from __future__ import annotations

import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F

from .asset_tokens import asset_tokens_from_csv, load_tokenizer
from .ckpt import load_checkpoint, save_checkpoint
from .device import resolve_device


RewardType = Literal["token_acc", "return_mse"]


@torch.no_grad()
def _reward_token_accuracy(sampled: list[int], target: list[int]) -> float:
    if not target:
        return 0.0
    n = min(len(sampled), len(target))
    correct = sum(1 for i in range(n) if sampled[i] == target[i])
    return float(correct) / float(n)


@torch.no_grad()
def _reward_return_mse(
    *,
    sampled: list[int],
    target: list[int],
    tokenizer,
    device: torch.device,
) -> float:
    """Reward based on decoded return-path similarity.

    We decode token sequences back into reconstructed return sequences and use negative MSE.
    Larger reward is better.
    """
    if not sampled or not target:
        return 0.0

    s = torch.tensor([sampled], dtype=torch.long, device=device)
    t = torch.tensor([target], dtype=torch.long, device=device)

    # Decode to (1, T, 1) where T ~ patch_size * len(tokens)
    sr = tokenizer.decode(s)[0, :, 0]
    tr = tokenizer.decode(t)[0, :, 0]

    n = min(sr.numel(), tr.numel())
    if n == 0:
        return 0.0

    sr = sr[:n]
    tr = tr[:n]
    mse = F.mse_loss(sr, tr, reduction="mean").item()

    # Negative MSE as reward (closer -> higher)
    return -float(mse)


def _top_k_filter(logits: torch.Tensor, top_k: int | None) -> torch.Tensor:
    if top_k is None or top_k <= 0:
        return logits
    k = min(int(top_k), logits.size(-1))
    v, _ = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float("inf")
    return out


def _sample_one_episode(
    *,
    model,
    context: list[int],
    target: list[int],
    horizon: int,
    temperature: float,
    top_k: int | None,
    device: torch.device,
) -> tuple[list[int], torch.Tensor, torch.Tensor]:
    """Sample an action sequence + compute sum logprob and sum entropy."""

    ids = torch.tensor([context], dtype=torch.long, device=device)
    sampled: list[int] = []
    sum_logprob = torch.zeros((), device=device)
    sum_entropy = torch.zeros((), device=device)

    for _ in range(horizon):
        logits = model(ids)[:, -1, :]

        if temperature == 0.0:
            next_tok = torch.argmax(logits, dim=-1)
            # logprob for greedy token (still defined)
            logp = F.log_softmax(logits, dim=-1).gather(1, next_tok[:, None]).squeeze(1)
            ent = torch.zeros_like(logp)
        else:
            scaled = logits / max(1e-8, float(temperature))
            scaled = _top_k_filter(scaled, top_k)
            probs = F.softmax(scaled, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1).squeeze(1)
            logp = torch.log(probs.gather(1, next_tok[:, None]).squeeze(1) + 1e-12)
            ent = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)

        sum_logprob = sum_logprob + logp.mean()
        sum_entropy = sum_entropy + ent.mean()

        tok_int = int(next_tok.item())
        sampled.append(tok_int)
        ids = torch.cat([ids, next_tok[:, None]], dim=1)

    return sampled, sum_logprob, sum_entropy


def reinforce_finetune_btc(
    *,
    checkpoint_in: str,
    checkpoint_out: str,
    btc_csv: str,
    steps: int = 200,
    batch_episodes: int = 8,
    context_len: int = 128,
    horizon: int = 32,
    learning_rate: float = 1e-5,
    temperature: float = 1.0,
    top_k: int | None = 50,
    entropy_coef: float = 0.0,
    reward_type: RewardType = "token_acc",
    device: str = "auto",
) -> None:
    dev = resolve_device(device)

    # Load policy model
    model, cfg, dev, ckpt = load_checkpoint(checkpoint_in, device=str(dev))
    model.train()

    # Load tokenizer for BTC tokenization + optional decoded-reward
    tok = load_tokenizer(device=str(dev))

    btc = asset_tokens_from_csv(csv_path=btc_csv, tokenizer=tok, device=str(dev), normalize=True)
    tokens = btc.tokens
    if len(tokens) < context_len + horizon + 2:
        raise ValueError(f"BTC token stream too short: {len(tokens)} tokens")

    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.01)

    print("REINFORCE fine-tune")
    print(f"  reward_type: {reward_type}")
    print(f"  btc_tokens:  {len(tokens)}")
    print(f"  context_len: {context_len}")
    print(f"  horizon:     {horizon}")
    print(f"  steps:       {steps}")
    print(f"  batch_eps:   {batch_episodes}")
    print(f"  lr:          {learning_rate}")

    for step in range(1, steps + 1):
        episode_rewards: list[float] = []
        episode_logps: list[torch.Tensor] = []
        episode_ents: list[torch.Tensor] = []

        for _ in range(batch_episodes):
            start = random.randint(0, len(tokens) - (context_len + horizon) - 1)
            context = tokens[start : start + context_len]
            target = tokens[start + context_len : start + context_len + horizon]

            sampled, sum_logp, sum_ent = _sample_one_episode(
                model=model,
                context=context,
                target=target,
                horizon=horizon,
                temperature=temperature,
                top_k=top_k,
                device=dev,
            )

            if reward_type == "token_acc":
                r = _reward_token_accuracy(sampled, target)
            else:
                r = _reward_return_mse(sampled=sampled, target=target, tokenizer=tok, device=dev)

            episode_rewards.append(float(r))
            episode_logps.append(sum_logp)
            episode_ents.append(sum_ent)

        rewards = torch.tensor(episode_rewards, dtype=torch.float32, device=dev)
        baseline = rewards.mean()
        advantage = (rewards - baseline).detach()

        logps = torch.stack(episode_logps, dim=0)
        ents = torch.stack(episode_ents, dim=0)

        # Loss: maximize reward => minimize negative advantage * logprob
        loss_pg = -(advantage * logps).mean()
        loss_ent = -(entropy_coef * ents.mean()) if entropy_coef != 0.0 else torch.zeros((), device=dev)
        loss = loss_pg + loss_ent

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 10 == 0 or step == 1:
            r_mean = float(rewards.mean().item())
            r_std = float(rewards.std(unbiased=False).item())
            print(
                f"step {step:4d} | reward mean {r_mean:.4f} std {r_std:.4f} | loss {float(loss.item()):.4f}"
            )

    # Save final checkpoint
    save_checkpoint(
        checkpoint_out,
        model=model,
        optimizer=opt,
        epoch=int(ckpt.get("epoch", 0)),
        train_loss=float("nan"),
        val_loss=float("nan"),
    )
    print(f"Saved RL-finetuned checkpoint to: {checkpoint_out}")


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Stage 3: REINFORCE fine-tuning on BTC ground truth")
    p.add_argument("--in", dest="checkpoint_in", default="checkpoints/best_model.pt", help="Input checkpoint")
    p.add_argument("--out", dest="checkpoint_out", default="checkpoints/rl_btc.pt", help="Output checkpoint")
    p.add_argument(
        "--csv",
        "--btc-csv",
        dest="csv",
        default="data/BTC-USD_2014-09-17_2025-12-31_1d.csv",
        help="Asset CSV in repo format",
    )

    p.add_argument("--device", default="auto")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-episodes", type=int, default=8)
    p.add_argument("--context-len", type=int, default=128)
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--entropy-coef", type=float, default=0.0)
    p.add_argument("--reward", choices=["token_acc", "return_mse"], default="token_acc")

    args = p.parse_args()

    reinforce_finetune_btc(
        checkpoint_in=args.checkpoint_in,
        checkpoint_out=args.checkpoint_out,
        btc_csv=args.csv,
        steps=args.steps,
        batch_episodes=args.batch_episodes,
        context_len=args.context_len,
        horizon=args.horizon,
        learning_rate=args.lr,
        temperature=args.temp,
        top_k=args.top_k,
        entropy_coef=args.entropy_coef,
        reward_type=args.reward,
        device=args.device,
    )


if __name__ == "__main__":
    main()
