"""Nanochat-inspired GPT implementation for token streams."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    sequence_len: int = 256
    vocab_size: int = 512
    n_layer: int = 4
    n_head: int = 4
    n_kv_head: int = 4
    n_embd: int = 256
    dropout: float = 0.2


def rms_norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 4  # (B, T, H, D)
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head

        assert config.n_embd % config.n_head == 0
        assert config.n_kv_head <= config.n_head
        assert config.n_head % config.n_kv_head == 0

        self.c_q = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos_sin: Tuple[torch.Tensor, torch.Tensor],
        kv_cache: Optional["KVCache"] = None,
    ) -> torch.Tensor:
        B, T, _ = x.size()

        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        cos, sin = cos_sin
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = rms_norm(q)
        k = rms_norm(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)

        Tq = q.size(2)
        Tk = k.size(2)

        enable_gqa = self.n_head != self.n_kv_head
        if kv_cache is None or Tq == Tk:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)
        elif Tq == 1:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)
        else:
            attn_mask = torch.zeros((Tq, Tk), dtype=torch.bool, device=q.device)
            prefix_len = Tk - Tq
            attn_mask[:, :prefix_len] = True
            attn_mask[:, prefix_len:] = torch.tril(torch.ones((Tq, Tq), dtype=torch.bool, device=q.device))
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa)

        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.relu(x).square()
        return self.c_proj(x)


class Block(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        cos_sin: Tuple[torch.Tensor, torch.Tensor],
        kv_cache: Optional["KVCache"] = None,
    ) -> torch.Tensor:
        x = x + self.attn(rms_norm(x), cos_sin, kv_cache)
        x = x + self.mlp(rms_norm(x))
        return x


class KVCache:
    def __init__(self, batch_size: int, num_heads: int, seq_len: int, head_dim: int, num_layers: int):
        self.pos = 0
        self.cache = []
        for _ in range(num_layers):
            k = torch.zeros(batch_size, num_heads, seq_len, head_dim)
            v = torch.zeros(batch_size, num_heads, seq_len, head_dim)
            self.cache.append((k, v))

    def reset(self) -> None:
        self.pos = 0

    def insert_kv(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cached_k, cached_v = self.cache[layer_idx]
        cached_k[:, :, self.pos : self.pos + k.size(2)] = k
        cached_v[:, :, self.pos : self.pos + v.size(2)] = v
        self.pos += k.size(2)
        return cached_k[:, :, : self.pos], cached_v[:, :, : self.pos]

    def to(self, device: torch.device) -> "KVCache":
        self.cache = [(k.to(device), v.to(device)) for k, v in self.cache]
        return self


class GPT(nn.Module):
    def __init__(self, config: GPTConfig, pad_vocab_size_to: int = 64):
        super().__init__()
        self.config = config

        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to

        self.wte = nn.Embedding(padded_vocab_size, config.n_embd)
        self.h = nn.ModuleList([Block(config, i) for i in range(config.n_layer)])
        self.lm_head = nn.Linear(config.n_embd, padded_vocab_size, bias=False)

        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos: torch.Tensor
        self.sin: torch.Tensor
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _precompute_rotary_embeddings(
        self,
        seq_len: int,
        head_dim: int,
        base: float = 10000.0,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if device is None:
            device = self.wte.weight.device

        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos = freqs.cos().bfloat16()[None, :, None, :]
        sin = freqs.sin().bfloat16()[None, :, None, :]
        return cos, sin

    def init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                fan_out = module.weight.size(0)
                fan_in = module.weight.size(1)
                std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
                nn.init.normal_(module.weight, mean=0.0, std=std)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=1.0)

        nn.init.zeros_(self.lm_head.weight)
        for block in self.h:
            blk = cast(Block, block)
            nn.init.zeros_(blk.mlp.c_proj.weight)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        B, T = idx.size()
        if T > self.config.sequence_len:
            raise ValueError(f"Sequence length {T} exceeds model context {self.config.sequence_len}")

        x = self.wte(idx)
        cos = self.cos[:, :T]
        sin = self.sin[:, :T]
        cos_sin = (cos, sin)

        for block in self.h:
            x = block(x, cos_sin)

        x = rms_norm(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits

        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return loss

    @torch.no_grad()
    def generate(
        self,
        seed_tokens: list[int],
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 50,
    ):
        device = next(self.parameters()).device
        idx = torch.tensor([seed_tokens], dtype=torch.long, device=device)

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.sequence_len :]
            logits = self(idx_cond)
            logits = logits[:, -1, :]

            if temperature == 0:
                next_token = torch.argmax(logits, dim=-1)
            else:
                logits = logits / max(1e-8, temperature)
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("inf")
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)

            idx = torch.cat([idx, next_token[:, None]], dim=1)
            yield int(next_token.item())
