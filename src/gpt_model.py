"""
GPT model for financial time series prediction.
Based on nanochat architecture with modern features:
- Rotary embeddings (no positional embeddings)
- QK normalization
- Group Query Attention (GQA) support
- RMSNorm (no learnable parameters)
- No bias in linear layers
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    """GPT model configuration"""
    sequence_len: int = 512  # context window
    vocab_size: int = 128  # token vocabulary size
    n_layer: int = 8  # number of transformer blocks
    n_head: int = 8  # number of query heads
    n_kv_head: int = 8  # number of key/value heads (for GQA)
    n_embd: int = 512  # embedding dimension
    dropout: float = 0.1  # dropout rate


def rms_norm(x: torch.Tensor) -> torch.Tensor:
    """RMSNorm: purely functional (no learnable parameters)"""
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to queries and keys"""
    assert x.ndim == 4  # (batch, seq_len, heads, head_dim)
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with GQA support and KV caching"""

    def __init__(self, config: GPTConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
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
        B, T, C = x.size()

        # Project and reshape for multi-head attention
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Apply rotary embeddings and normalization
        cos, sin = cos_sin
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = rms_norm(q)
        k = rms_norm(k)

        # Reshape for attention: (B, T, H, head_dim) -> (B, H, T, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Handle KV cache for inference
        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)

        Tq = q.size(2)
        Tk = k.size(2)

        # Compute attention
        enable_gqa = self.n_head != self.n_kv_head
        if kv_cache is None or Tq == Tk:
            # Training: standard causal attention
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)
        elif Tq == 1:
            # Inference: single token, attends to cache
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)
        else:
            # Inference: chunk of tokens, attend to cache + causal
            attn_mask = torch.zeros((Tq, Tk), dtype=torch.bool, device=q.device)
            prefix_len = Tk - Tq
            attn_mask[:, :prefix_len] = True
            attn_mask[:, prefix_len:] = torch.tril(
                torch.ones((Tq, Tq), dtype=torch.bool, device=q.device)
            )
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa)

        # Project back
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    """Feed-forward network with SwiGLU activation"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ReLU^2 activation: relu(x)^2
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    """Transformer block: attention + MLP with residual connections"""

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
        # Pre-norm: norm before attention
        x = x + self.attn(rms_norm(x), cos_sin, kv_cache)
        # Pre-norm: norm before MLP
        x = x + self.mlp(rms_norm(x))
        return x


class KVCache:
    """Key-value cache for efficient inference"""

    def __init__(self, batch_size: int, num_heads: int, seq_len: int, head_dim: int, num_layers: int):
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.pos = 0

        # Initialize cache for each layer
        self.cache = []
        for _ in range(num_layers):
            k = torch.zeros(batch_size, num_heads, seq_len, head_dim)
            v = torch.zeros(batch_size, num_heads, seq_len, head_dim)
            self.cache.append((k, v))

    def reset(self):
        """Reset cache and position"""
        self.pos = 0

    def get_pos(self) -> int:
        """Get current position in cache"""
        return self.pos

    def insert_kv(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Insert new k,v into cache and return full k,v"""
        cached_k, cached_v = self.cache[layer_idx]
        cached_k[:, :, self.pos : self.pos + k.size(2)] = k
        cached_v[:, :, self.pos : self.pos + v.size(2)] = v
        self.pos += k.size(2)
        return cached_k[:, :, : self.pos], cached_v[:, :, : self.pos]

    def to(self, device: torch.device):
        """Move cache to device"""
        self.cache = [(k.to(device), v.to(device)) for k, v in self.cache]
        return self


class GPT(nn.Module):
    """GPT model for sequence prediction"""

    def __init__(self, config: GPTConfig, pad_vocab_size_to: int = 64):
        super().__init__()
        self.config = config

        # Pad vocab size for better performance
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size}")

        # Transformer blocks
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, padded_vocab_size, bias=False)

        # Precompute rotary embeddings
        self.rotary_seq_len = config.sequence_len * 10  # Over-allocate
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _precompute_rotary_embeddings(
        self, seq_len: int, head_dim: int, base: float = 10000.0, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Precompute rotary embedding frequencies"""
        if device is None:
            device = self.transformer.wte.weight.device

        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()

        # Convert to bfloat16 and add batch/head dims
        cos = cos.bfloat16()[None, :, None, :]
        sin = sin.bfloat16()[None, :, None, :]
        return cos, sin

    def init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # https://arxiv.org/pdf/2310.17813
                fan_out = module.weight.size(0)
                fan_in = module.weight.size(1)
                std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
                nn.init.normal_(module.weight, mean=0.0, std=std)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=1.0)

        # Zero out projection weights
        nn.init.zeros_(self.lm_head.weight)
        for block in self.transformer.h:
            nn.init.zeros_(block.mlp.c_proj.weight)
            nn.init.zeros_(block.attn.c_proj.weight)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        """
        Forward pass
        Args:
            idx: (batch, seq_len) token indices
            targets: (batch, seq_len) target token indices (for training)
            kv_cache: KVCache object (for inference)
        Returns:
            loss (if targets provided) or logits (batch, seq_len, vocab_size)
        """
        B, T = idx.size()

        # Get rotary embeddings
        assert T <= self.cos.size(1), f"Sequence length {T} exceeds cache {self.cos.size(1)}"
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0 : T0 + T], self.sin[:, T0 : T0 + T]

        # Token embedding
        x = self.transformer.wte(idx)
        x = rms_norm(x)

        # Forward through transformer blocks
        for block in self.transformer.h:
            x = block(x, cos_sin, kv_cache)

        x = rms_norm(x)

        # Language model head
        softcap = 15.0
        logits = self.lm_head(x)
        logits = logits[..., : self.config.vocab_size]
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            # Compute loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return loss
        else:
            return logits

    @torch.inference_mode()
    def generate(
        self,
        tokens: list,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        seed: int = 42,
    ):
        """
        Generate tokens autoregressively
        Args:
            tokens: Starting token list
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            seed: Random seed
        Yields:
            Generated token IDs
        """
        assert isinstance(tokens, list)
        device = self.lm_head.weight.device
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)

        ids = torch.tensor([tokens], dtype=torch.long, device=device)

        for _ in range(max_new_tokens):
            logits = self.forward(ids)
            logits = logits[:, -1, :]

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)

            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
