"""
OHLC Candle Tokenizer using VQ-VAE

Based on "Neural Discrete Representation Learning" (VQ-VAE paper):
https://arxiv.org/abs/1711.00937

This tokenizer converts each OHLC candle directly into a single discrete token.
Each row in a price CSV (Open, High, Low, Close) becomes one token.

Key features:
- Each candle = one token (no patching/grouping)
- Uses normalized OHLC features for stable training
- VQ-VAE with EMA codebook updates
- Supports both encoding and decoding
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, Dict, List


# =============================================================================
# OHLC Feature Extraction
# =============================================================================

def extract_ohlc_features(df: pd.DataFrame, 
                          o_col: str = "Open", 
                          h_col: str = "High", 
                          l_col: str = "Low", 
                          c_col: str = "Close",
                          eps: float = 1e-12) -> np.ndarray:
    """
    Extract normalized OHLC features for each candle.
    
    Features per candle (5 dimensions):
    1. log_return: log(C_t / C_{t-1}) - price movement from previous close
    2. body: log(C_t / O_t) - candle body direction
    3. upper_wick: log(H_t / max(O_t, C_t)) - upper shadow
    4. lower_wick: log(min(O_t, C_t) / L_t) - lower shadow  
    5. range: log(H_t / L_t) - total candle range
    
    Args:
        df: DataFrame with OHLC columns
        o_col, h_col, l_col, c_col: Column names
        eps: Small value for numerical stability
        
    Returns:
        np.ndarray of shape (T, 5) with normalized features per candle
    """
    # Extract and clip OHLC values
    O = pd.to_numeric(df[o_col], errors="coerce").to_numpy(dtype=np.float64)
    H = pd.to_numeric(df[h_col], errors="coerce").to_numpy(dtype=np.float64)
    L = pd.to_numeric(df[l_col], errors="coerce").to_numpy(dtype=np.float64)
    C = pd.to_numeric(df[c_col], errors="coerce").to_numpy(dtype=np.float64)
    
    O = np.clip(O, eps, None)
    H = np.clip(H, eps, None)
    L = np.clip(L, eps, None)
    C = np.clip(C, eps, None)
    
    T = len(O)
    
    # Feature 1: Log return from previous close (first candle uses O as reference)
    log_return = np.zeros(T, dtype=np.float64)
    log_return[0] = np.log(C[0] / O[0])  # First candle: use open as reference
    log_return[1:] = np.log(C[1:] / C[:-1])  # Subsequent: log(C_t / C_{t-1})
    
    # Feature 2: Candle body direction
    body = np.log(C / O)
    
    # Feature 3: Upper wick (high relative to body top)
    body_top = np.maximum(O, C)
    upper_wick = np.log(H / body_top)
    
    # Feature 4: Lower wick (body bottom relative to low)
    body_bottom = np.minimum(O, C)
    lower_wick = np.log(body_bottom / L)
    
    # Feature 5: Total candle range
    candle_range = np.log(H / L)
    
    # Stack features: (T, 5)
    features = np.stack([log_return, body, upper_wick, lower_wick, candle_range], axis=1)
    
    # Replace inf/nan with 0
    features = np.where(np.isfinite(features), features, 0.0)
    
    return features.astype(np.float32)


def ohlc_to_log_oc(df: pd.DataFrame, o="open", c="close", eps=1e-12) -> np.ndarray:
    """Legacy function: Calculate log(C/O) for each candle."""
    O = pd.to_numeric(df[o], errors="coerce").to_numpy(dtype=np.float64)
    C = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float64)

    O = np.clip(O, eps, None)
    C = np.clip(C, eps, None)

    x = np.log(C / O)  # shape (T,)
    x = np.where(np.isfinite(x), x, np.nan)
    return x


def consecutive_log_returns(df: pd.DataFrame, c="Close", eps=1e-12) -> np.ndarray:
    """Calculate log returns from consecutive close prices: log(C_t+1 / C_t)"""
    C = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float64)
    C = np.clip(C, eps, None)
    
    log_returns = np.diff(np.log(C))  # shape (T-1,)
    log_returns = np.where(np.isfinite(log_returns), log_returns, 0.0)
    return log_returns


def pad_to_multiple(x: torch.Tensor, multiple: int, dim: int = 1):
    size = x.size(dim)
    pad_len = (multiple - (size % multiple)) % multiple
    if pad_len == 0:
        return x, 0
    pad_shape = list(x.shape)
    pad_shape[dim] = pad_len
    pad = torch.zeros(*pad_shape, device=x.device, dtype=x.dtype)
    return torch.cat([x, pad], dim=dim), pad_len


# =============================================================================
# VQ-VAE Components
# =============================================================================

class CandleEncoder(nn.Module):
    """
    Encodes each OHLC candle (5 features) into a latent embedding.
    Each candle maps to one embedding vector independently.
    
    Input: (B, T, 5) - batch of candle sequences with 5 features each
    Output: (B, T, D) - embedding per candle
    """
    def __init__(self, input_dim: int = 5, hidden_dim: int = 64, emb_dim: int = 32, 
                 num_layers: int = 2, use_context: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.use_context = use_context
        
        # MLP encoder for each candle's features
        layers = [nn.Linear(input_dim, hidden_dim), nn.GELU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        layers.append(nn.Linear(hidden_dim, emb_dim))
        self.mlp = nn.Sequential(*layers)
        
        # Optional: 1D conv for local context (looks at neighboring candles)
        if use_context:
            self.context_conv = nn.Sequential(
                nn.Conv1d(emb_dim, emb_dim, kernel_size=3, padding=1, groups=1),
                nn.GELU(),
                nn.Conv1d(emb_dim, emb_dim, kernel_size=1),
            )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, input_dim)
        returns: (B, T, emb_dim)
        """
        # Per-candle MLP encoding
        z = self.mlp(x)  # (B, T, D)
        
        # Add local context
        if self.use_context:
            z_conv = z.transpose(1, 2)  # (B, D, T)
            z_conv = self.context_conv(z_conv)  # (B, D, T)
            z = z + z_conv.transpose(1, 2)  # (B, T, D) residual connection
        
        return z


class CandleDecoder(nn.Module):
    """
    Decodes embeddings back to OHLC features.
    
    Input: (B, T, D) - embeddings
    Output: (B, T, 5) - reconstructed candle features
    """
    def __init__(self, output_dim: int = 5, hidden_dim: int = 64, emb_dim: int = 32,
                 num_layers: int = 2, use_context: bool = True):
        super().__init__()
        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.use_context = use_context
        
        # Optional: 1D conv for local context
        if use_context:
            self.context_conv = nn.Sequential(
                nn.Conv1d(emb_dim, emb_dim, kernel_size=3, padding=1, groups=1),
                nn.GELU(),
                nn.Conv1d(emb_dim, emb_dim, kernel_size=1),
            )
        
        # MLP decoder for each candle
        layers = [nn.Linear(emb_dim, hidden_dim), nn.GELU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B, T, emb_dim)
        returns: (B, T, output_dim)
        """
        # Add local context
        if self.use_context:
            z_conv = z.transpose(1, 2)  # (B, D, T)
            z_conv = self.context_conv(z_conv)  # (B, D, T)
            z = z + z_conv.transpose(1, 2)  # (B, T, D) residual connection
        
        # Per-candle MLP decoding
        x_recon = self.mlp(z)  # (B, T, output_dim)
        return x_recon


class VectorQuantizerEMA(nn.Module):
    """
    VQ-VAE with EMA codebook updates (stable in practice).
    Outputs token ids + straight-through quantized latents.
    
    Based on: https://arxiv.org/abs/1711.00937
    """
    def __init__(self, num_codes: int = 512, code_dim: int = 32, 
                 beta: float = 0.25, decay: float = 0.99, eps: float = 1e-5):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.beta = beta
        self.decay = decay
        self.eps = eps

        self.codebook = nn.Embedding(num_codes, code_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / num_codes, 1.0 / num_codes)

        self.register_buffer("ema_cluster_size", torch.zeros(num_codes))
        self.register_buffer("ema_embed_sum", torch.zeros(num_codes, code_dim))

    @torch.no_grad()
    def _ema_update(self, ze_flat: torch.Tensor, indices: torch.Tensor):
        """Update codebook using exponential moving average."""
        K, D = self.num_codes, self.code_dim
        onehot = F.one_hot(indices, num_classes=K).type_as(ze_flat)  # (N, K)

        cluster_size = onehot.sum(dim=0)  # (K,)
        embed_sum = onehot.t() @ ze_flat  # (K, D)

        self.ema_cluster_size.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
        self.ema_embed_sum.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

        n = self.ema_cluster_size.sum()
        smoothed = (self.ema_cluster_size + self.eps) / (n + K * self.eps) * n
        new_codes = self.ema_embed_sum / smoothed.unsqueeze(1)
        self.codebook.weight.data.copy_(new_codes)

    def forward(self, ze: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        ze: (B, T, D) - encoder outputs
        returns: zq_st (B, T, D), token_ids (B, T), stats dict
        """
        B, T, D = ze.shape
        ze_flat = ze.reshape(-1, D)  # (N, D) where N = B * T

        codes = self.codebook.weight  # (K, D)
        
        # Compute distances to all codes
        dists = (
            (ze_flat ** 2).sum(1, keepdim=True)
            + (codes ** 2).sum(1).unsqueeze(0)
            - 2 * (ze_flat @ codes.t())
        )  # (N, K)

        indices = torch.argmin(dists, dim=1)  # (N,)
        token_ids = indices.view(B, T)  # (B, T)

        zq = self.codebook(indices).view(B, T, D)  # (B, T, D)

        if self.training:
            self._ema_update(ze_flat.detach(), indices.detach())

        # Commitment loss
        commit_loss = self.beta * F.mse_loss(ze, zq.detach())
        
        # Straight-through estimator
        zq_st = ze + (zq - ze).detach()

        # Perplexity (measure of codebook usage)
        onehot = F.one_hot(indices, num_classes=self.num_codes).float()
        avg_probs = onehot.mean(0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        stats = {"vq_commit_loss": commit_loss, "vq_perplexity": perplexity}
        return zq_st, token_ids, stats


# =============================================================================
# Main OHLC Candle Tokenizer
# =============================================================================

class OHLCCandleTokenizer(nn.Module):
    """
    VQ-VAE Tokenizer that converts each OHLC candle directly into one discrete token.
    
    Following "Neural Discrete Representation Learning" (VQ-VAE):
    https://arxiv.org/abs/1711.00937
    
    Each candle (Open, High, Low, Close) -> one token
    
    Architecture:
    - Encoder: MLP + optional local context conv
    - Vector Quantizer: EMA-updated codebook
    - Decoder: MLP + optional local context conv
    
    Args:
        input_dim: Number of input features per candle (default 5)
        hidden_dim: Hidden layer dimension
        emb_dim: Embedding/code dimension
        num_codes: Number of discrete codes (vocabulary size)
        num_layers: Number of MLP layers in encoder/decoder
        use_context: Whether to use conv layers for local context
        beta: Commitment loss weight
        ema_decay: EMA decay for codebook updates
    """
    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 64,
        emb_dim: int = 32,
        num_codes: int = 512,
        num_layers: int = 2,
        use_context: bool = True,
        beta: float = 0.25,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.num_codes = num_codes
        
        self.encoder = CandleEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            emb_dim=emb_dim,
            num_layers=num_layers,
            use_context=use_context,
        )
        
        self.quantizer = VectorQuantizerEMA(
            num_codes=num_codes,
            code_dim=emb_dim,
            beta=beta,
            decay=ema_decay,
        )
        
        self.decoder = CandleDecoder(
            output_dim=input_dim,
            hidden_dim=hidden_dim,
            emb_dim=emb_dim,
            num_layers=num_layers,
            use_context=use_context,
        )
        
        # Statistics for normalization
        self.register_buffer("feature_mean", torch.zeros(input_dim))
        self.register_buffer("feature_std", torch.ones(input_dim))

    def set_normalization_stats(self, mean: torch.Tensor, std: torch.Tensor):
        """Set normalization statistics from training data."""
        self.feature_mean.copy_(mean)
        self.feature_std.copy_(std)
    
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input features."""
        return (x - self.feature_mean) / (self.feature_std + 1e-8)
    
    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize features back to original scale."""
        return x * (self.feature_std + 1e-8) + self.feature_mean

    @torch.no_grad()
    def encode(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Encode OHLC features to token IDs.
        
        Args:
            x: (B, T, input_dim) - batch of candle feature sequences
            normalize: whether to apply normalization
            
        Returns:
            token_ids: (B, T) - one token per candle
        """
        if normalize:
            x = self.normalize(x)
        ze = self.encoder(x)  # (B, T, D)
        _, token_ids, _ = self.quantizer(ze)
        return token_ids

    @torch.no_grad()
    def decode(self, token_ids: torch.Tensor, denormalize: bool = True) -> torch.Tensor:
        """
        Decode token IDs back to OHLC features.
        
        Args:
            token_ids: (B, T) - token IDs
            denormalize: whether to denormalize output
            
        Returns:
            x_recon: (B, T, input_dim) - reconstructed features
        """
        zq = self.quantizer.codebook(token_ids)  # (B, T, D)
        x_recon = self.decoder(zq)  # (B, T, input_dim)
        if denormalize:
            x_recon = self.denormalize(x_recon)
        return x_recon

    @torch.no_grad()
    def encode_df(self, df: pd.DataFrame, 
                  o_col: str = "Open", h_col: str = "High",
                  l_col: str = "Low", c_col: str = "Close") -> torch.Tensor:
        """
        Convenience method to encode a DataFrame directly.
        
        Args:
            df: DataFrame with OHLC columns
            o_col, h_col, l_col, c_col: column names
            
        Returns:
            token_ids: (1, T) - token for each candle
        """
        features = extract_ohlc_features(df, o_col, h_col, l_col, c_col)
        x = torch.from_numpy(features).unsqueeze(0).to(next(self.parameters()).device)
        return self.encode(x)

    def forward(self, x: torch.Tensor, normalize: bool = True) -> Dict:
        """
        Forward pass with reconstruction.
        
        Args:
            x: (B, T, input_dim) - batch of candle feature sequences
            normalize: whether to apply normalization
            
        Returns:
            dict with x_recon, token_ids, losses
        """
        if normalize:
            x_norm = self.normalize(x)
        else:
            x_norm = x
            
        ze = self.encoder(x_norm)  # (B, T, D)
        zq_st, token_ids, vq_stats = self.quantizer(ze)
        x_hat = self.decoder(zq_st)  # (B, T, input_dim)

        recon_loss = F.mse_loss(x_hat, x_norm)
        loss_total = recon_loss + vq_stats["vq_commit_loss"]

        # Denormalize for output
        x_recon = self.denormalize(x_hat) if normalize else x_hat

        return {
            "x_recon": x_recon,
            "token_ids": token_ids,
            "loss_total": loss_total,
            "loss_recon": recon_loss,
            **vq_stats,
        }


# =============================================================================
# Legacy 1D Convolutional VQ-VAE (kept for backwards compatibility)
# =============================================================================

class ResBlock1D(nn.Module):
    """Residual block for 1D convolutions."""
    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv1d(ch, ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(ch, ch, 1),
        )

    def forward(self, x):
        return x + self.net(x)


class Encoder1D(nn.Module):
    """
    (B, 1, T) -> (B, D, T')
    with T' ~ T/patch_size
    """
    def __init__(self, hidden=128, emb_dim=64, patch_size=8, n_res=2):
        super().__init__()
        self.patch_size = patch_size
        self.net = nn.Sequential(
            nn.Conv1d(1, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden, kernel_size=patch_size, stride=patch_size, padding=0),  # patching
            *[ResBlock1D(hidden) for _ in range(n_res)],
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, emb_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class Decoder1D(nn.Module):
    """
    (B, D, T') -> (B, 1, T)
    """
    def __init__(self, hidden=128, emb_dim=64, patch_size=8, n_res=2):
        super().__init__()
        self.patch_size = patch_size
        self.net = nn.Sequential(
            nn.Conv1d(emb_dim, hidden, 3, padding=1),
            *[ResBlock1D(hidden) for _ in range(n_res)],
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(hidden, hidden, kernel_size=patch_size, stride=patch_size, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, 1, 3, padding=1),
        )

    def forward(self, zq):
        return self.net(zq)


class VectorQuantizerEMALegacy(nn.Module):
    """
    Legacy VQ-VAE with EMA codebook updates for 1D conv architecture.
    Kept for backwards compatibility with VQVAETwitterizerOC.
    """
    def __init__(self, num_codes=512, code_dim=64, beta=0.25, decay=0.99, eps=1e-5):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.beta = beta
        self.decay = decay
        self.eps = eps

        self.codebook = nn.Embedding(num_codes, code_dim)
        nn.init.normal_(self.codebook.weight, mean=0.0, std=0.02)

        self.register_buffer("ema_cluster_size", torch.zeros(num_codes))
        self.register_buffer("ema_embed_sum", torch.zeros(num_codes, code_dim))

    @torch.no_grad()
    def _ema_update(self, ze_flat: torch.Tensor, indices: torch.Tensor):
        K, D = self.num_codes, self.code_dim
        onehot = F.one_hot(indices, num_classes=K).type_as(ze_flat)

        cluster_size = onehot.sum(dim=0)
        embed_sum = onehot.t() @ ze_flat

        self.ema_cluster_size.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
        self.ema_embed_sum.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

        n = self.ema_cluster_size.sum()
        smoothed = (self.ema_cluster_size + self.eps) / (n + K * self.eps) * n
        new_codes = self.ema_embed_sum / smoothed.unsqueeze(1)
        self.codebook.weight.data.copy_(new_codes)

    def forward(self, ze: torch.Tensor):
        """
        ze: (B, D, T') - Conv1D format
        returns: zq_st (B, D, T'), token_ids (B, T'), stats dict
        """
        B, D, Tp = ze.shape
        ze_flat = ze.permute(0, 2, 1).contiguous().view(-1, D)

        codes = self.codebook.weight
        dists = (
            (ze_flat**2).sum(1, keepdim=True)
            + (codes**2).sum(1).unsqueeze(0)
            - 2 * (ze_flat @ codes.t())
        )

        indices = torch.argmin(dists, dim=1)
        token_ids = indices.view(B, Tp)

        zq = self.codebook(indices).view(B, Tp, D).permute(0, 2, 1).contiguous()

        if self.training:
            self._ema_update(ze_flat.detach(), indices.detach())

        commit_loss = self.beta * F.mse_loss(ze, zq.detach())
        zq_st = ze + (zq - ze).detach()

        onehot = F.one_hot(indices, num_classes=self.num_codes).float()
        avg_probs = onehot.mean(0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        stats = {"vq_commit_loss": commit_loss, "vq_perplexity": perplexity}
        return zq_st, token_ids, stats


class VQVAETwitterizerOC(nn.Module):
    """
    Learnable tokenizer for x_t = log(C_t/O_t) as a single-channel time series.

    x: (B,T,1) -> token_ids: (B,T') where each token represents patch_size candles.
    """
    def __init__(
        self,
        patch_size=8,
        emb_dim=64,
        num_codes=512,
        hidden=128,
        beta=0.25,
        ema_decay=0.99
    ):
        super().__init__()
        self.patch_size = patch_size
        self.encoder = Encoder1D(hidden=hidden, emb_dim=emb_dim, patch_size=patch_size)
        self.quantizer = VectorQuantizerEMALegacy(num_codes=num_codes, code_dim=emb_dim, beta=beta, decay=ema_decay)
        self.decoder = Decoder1D(hidden=hidden, emb_dim=emb_dim, patch_size=patch_size)

        self.num_codes = num_codes

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,T,1) float
        returns token_ids: (B,T')
        """
        x_pad, _ = pad_to_multiple(x, self.patch_size, dim=1)
        x_ch = x_pad.transpose(1, 2).contiguous()     # (B,1,T)
        ze = self.encoder(x_ch)                        # (B,D,T')
        _, token_ids, _ = self.quantizer(ze)
        return token_ids

    @torch.no_grad()
    def decode(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Decode token IDs back to reconstructed values.
        token_ids: (B, T')
        returns: reconstructed values (B, T, 1)
        """
        # Get quantized latents from token IDs
        zq = self.quantizer.codebook(token_ids)  # (B, T', D)
        
        # Upsample back to original length
        zq = zq.permute(0, 2, 1).contiguous()  # (B, D, T')
        x_recon = self.decoder(zq)  # (B, 1, T)
        x_recon = x_recon.transpose(1, 2).contiguous()  # (B, T, 1)
        
        return x_recon
        """
        Returns a whitespace-separated string like "T012 T087 ..."
        (for B=1)
        """
        token_ids = self.encode(x)  # (B,T')
        if token_ids.size(0) != 1:
            raise ValueError("encode_sentence expects batch size B=1")
        ids = token_ids[0].tolist()
        width = max(3, len(str(self.num_codes - 1)))
        return " ".join([f"T{i:0{width}d}" for i in ids])

    def forward(self, x: torch.Tensor):
        """
        x: (B,T,1)
        returns recon + losses
        """
        orig_T = x.size(1)
        x_pad, _ = pad_to_multiple(x, self.patch_size, dim=1)

        x_ch = x_pad.transpose(1, 2).contiguous()     # (B,1,T)
        ze = self.encoder(x_ch)                        # (B,D,T')
        zq_st, token_ids, vq_stats = self.quantizer(ze)
        x_hat_ch = self.decoder(zq_st)                 # (B,1,T)
        x_hat = x_hat_ch.transpose(1, 2).contiguous()  # (B,T,1)

        recon_loss = F.mse_loss(x_hat, x_pad)
        loss_total = recon_loss + vq_stats["vq_commit_loss"]

        return {
            "x_recon": x_hat[:, :orig_T, :],
            "token_ids": token_ids,
            "loss_total": loss_total,
            "loss_recon": recon_loss,
            **vq_stats,
        }


class WindowDataset(Dataset):
    """
    Turns a long 1D series into overlapping windows for training.
    Legacy dataset for VQVAETwitterizerOC.
    """
    def __init__(self, x: np.ndarray, seq_len: int = 512, stride: int = 128):
        x = x[np.isfinite(x)]
        self.x = x.astype(np.float32)
        self.seq_len = seq_len
        self.stride = stride
        self.starts = list(range(0, max(1, len(self.x) - seq_len + 1), stride))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s = self.starts[idx]
        w = self.x[s:s + self.seq_len]
        return torch.from_numpy(w).unsqueeze(-1)


class OHLCCandleDataset(Dataset):
    """
    Dataset for OHLC candle tokenizer training.
    Creates overlapping windows of OHLC feature sequences.
    
    Each sample is a window of candle features: (seq_len, 5)
    """
    def __init__(self, features: np.ndarray, seq_len: int = 256, stride: int = 64):
        """
        Args:
            features: np.ndarray of shape (T, 5) with OHLC features per candle
            seq_len: Number of candles per training window
            stride: Step size between windows
        """
        # Remove rows with any non-finite values
        valid_mask = np.all(np.isfinite(features), axis=1)
        self.features = features[valid_mask].astype(np.float32)
        self.seq_len = seq_len
        self.stride = stride
        self.starts = list(range(0, max(1, len(self.features) - seq_len + 1), stride))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s = self.starts[idx]
        window = self.features[s:s + self.seq_len]  # (seq_len, 5)
        return torch.from_numpy(window)


def train_ohlc_tokenizer(
    tokenizer: OHLCCandleTokenizer,
    loader: DataLoader,
    device: str = "cuda",
    epochs: int = 50,
    lr: float = 1e-3,
    warmup_epochs: int = 5,
) -> OHLCCandleTokenizer:
    """
    Train the OHLC candle tokenizer.
    
    Args:
        tokenizer: OHLCCandleTokenizer instance
        loader: DataLoader with OHLCCandleDataset
        device: Device to train on
        epochs: Number of training epochs
        lr: Learning rate
        warmup_epochs: Epochs for learning rate warmup
        
    Returns:
        Trained tokenizer
    """
    tokenizer.to(device)
    opt = torch.optim.AdamW(tokenizer.parameters(), lr=lr, weight_decay=0.01)
    
    # Cosine annealing scheduler with warmup
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # Initialize codebook with data embeddings
    print("Initializing codebook from data...")
    all_ze = []
    tokenizer.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            x_norm = tokenizer.normalize(batch)
            ze = tokenizer.encoder(x_norm)  # (B, T, D)
            all_ze.append(ze.reshape(-1, ze.size(-1)))
            if len(all_ze) * all_ze[0].size(0) > tokenizer.num_codes * 10:
                break
    
    all_ze = torch.cat(all_ze, dim=0)
    if len(all_ze) >= tokenizer.num_codes:
        indices = torch.randperm(all_ze.size(0))[:tokenizer.num_codes]
        tokenizer.quantizer.codebook.weight.data.copy_(all_ze[indices])
        print(f"Initialized codebook with {tokenizer.num_codes} codes from data")
    else:
        print(f"Warning: Not enough data to initialize all {tokenizer.num_codes} codes")

    tokenizer.train()
    best_loss = float('inf')
    
    for ep in range(1, epochs + 1):
        total_loss = 0.0
        total_recon = 0.0
        total_perp = 0.0
        n_batches = 0
        
        for batch in loader:
            batch = batch.to(device)  # (B, T, 5)
            out = tokenizer(batch, normalize=True)

            loss = out["loss_total"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(tokenizer.parameters(), max_norm=1.0)
            opt.step()

            total_loss += float(loss.detach().cpu())
            total_recon += float(out["loss_recon"].detach().cpu())
            total_perp += float(out["vq_perplexity"].detach().cpu())
            n_batches += 1
        
        scheduler.step()

        avg_loss = total_loss / max(1, n_batches)
        avg_recon = total_recon / max(1, n_batches)
        avg_perp = total_perp / max(1, n_batches)
        current_lr = scheduler.get_last_lr()[0]
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_marker = " *"
        else:
            best_marker = ""
        
        print(f"epoch {ep:03d} | loss={avg_loss:.6f} | recon={avg_recon:.6f} | "
              f"perplexity={avg_perp:.1f} | lr={current_lr:.2e}{best_marker}")

    tokenizer.eval()
    return tokenizer


def train_tokenizer(tokenizer, loader, device="cuda", epochs=5, lr=2e-4):
    """Legacy training function for VQVAETwitterizerOC."""
    tokenizer.to(device)
    opt = torch.optim.AdamW(tokenizer.parameters(), lr=lr)

    # Initialize codebook with data
    print("Initializing codebook from data...")
    all_ze = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            x_pad, _ = pad_to_multiple(batch, tokenizer.patch_size, dim=1)
            x_ch = x_pad.transpose(1, 2).contiguous()
            ze = tokenizer.encoder(x_ch)
            all_ze.append(ze.permute(0, 2, 1).contiguous().view(-1, ze.size(1)))
            if len(all_ze) * all_ze[0].size(0) > tokenizer.num_codes * 10:
                break
    
    all_ze = torch.cat(all_ze, dim=0)
    indices = torch.randperm(all_ze.size(0))[:tokenizer.num_codes]
    tokenizer.quantizer.codebook.weight.data.copy_(all_ze[indices])
    print(f"Initialized codebook with {tokenizer.num_codes} codes from data")

    tokenizer.train()
    for ep in range(1, epochs + 1):
        total = 0.0
        perp_total = 0.0
        for batch in loader:
            batch = batch.to(device)  # (B,T,1)
            out = tokenizer(batch)

            loss = out["loss_total"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total += float(loss.detach().cpu())
            perp_total += float(out["vq_perplexity"].detach().cpu())

        avg = total / max(1, len(loader))
        avg_perp = perp_total / max(1, len(loader))
        print(f"epoch {ep:02d} | loss={avg:.6f} | perplexity={avg_perp:.2f}")

    tokenizer.eval()
    return tokenizer


# ---- Example usage: OHLC Candle Tokenizer ----
if __name__ == "__main__":
    import glob
    import os
    import argparse
    
    parser = argparse.ArgumentParser(description="Train OHLC Candle Tokenizer (VQ-VAE)")
    parser.add_argument("--legacy", action="store_true", help="Use legacy 1D conv tokenizer")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--num_codes", type=int, default=512, help="Vocabulary size")
    parser.add_argument("--emb_dim", type=int, default=32, help="Embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden layer dimension")
    parser.add_argument("--seq_len", type=int, default=256, help="Sequence length for training")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()
    
    # Process all CSV files in data folder
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    print(f"{'='*60}")
    print(f"OHLC Candle Tokenizer (VQ-VAE)")
    print(f"Based on: Neural Discrete Representation Learning")
    print(f"Paper: https://arxiv.org/abs/1711.00937")
    print(f"{'='*60}")
    print(f"\nFound {len(csv_files)} CSV files in data folder")
    
    if args.legacy:
        # Legacy mode: use 1D conv tokenizer with log returns
        print("\n[Legacy Mode] Using 1D convolutional tokenizer...")
        all_x = []
        for csv_file in csv_files:
            ticker = os.path.basename(csv_file).split("_")[0]
            print(f"  Loading {ticker}...")
            df = pd.read_csv(csv_file)
            x = consecutive_log_returns(df, c="Close")
            all_x.append(x)
        
        combined_x = np.concatenate([x[np.isfinite(x)] for x in all_x])
        print(f"Total data points: {len(combined_x)}")
        
        mean = combined_x.mean()
        std = combined_x.std()
        combined_x = (combined_x - mean) / (std + 1e-8)
        print(f"Data normalized: mean={mean:.6f}, std={std:.6f}")
        
        ds = WindowDataset(combined_x, seq_len=args.seq_len, stride=64)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)
        print(f"Dataset windows: {len(ds)}")

        tokenizer = VQVAETwitterizerOC(
            patch_size=2,
            emb_dim=args.emb_dim,
            num_codes=args.num_codes,
            hidden=args.hidden_dim,
            beta=0.25,
            ema_decay=0.95,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\nTraining on {device}")
        tokenizer = train_tokenizer(tokenizer, dl, device=device, epochs=args.epochs, lr=args.lr)

    else:
        # New mode: OHLC Candle Tokenizer (1 candle = 1 token)
        print("\n[New Mode] Using OHLC Candle Tokenizer (1 candle = 1 token)...")
        
        # Extract OHLC features from all CSV files
        all_features = []
        for csv_file in csv_files:
            ticker = os.path.basename(csv_file).split("_")[0]
            print(f"  Loading {ticker}...")
            
            # Read CSV - handle Yahoo Finance multi-header format
            # Format: Row 0: Price, Adj Close, Close, High, Low, Open, Volume
            #         Row 1: Ticker names
            #         Row 2: Date header
            #         Row 3+: Data
            df = pd.read_csv(csv_file, header=0, skiprows=[1, 2])
            
            # Debug: print columns
            # print(f"    Columns: {list(df.columns)}")
            
            # Try different column name formats
            cols_upper = ["Open", "High", "Low", "Close"]
            cols_lower = ["open", "high", "low", "close"]
            
            if all(c in df.columns for c in cols_upper):
                features = extract_ohlc_features(df, "Open", "High", "Low", "Close")
            elif all(c in df.columns for c in cols_lower):
                features = extract_ohlc_features(df, "open", "high", "low", "close")
            else:
                print(f"    Warning: Could not find OHLC columns in {ticker}")
                print(f"    Available columns: {list(df.columns)}")
                continue
            
            all_features.append(features)
            print(f"    -> {len(features)} candles, features shape: {features.shape}")
        
        # Concatenate all features
        combined_features = np.concatenate(all_features, axis=0)
        print(f"\nTotal candles: {len(combined_features)}")
        print(f"Features per candle: {combined_features.shape[1]}")
        
        # Compute normalization statistics
        valid_mask = np.all(np.isfinite(combined_features), axis=1)
        valid_features = combined_features[valid_mask]
        
        feature_mean = valid_features.mean(axis=0)
        feature_std = valid_features.std(axis=0)
        
        print(f"\nFeature statistics:")
        feature_names = ["log_return", "body", "upper_wick", "lower_wick", "range"]
        for i, name in enumerate(feature_names):
            print(f"  {name:12s}: mean={feature_mean[i]:+.6f}, std={feature_std[i]:.6f}")
        
        # Create dataset and dataloader
        ds = OHLCCandleDataset(combined_features, seq_len=args.seq_len, stride=64)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)
        print(f"\nDataset windows: {len(ds)}")

        # Initialize tokenizer
        tokenizer = OHLCCandleTokenizer(
            input_dim=5,
            hidden_dim=args.hidden_dim,
            emb_dim=args.emb_dim,
            num_codes=args.num_codes,
            num_layers=2,
            use_context=True,
            beta=0.25,
            ema_decay=0.99,
        )
        
        # Set normalization statistics
        tokenizer.set_normalization_stats(
            torch.from_numpy(feature_mean.astype(np.float32)),
            torch.from_numpy(feature_std.astype(np.float32))
        )
        
        print(f"\nTokenizer config:")
        print(f"  Input dim: {tokenizer.input_dim}")
        print(f"  Embedding dim: {tokenizer.emb_dim}")
        print(f"  Vocabulary size: {tokenizer.num_codes}")
        print(f"  Parameters: {sum(p.numel() for p in tokenizer.parameters()):,}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\nTraining on {device}")
        tokenizer = train_ohlc_tokenizer(
            tokenizer, dl, 
            device=device, 
            epochs=args.epochs, 
            lr=args.lr,
            warmup_epochs=5
        )

    # Save model
    models_dir = os.path.join(data_dir, "..", "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "tokenizer_model.pt")
    
    # Save with metadata
    save_dict = {
        "state_dict": tokenizer.state_dict(),
        "config": {
            "type": "legacy" if args.legacy else "ohlc_candle",
            "num_codes": args.num_codes,
            "emb_dim": args.emb_dim,
            "hidden_dim": args.hidden_dim,
        }
    }
    if not args.legacy:
        save_dict["config"]["input_dim"] = 5
        save_dict["normalization"] = {
            "mean": tokenizer.feature_mean.cpu().numpy(),
            "std": tokenizer.feature_std.cpu().numpy(),
        }
    
    torch.save(save_dict, model_path)
    print(f"\nModel saved to {model_path}")

    # Tokenize all data
    print("\n" + "="*60)
    print("Tokenizing all candles...")
    tokenizer.eval()
    
    if args.legacy:
        x_clean = combined_x[np.isfinite(combined_x)]
        x_tensor = torch.from_numpy(x_clean.astype(np.float32)).unsqueeze(0).unsqueeze(-1)
        x_tensor = x_tensor.to(device)
        token_ids = tokenizer.encode(x_tensor)
        token_ids_np = token_ids[0].cpu().numpy().astype(np.int32)
        total_candles = len(x_clean)
    else:
        # Process in batches to avoid memory issues
        batch_size = 10000
        all_token_ids = []
        
        x_tensor = torch.from_numpy(valid_features.astype(np.float32)).unsqueeze(0)
        x_tensor = x_tensor.to(device)
        
        with torch.no_grad():
            token_ids = tokenizer.encode(x_tensor)
        
        token_ids_np = token_ids[0].cpu().numpy().astype(np.int32)
        total_candles = len(valid_features)
    
    # Statistics
    unique_tokens = len(np.unique(token_ids_np))
    total_tokens = len(token_ids_np)
    compression_ratio = total_candles / total_tokens if args.legacy else 1.0
    
    print(f"\nTokenization complete:")
    print(f"  Total candles: {total_candles:,}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Unique tokens used: {unique_tokens} / {args.num_codes}")
    print(f"  Codebook utilization: {100 * unique_tokens / args.num_codes:.1f}%")
    if args.legacy:
        print(f"  Compression ratio: {compression_ratio:.1f}x")
    else:
        print(f"  Compression ratio: 1:1 (one token per candle)")
    
    # Save token IDs
    tokens_file = os.path.join(data_dir, "all_tokens.txt")
    np.savetxt(tokens_file, token_ids_np, fmt='%d', delimiter=' ', newline=' ')
    print(f"\nSaved token IDs to {tokens_file}")
    print(f"File size: {os.path.getsize(tokens_file) / 1024:.1f} KB")
    
    # Show example tokens
    print(f"\nFirst 20 tokens: {' '.join(map(str, token_ids_np[:20]))}")
    print(f"Last 20 tokens:  {' '.join(map(str, token_ids_np[-20:]))}")
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)