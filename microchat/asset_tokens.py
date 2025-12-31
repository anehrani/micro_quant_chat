from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch

# Import both tokenizer types
from src.tokenizer import (
    VQVAETwitterizerOC, 
    OHLCCandleTokenizer,
    consecutive_log_returns,
    extract_ohlc_features,
)

logger = logging.getLogger(__name__)

# Module exports
__all__ = [
    "TokenizeResult",
    "TokenizerType",
    "load_tokenizer",
    "asset_tokens_from_csv",
    "btc_tokens_from_csv",
]

# Default hyperparameters
DEFAULT_BETA = 0.25
DEFAULT_EMA_DECAY_LEGACY = 0.95
DEFAULT_EMA_DECAY_OHLC = 0.99
DEFAULT_NUM_LAYERS = 2
DEFAULT_HIDDEN_DIM = 64
DEFAULT_EMB_DIM = 32
DEFAULT_NUM_CODES = 512
DEFAULT_INPUT_DIM = 5


@dataclass(frozen=True)
class TokenizeResult:
    tokens: list[int]
    log_returns: np.ndarray


# Type alias for both tokenizer types
TokenizerType = VQVAETwitterizerOC | OHLCCandleTokenizer


def _read_repo_csv_with_two_header_rows(csv_path: str | Path) -> pd.DataFrame:
    """Read repo CSV format that contains 2 header rows + a blank 'Date' row."""
    df = pd.read_csv(csv_path)

    # Repo CSVs typically look like:
    # row0: Price, Adj Close, Close, ...
    # row1: Ticker, <TICKER>, <TICKER>, ...
    # row2: Date, , , ,
    # then data rows.
    if len(df) >= 3 and str(df.iloc[1].get("Price", "")).strip() == "Ticker":
        df = df.iloc[3:].reset_index(drop=True)

    # Ensure numeric Close
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Close"]).reset_index(drop=True)
    return df


@lru_cache(maxsize=1)
def _compute_global_log_return_stats(data_dir: str = "data") -> dict[str, float] | None:
    """Compute mean/std of log-returns across all CSVs as a fallback normalization.

    This mirrors the best-effort fallback used in src/predict.py.
    Results are cached to avoid re-reading CSV files on every call.
    """
    p = Path(data_dir)
    if not p.exists():
        return None

    csv_files = sorted(p.glob("*.csv"))
    if not csv_files:
        return None

    all_r: list[np.ndarray] = []
    for f in csv_files:
        try:
            df = _read_repo_csv_with_two_header_rows(f)
            r = consecutive_log_returns(df, c="Close")
            if r.size:
                all_r.append(r.astype(np.float64))
        except Exception as e:
            logger.warning("Failed to process %s: %s", f, e)
            continue

    if not all_r:
        return None

    r = np.concatenate(all_r)
    r = r[np.isfinite(r)]
    if r.size < 10:
        return None

    return {"mean": float(r.mean()), "std": float(r.std() + 1e-8)}


def load_tokenizer(
    *,
    model_path: str | Path = "models/tokenizer_model.pt",
    device: str = "cpu",
) -> TokenizerType:
    """Load tokenizer model - supports both legacy and new OHLC tokenizer."""
    sd = torch.load(model_path, map_location="cpu", weights_only=False)
    
    # Check if it's the new format with config
    if isinstance(sd, dict) and "config" in sd:
        config = sd["config"]
        state_dict = sd["state_dict"]
        
        if config.get("type") == "ohlc_candle":
            # New OHLC Candle Tokenizer
            tok = OHLCCandleTokenizer(
                input_dim=config.get("input_dim", DEFAULT_INPUT_DIM),
                hidden_dim=config.get("hidden_dim", DEFAULT_HIDDEN_DIM),
                emb_dim=config.get("emb_dim", DEFAULT_EMB_DIM),
                num_codes=config.get("num_codes", DEFAULT_NUM_CODES),
                num_layers=config.get("num_layers", DEFAULT_NUM_LAYERS),
                use_context=config.get("use_context", True),
                beta=config.get("beta", DEFAULT_BETA),
                ema_decay=config.get("ema_decay", DEFAULT_EMA_DECAY_OHLC),
            )
            tok.load_state_dict(state_dict)
            
            # Load normalization stats if available
            if "normalization" in sd:
                norm = sd["normalization"]
                tok.set_normalization_stats(
                    torch.from_numpy(norm["mean"].astype(np.float32)),
                    torch.from_numpy(norm["std"].astype(np.float32))
                )
            
            tok.eval()
            return tok.to(device)
    
    # Legacy format - VQVAETwitterizerOC
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]

    # Infer tokenizer hyperparameters from the saved state dict.
    codebook_w = sd["quantizer.codebook.weight"]
    enc0_w = sd["encoder.net.0.weight"]
    enc2_w = sd["encoder.net.2.weight"]

    num_codes = int(codebook_w.shape[0])
    emb_dim = int(codebook_w.shape[1])
    hidden = int(enc0_w.shape[0])
    patch_size = int(enc2_w.shape[2])

    tok = VQVAETwitterizerOC(
        patch_size=patch_size,
        emb_dim=emb_dim,
        num_codes=num_codes,
        hidden=hidden,
        beta=DEFAULT_BETA,
        ema_decay=DEFAULT_EMA_DECAY_LEGACY,
    )

    tok.load_state_dict(sd)
    tok.eval()
    return tok.to(device)


def asset_tokens_from_csv(
    *,
    csv_path: str | Path,
    tokenizer: TokenizerType,
    device: str = "cpu",
    normalize: bool = True,
    stats_source: Literal["global"] = "global",
    precomputed_stats: dict[str, float] | None = None,
) -> TokenizeResult:
    """Convert an asset CSV to tokens.

    Supports both:
    - Legacy VQVAETwitterizerOC: uses log returns (1D)
    - New OHLCCandleTokenizer: uses OHLC features (5D)

    Args:
        csv_path: Path to the OHLC CSV file.
        tokenizer: Loaded tokenizer model.
        device: Device for computation ('cpu' or 'cuda').
        normalize: Whether to normalize log returns (legacy tokenizer only).
        stats_source: Source for normalization stats ('global').
        precomputed_stats: Pre-computed mean/std dict to avoid recomputation.

    Returns:
        TokenizeResult with tokens and log returns.
    """
    df = _read_repo_csv_with_two_header_rows(csv_path)
    
    # Check tokenizer type
    if isinstance(tokenizer, OHLCCandleTokenizer):
        # New OHLC tokenizer - extract 5 features per candle
        features = extract_ohlc_features(df, "Open", "High", "Low", "Close")
        x = torch.from_numpy(features.astype(np.float32)).unsqueeze(0).to(device)
        
        with torch.no_grad():
            token_ids = tokenizer.encode(x, normalize=True)  # (1, T)
        
        tokens = token_ids[0].detach().cpu().tolist()
        # Return log_returns for compatibility (use the body feature which is log(C/O))
        log_r = features[:, 1]  # body = log(C/O)
        return TokenizeResult(tokens=tokens, log_returns=log_r)
    
    else:
        # Legacy tokenizer - use log returns
        log_r = consecutive_log_returns(df, c="Close")

        if normalize:
            stats = precomputed_stats
            if stats is None and stats_source == "global":
                stats = _compute_global_log_return_stats("data")
            if stats is not None and "mean" in stats and "std" in stats:
                log_r = (log_r - float(stats["mean"])) / float(stats["std"])

        x = torch.from_numpy(log_r.astype(np.float32)).view(1, -1, 1).to(device)
        with torch.no_grad():
            token_ids = tokenizer.encode(x)  # (1, T')

        tokens = token_ids[0].detach().cpu().tolist()
        return TokenizeResult(tokens=tokens, log_returns=log_r)


def btc_tokens_from_csv(
    *,
    csv_path: str | Path,
    tokenizer: TokenizerType,
    device: str = "cpu",
    normalize: bool = True,
    stats_source: Literal["global"] = "global",
    precomputed_stats: dict[str, float] | None = None,
) -> TokenizeResult:
    """Backward-compatible alias for older BTC-focused callers."""
    return asset_tokens_from_csv(
        csv_path=csv_path,
        tokenizer=tokenizer,
        device=device,
        normalize=normalize,
        stats_source=stats_source,
        precomputed_stats=precomputed_stats,
    )
