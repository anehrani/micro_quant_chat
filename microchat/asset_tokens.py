from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch

# Reuse the repo's existing tokenizer implementation.
from src.tokenizer import VQVAETwitterizerOC, consecutive_log_returns


@dataclass(frozen=True)
class TokenizeResult:
    tokens: list[int]
    log_returns: np.ndarray


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


def _compute_global_log_return_stats(data_dir: str | Path = "data") -> dict[str, float] | None:
    """Compute mean/std of log-returns across all CSVs as a fallback normalization.

    This mirrors the best-effort fallback used in src/predict.py.
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
        except Exception:
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
) -> VQVAETwitterizerOC:
    # Infer tokenizer hyperparameters from the saved state dict.
    sd = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]

    # Required keys (from src/tokenizer.py modules)
    # - quantizer.codebook.weight: (num_codes, emb_dim)
    # - encoder.net.0.weight: (hidden, 1, 3)
    # - encoder.net.2.weight: (hidden, hidden, patch_size)
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
        beta=0.25,
        ema_decay=0.95,
    )

    tok.load_state_dict(sd)
    tok.eval()
    return tok.to(device)


def asset_tokens_from_csv(
    *,
    csv_path: str | Path,
    tokenizer: VQVAETwitterizerOC,
    device: str = "cpu",
    normalize: bool = True,
    stats_source: Literal["global"] = "global",
) -> TokenizeResult:
    """Convert an asset CSV close prices -> log returns -> tokens.

    Reward-learning uses tokens for discrete generation.

    normalize:
      If True, normalize log returns by mean/std estimated from the repo's CSVs.
      (This is consistent with earlier usage in src/predict.py.)
    """
    df = _read_repo_csv_with_two_header_rows(csv_path)

    log_r = consecutive_log_returns(df, c="Close")

    if normalize:
        stats = None
        if stats_source == "global":
            stats = _compute_global_log_return_stats("data")
        if stats is not None and "mean" in stats and "std" in stats:
            log_r = (log_r - float(stats["mean"])) / float(stats["std"])  # type: ignore[assignment]

    x = torch.from_numpy(log_r.astype(np.float32)).view(1, -1, 1).to(device)
    with torch.no_grad():
        token_ids = tokenizer.encode(x)  # (1, T')

    tokens = token_ids[0].detach().cpu().tolist()
    return TokenizeResult(tokens=tokens, log_returns=log_r)


def btc_tokens_from_csv(
    *,
    csv_path: str | Path,
    tokenizer: VQVAETwitterizerOC,
    device: str = "cpu",
    normalize: bool = True,
    stats_source: Literal["global"] = "global",
) -> TokenizeResult:
    """Backward-compatible alias for older BTC-focused callers."""
    return asset_tokens_from_csv(
        csv_path=csv_path,
        tokenizer=tokenizer,
        device=device,
        normalize=normalize,
        stats_source=stats_source,
    )
