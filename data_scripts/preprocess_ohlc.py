#!/usr/bin/env python3
import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass
class TextCandleTokenizer:
    """
    Turns candles into a text-like token sequence using only Open/Close.

    fit(df)   learns bin edges from training data
    encode(df) returns token_ids and a whitespace-separated "text"
    decode(token_ids, open_prices) returns an approximate reconstructed close series
    """
    vocab_size: int = 256
    method: str = "quantile"   # "quantile" or "fixed"
    signal: str = "log_oc"     # "log_oc" or "pct_oc" or "delta"
    clip_sigma: float = 5.0    # for method="fixed"
    rolling_z: int | None = None  # e.g., 256 for rolling z-score; None disables
    eps: float = 1e-12

    bin_edges_: np.ndarray | None = None  # length = vocab_size + 1
    token_str_: list[str] | None = None   # length = vocab_size

    def _compute_signal(self, df: pd.DataFrame, o="open", c="close") -> pd.Series:
        O = pd.to_numeric(df[o], errors="coerce")
        C = pd.to_numeric(df[c], errors="coerce")

        if self.signal == "delta":
            s = C - O
        elif self.signal == "pct_oc":
            s = (C - O) / (O.clip(lower=self.eps))
        elif self.signal == "log_oc":
            s = np.log(C.clip(lower=self.eps) / O.clip(lower=self.eps))
        else:
            raise ValueError(f"Unknown signal={self.signal}")

        if self.rolling_z is not None:
            mu = s.rolling(self.rolling_z, min_periods=max(10, self.rolling_z // 4)).mean().shift(1)
            sd = s.rolling(self.rolling_z, min_periods=max(10, self.rolling_z // 4)).std(ddof=0).shift(1)
            s = (s - mu) / (sd + 1e-6)

        return s

    def fit(self, df: pd.DataFrame, o="open", c="close"):
        s = self._compute_signal(df, o=o, c=c).dropna().to_numpy()

        if self.method == "quantile":
            # vocab_size bins -> vocab_size+1 edges
            qs = np.linspace(0.0, 1.0, self.vocab_size + 1)
            edges = np.quantile(s, qs)
            # Ensure strictly increasing edges (handle duplicates)
            edges = np.unique(edges)
            if len(edges) < self.vocab_size + 1:
                # fall back: small jitter + re-quantize via linspace over min/max
                lo, hi = float(np.min(s)), float(np.max(s))
                edges = np.linspace(lo, hi, self.vocab_size + 1)
        elif self.method == "fixed":
            mu = float(np.mean(s))
            sd = float(np.std(s) + 1e-9)
            lo = mu - self.clip_sigma * sd
            hi = mu + self.clip_sigma * sd
            edges = np.linspace(lo, hi, self.vocab_size + 1)
        else:
            raise ValueError(f"Unknown method={self.method}")

        self.bin_edges_ = edges

        # Human-readable token strings (you can feed IDs to models, but text is convenient for inspection)
        self.token_str_ = [f"T{i:03d}" for i in range(self.vocab_size)]
        return self

    def encode(self, df: pd.DataFrame, o="open", c="close"):
        if self.bin_edges_ is None:
            raise RuntimeError("Call fit() first to learn bin edges.")

        s = self._compute_signal(df, o=o, c=c)
        # digitize: returns 1..len(edges)-1; subtract 1 -> 0..bins-1
        ids = np.digitize(s.to_numpy(), self.bin_edges_[1:-1], right=False)

        # handle NaNs from rolling stats (if enabled)
        valid = ~np.isnan(s.to_numpy())
        ids = ids.astype(np.int64)
        ids[~valid] = -1  # mark invalid

        # Make "text": skip invalids
        toks = [self.token_str_[i] for i in ids if i >= 0]
        text = " ".join(toks)
        return ids, text, s  # returning s is handy for debugging

    def decode(self, token_ids: np.ndarray, open_prices: np.ndarray):
        """
        Decode tokens back to an approximate close series.
        Uses the bin center as the reconstructed signal value.
        """
        if self.bin_edges_ is None:
            raise RuntimeError("Call fit() first.")

        token_ids = np.asarray(token_ids)
        O = np.asarray(open_prices)

        # bin centers
        centers = (self.bin_edges_[:-1] + self.bin_edges_[1:]) / 2.0

        # invalid ids -> NaN
        s_hat = np.full_like(token_ids, fill_value=np.nan, dtype=np.float64)
        mask = token_ids >= 0
        s_hat[mask] = centers[token_ids[mask]]

        if self.signal == "delta":
            C_hat = O + s_hat
        elif self.signal == "pct_oc":
            C_hat = O * (1.0 + s_hat)
        elif self.signal == "log_oc":
            C_hat = O * np.exp(s_hat)
        else:
            raise ValueError

        return C_hat, s_hat


# -----------------------
# Example usage
# -----------------------
if __name__ == "__main__":
    df = pd.DataFrame({
        "open":  [100, 101, 100, 102, 103],
        "close": [101,  99, 102, 101, 105],
    })

    tok = TextCandleTokenizer(
        vocab_size=32,
        method="quantile",
        signal="log_oc",     # recommended
        rolling_z=None,      # set e.g. 256 for rolling z-score
    ).fit(df)

    ids, text, signal = tok.encode(df)
    print("signal:", signal.to_list())
    print("ids:", ids)
    print("text:", text)

    C_hat, s_hat = tok.decode(ids, df["open"].to_numpy())
    print("recon_close:", C_hat)