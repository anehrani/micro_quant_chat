import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

def ohlc_to_log_oc(df: pd.DataFrame, o="open", c="close", eps=1e-12) -> np.ndarray:
    O = pd.to_numeric(df[o], errors="coerce").to_numpy(dtype=np.float64)
    C = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float64)

    O = np.clip(O, eps, None)
    C = np.clip(C, eps, None)

    x = np.log(C / O)  # shape (T,)
    # Replace inf with nan then drop nans
    x = np.where(np.isfinite(x), x, np.nan)
    return x


def consecutive_log_returns(df: pd.DataFrame, c="Close", eps=1e-12) -> np.ndarray:
    """Calculate log returns from consecutive close prices: log(C_t+1 / C_t)"""
    C = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float64)
    C = np.clip(C, eps, None)
    
    # Calculate log returns: log(C_t+1 / C_t)
    log_returns = np.diff(np.log(C))  # shape (T-1,)
    
    # Replace inf/nan with 0
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


class ResBlock1D(nn.Module):
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


class VectorQuantizerEMA(nn.Module):
    """
    VQ-VAE with EMA codebook updates (stable in practice).
    Outputs token ids + straight-through quantized latents.
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
        onehot = F.one_hot(indices, num_classes=K).type_as(ze_flat)   # (N,K)

        cluster_size = onehot.sum(dim=0)             # (K,)
        embed_sum = onehot.t() @ ze_flat             # (K,D)

        self.ema_cluster_size.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
        self.ema_embed_sum.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

        n = self.ema_cluster_size.sum()
        smoothed = (self.ema_cluster_size + self.eps) / (n + K * self.eps) * n
        new_codes = self.ema_embed_sum / smoothed.unsqueeze(1)
        self.codebook.weight.data.copy_(new_codes)

    def forward(self, ze: torch.Tensor):
        """
        ze: (B,D,T')
        returns: zq_st (B,D,T'), token_ids (B,T'), stats dict
        """
        B, D, Tp = ze.shape
        ze_flat = ze.permute(0, 2, 1).contiguous().view(-1, D)  # (N,D)

        codes = self.codebook.weight                               # (K,D)
        dists = (
            (ze_flat**2).sum(1, keepdim=True)
            + (codes**2).sum(1).unsqueeze(0)
            - 2 * (ze_flat @ codes.t())
        )                                                          # (N,K)

        indices = torch.argmin(dists, dim=1)                        # (N,)
        token_ids = indices.view(B, Tp)                             # (B,T')

        zq = self.codebook(indices).view(B, Tp, D).permute(0, 2, 1).contiguous()  # (B,D,T')

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
        self.quantizer = VectorQuantizerEMA(num_codes=num_codes, code_dim=emb_dim, beta=beta, decay=ema_decay)
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
        w = self.x[s:s + self.seq_len]                     # (T,)
        return torch.from_numpy(w).unsqueeze(-1)           # (T,1)


def train_tokenizer(tokenizer, loader, device="cuda", epochs=5, lr=2e-4):
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


# ---- Example usage ----
if __name__ == "__main__":
    import glob
    import os
    
    # Process all CSV files in data folder
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    print(f"Found {len(csv_files)} CSV files in data folder")
    
    # Combine all data for training
    all_x = []
    for csv_file in csv_files:
        ticker = os.path.basename(csv_file).split("_")[0]
        print(f"Loading {ticker}...")
        df = pd.read_csv(csv_file)
        x = consecutive_log_returns(df, c="Close")  # Use consecutive close prices
        all_x.append(x)
    
    # Concatenate all series
    combined_x = np.concatenate([x[np.isfinite(x)] for x in all_x])
    print(f"Total data points: {len(combined_x)}")
    
    # Normalize data
    mean = combined_x.mean()
    std = combined_x.std()
    combined_x = (combined_x - mean) / (std + 1e-8)
    print(f"Data normalized: mean={mean:.6f}, std={std:.6f}")
    
    # Create dataset and dataloader with shorter windows for more variation
    ds = WindowDataset(combined_x, seq_len=256, stride=64)
    dl = DataLoader(ds, batch_size=16, shuffle=True, drop_last=True, num_workers=0)
    print(f"Dataset windows: {len(ds)}")

    # Initialize tokenizer
    tokenizer = VQVAETwitterizerOC(
        patch_size=2,    # 16 candles -> 1 token (larger patches for more variation)
        emb_dim=32,       # Smaller embedding dim
        num_codes=128,    # Smaller codebook
        hidden=64,        # Smaller hidden dim
        beta=0.25,
        ema_decay=0.95,
    )

    # Train
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")
    tokenizer = train_tokenizer(tokenizer, dl, device=device, epochs=50, lr=1e-3)

    # Save model
    models_dir = os.path.join(data_dir, "..", "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "tokenizer_model.pt")
    torch.save(tokenizer.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Encode the entire combined series as one continuous sequence
    print("\nTokenizing combined series...")
    tokenizer.eval()
    
    # Use the original combined_x (normalized) for tokenization
    x_clean = combined_x[np.isfinite(combined_x)]
    
    # Convert to tensor - process in one batch
    x_tensor = torch.from_numpy(x_clean.astype(np.float32)).unsqueeze(0).unsqueeze(-1)  # (1,T,1)
    x_tensor = x_tensor.to(device)
    
    # Encode the entire series
    token_ids = tokenizer.encode(x_tensor)  # (1,T')
    
    # Extract token IDs as numpy array
    token_ids_np = token_ids[0].cpu().numpy().astype(np.int32)
    
    # Count unique tokens
    unique_tokens = len(np.unique(token_ids_np))
    total_tokens = len(token_ids_np)
    print(f"Tokenized {len(x_clean)} data points into {total_tokens} tokens")
    print(f"Unique tokens: {unique_tokens} out of {tokenizer.num_codes}")
    
    # Save token IDs as text file (space-separated)
    tokens_file = os.path.join(data_dir, "all_tokens.txt")
    np.savetxt(tokens_file, token_ids_np, fmt='%d', delimiter=' ', newline=' ')
    print(f"Saved token IDs to {tokens_file}")
    print(f"File size: {os.path.getsize(tokens_file) / 1024:.1f} KB")
    
    print("Done!")