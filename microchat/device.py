from __future__ import annotations

import torch


def resolve_device(device: str) -> torch.device:
    """Resolve device string ('auto'|'cpu'|'cuda') to torch.device."""
    device = (device or "auto").lower()
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)
