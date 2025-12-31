from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from .device import resolve_device
from .gpt import GPT, GPTConfig


def coerce_config(obj: Any) -> GPTConfig:
    if isinstance(obj, GPTConfig):
        return obj
    if isinstance(obj, dict):
        return GPTConfig(**obj)
    raise TypeError(f"Unsupported config type in checkpoint: {type(obj)}")


def save_checkpoint(
    path: str | Path,
    *,
    model: GPT,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    train_loss: float | None = None,
    val_loss: float | None = None,
) -> None:
    """Save a checkpoint in a non-pickled, portable format.

    Note: config is saved as a plain dict to avoid pickling issues.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "config": asdict(model.config),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if train_loss is not None:
        payload["train_loss"] = float(train_loss)
    if val_loss is not None:
        payload["val_loss"] = float(val_loss)

    torch.save(payload, p)


def load_checkpoint(
    path: str | Path,
    *,
    device: str = "auto",
) -> Tuple[GPT, GPTConfig, torch.device, Dict[str, Any]]:
    """Load checkpoint, supporting both legacy (pickled GPTConfig) and dict config."""
    dev = resolve_device(device)
    ckpt = torch.load(path, map_location=dev, weights_only=False)

    config = coerce_config(ckpt["config"])
    model = GPT(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(dev)
    model.eval()

    return model, config, dev, ckpt
