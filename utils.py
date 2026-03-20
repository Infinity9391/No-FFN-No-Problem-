"""Utility functions: seeding, metrics, device setup, and helpers."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> int:
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def mse_metric(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute mean squared error."""
    return torch.mean((predictions - targets) ** 2).item()


def r_squared(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute R^2 (coefficient of determination)."""
    predictions = predictions.detach().flatten()
    targets = targets.detach().flatten()
    ss_res = torch.sum((targets - predictions) ** 2)
    ss_tot = torch.sum((targets - targets.mean()) ** 2)
    if ss_tot < 1e-12:
        return 0.0
    return (1.0 - ss_res / ss_tot).item()


def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it does not exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    loss: float,
    path: str | Path,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a training checkpoint."""
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "loss": loss,
    }
    if extra is not None:
        state.update(extra)
    ensure_dir(Path(path).parent)
    torch.save(state, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Load a training checkpoint."""
    map_location = device if device is not None else get_device()
    state = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    return {"step": state.get("step", 0), "loss": state.get("loss", None)}


def attention_entropy(attn_weights: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Compute per-column entropy of attention weight matrices.

    Lower entropy = sharper (more one-hot-like) = interpolation selection.
    """
    log_attn = torch.log(attn_weights + eps)
    entropy = -torch.sum(attn_weights * log_attn, dim=-1)
    return entropy


def effective_rank(matrix: torch.Tensor) -> float:
    """Compute the effective rank of a matrix via singular value entropy."""
    s = torch.linalg.svdvals(matrix.float())
    s = s / (s.sum() + 1e-12)
    log_s = torch.log(s + 1e-12)
    entropy = -torch.sum(s * log_s)
    return torch.exp(entropy).item()
