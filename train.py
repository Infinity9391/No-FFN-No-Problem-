"""Training loop for in-context learning experiments."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from tqdm import tqdm

from attnpure.tasks import ICLTask
from attnpure.utils import (
    ensure_dir,
    get_device,
    mse_metric,
    r_squared,
    save_checkpoint,
    attention_entropy,
)


@dataclass
class TrainConfig:
    """Training hyper-parameters."""
    training_steps: int = 50_000
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 500
    eval_every: int = 1000
    eval_batches: int = 16
    checkpoint_every: int = 10_000
    checkpoint_dir: str = "checkpoints"
    log_every: int = 200


@dataclass
class TrainResult:
    """Container for training results."""
    train_losses: List[float] = field(default_factory=list)
    eval_losses: List[float] = field(default_factory=list)
    eval_r2s: List[float] = field(default_factory=list)
    eval_steps: List[int] = field(default_factory=list)
    final_eval_mse: float = 0.0
    final_eval_r2: float = 0.0
    total_time_s: float = 0.0
    entropy_history: List[float] = field(default_factory=list)
    entropy_steps: List[int] = field(default_factory=list)


def _warmup_cosine_schedule(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup followed by cosine annealing."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _sample_with_target(
    task: ICLTask, batch_size: int, device: torch.device
) -> tuple:
    """Sample a batch and return (xs, ys_input, ys_target).

    Uses a monkey-patch on _mask_query_label to capture the true target
    before it gets zeroed out.
    """
    original_mask = task._mask_query_label
    captured_target = {}

    def _capture_mask(ys: torch.Tensor) -> torch.Tensor:
        captured_target["target"] = ys[:, -1:, :].clone()
        return original_mask(ys)

    task._mask_query_label = _capture_mask  # type: ignore
    xs, ys_input = task.sample_batch(batch_size, device)
    task._mask_query_label = original_mask  # type: ignore

    ys_target = captured_target["target"]  # (B, 1, 1)
    return xs, ys_input, ys_target


def _evaluate_clean(
    model: nn.Module,
    task: ICLTask,
    device: torch.device,
    n_batches: int,
    batch_size: int,
) -> Dict[str, float]:
    """Evaluate model on freshly sampled episodes."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for _ in range(n_batches):
            xs, ys_input, ys_target = _sample_with_target(task, batch_size, device)
            preds = model(xs, ys_input)
            target = ys_target.squeeze(-1)
            if preds.dim() == 2 and target.dim() == 1:
                target = target.unsqueeze(-1)
            all_preds.append(preds)
            all_targets.append(target)

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    mse = mse_metric(all_preds, all_targets)
    r2 = r_squared(all_preds, all_targets)
    return {"mse": mse, "r2": r2}


def train(
    model: nn.Module,
    task: ICLTask,
    config: TrainConfig,
    device: Optional[torch.device] = None,
    track_entropy: bool = False,
) -> TrainResult:
    """Train a model on ICL episodes.

    Args:
        model: An AttentionOnlyTransformer or StandardTransformer.
        task: ICL task generator.
        config: Training hyper-parameters.
        device: Device to train on (auto-detected if None).
        track_entropy: If True, periodically compute attention weight entropy.

    Returns:
        TrainResult with training curves and final metrics.
    """
    if device is None:
        device = get_device()
    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = _warmup_cosine_schedule(optimizer, config.warmup_steps, config.training_steps)
    loss_fn = nn.MSELoss()

    result = TrainResult()
    ensure_dir(config.checkpoint_dir)

    start_time = time.time()
    running_loss = 0.0
    running_count = 0

    pbar = tqdm(range(1, config.training_steps + 1), desc="Training", ncols=100)
    for step in pbar:
        model.train()

        xs, ys_input, ys_target = _sample_with_target(task, config.batch_size, device)

        store = track_entropy and (step % config.eval_every == 0)
        preds = model(xs, ys_input, store_weights=store)
        target = ys_target.squeeze(-1)
        if preds.shape != target.shape:
            target = target.view_as(preds)

        loss = loss_fn(preds, target)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        running_count += 1

        if step % config.log_every == 0:
            avg_loss = running_loss / running_count
            result.train_losses.append(avg_loss)
            pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
            running_loss = 0.0
            running_count = 0

        if step % config.eval_every == 0:
            metrics = _evaluate_clean(
                model, task, device, config.eval_batches, config.batch_size
            )
            result.eval_losses.append(metrics["mse"])
            result.eval_r2s.append(metrics["r2"])
            result.eval_steps.append(step)

            if track_entropy and hasattr(model, "get_attention_weights"):
                weights = model.get_attention_weights()
                if weights:
                    entropies = []
                    for w in weights.values():
                        ent = attention_entropy(w)
                        entropies.append(ent.mean().item())
                    result.entropy_history.append(sum(entropies) / len(entropies))
                    result.entropy_steps.append(step)

        if step % config.checkpoint_every == 0:
            ckpt_path = Path(config.checkpoint_dir) / f"step_{step}.pt"
            save_checkpoint(model, optimizer, step, loss.item(), ckpt_path)

    final_metrics = _evaluate_clean(
        model, task, device, config.eval_batches * 4, config.batch_size
    )
    result.final_eval_mse = final_metrics["mse"]
    result.final_eval_r2 = final_metrics["r2"]
    result.total_time_s = time.time() - start_time

    final_path = Path(config.checkpoint_dir) / "final.pt"
    save_checkpoint(
        model, optimizer, config.training_steps, result.final_eval_mse, final_path,
        extra={"final_r2": result.final_eval_r2},
    )

    return result
