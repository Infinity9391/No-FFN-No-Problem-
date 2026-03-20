"""Visualization module for attention analysis and experiment results."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from attnpure.utils import attention_entropy, effective_rank, ensure_dir

matplotlib.use("Agg")

FIGDIR = Path("figures")
DPI = 150
COLORS = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800", "#607D8B"]


def _save_fig(fig: plt.Figure, name: str, figdir: Path = FIGDIR) -> Path:
    ensure_dir(figdir)
    path = figdir / f"{name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_attention_heatmaps(
    model: torch.nn.Module,
    xs: torch.Tensor,
    ys: torch.Tensor,
    layer: int = 0,
    head: Optional[int] = None,
    sample_idx: int = 0,
    figdir: Path = FIGDIR,
) -> Path:
    """Plot attention weight heatmaps for a single forward pass.

    The theory predicts near-one-hot (sharp) column patterns corresponding
    to the interpolation selection mechanism (Theorem 3.1).
    """
    model.eval()
    with torch.no_grad():
        _ = model(xs, ys, store_weights=True)
    weights = model.get_attention_weights()

    key = f"layer_{layer}"
    if key not in weights:
        raise ValueError(f"No weights for {key}. Available: {list(weights.keys())}")

    attn = weights[key][sample_idx].cpu().numpy()
    n_heads = attn.shape[0]

    if head is not None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        im = ax.imshow(attn[head], cmap="Blues", aspect="auto", vmin=0, vmax=1)
        ax.set_title(f"Layer {layer}, Head {head}", fontsize=12)
        ax.set_xlabel("Key position")
        ax.set_ylabel("Query position")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        return _save_fig(fig, f"attn_heatmap_L{layer}_H{head}", figdir)

    cols = min(4, n_heads)
    rows = (n_heads + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    if n_heads == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for i in range(n_heads):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        im = ax.imshow(attn[i], cmap="Blues", aspect="auto", vmin=0, vmax=1)
        ax.set_title(f"Head {i}", fontsize=10)
        if c == 0:
            ax.set_ylabel("Query pos")
        if r == rows - 1:
            ax.set_xlabel("Key pos")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i in range(n_heads, rows * cols):
        r, c = divmod(i, cols)
        axes[r, c].set_visible(False)

    fig.suptitle(f"Attention Weights — Layer {layer}", fontsize=13, y=1.02)
    fig.tight_layout()
    return _save_fig(fig, f"attn_heatmap_L{layer}_all_heads", figdir)


def plot_interpolation_emergence(
    entropy_history: List[float],
    entropy_steps: List[int],
    figdir: Path = FIGDIR,
) -> Path:
    """Plot attention entropy over training showing emergence of sharp
    interpolation selection."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.plot(entropy_steps, entropy_history, color=COLORS[0], linewidth=2)
    ax.set_xlabel("Training Step", fontsize=11)
    ax.set_ylabel("Mean Attention Entropy (nats)", fontsize=11)
    ax.set_title("Interpolation Selection Emergence", fontsize=13)
    ax.grid(True, alpha=0.3)

    if len(entropy_history) > 1:
        ax.annotate(
            "Sharper attention\n(interpolation selection)",
            xy=(entropy_steps[-1], entropy_history[-1]),
            xytext=(entropy_steps[len(entropy_steps) // 2], max(entropy_history) * 0.8),
            arrowprops=dict(arrowstyle="->", color="gray"),
            fontsize=9, color="gray",
        )

    fig.tight_layout()
    return _save_fig(fig, "interpolation_emergence", figdir)


def plot_scaling_curves(
    results_dict: Dict[str, Any],
    figdir: Path = FIGDIR,
) -> Path:
    """Plot MSE vs p and MSE vs H. Expected: O(1/p) and O(1/H)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    p_vals = np.array(results_dict["p_values"], dtype=float)
    p_mse = np.array(results_dict["p_mse_mean"])
    p_std = np.array(results_dict["p_mse_std"])

    ax1.errorbar(p_vals, p_mse, yerr=p_std, fmt="o-", color=COLORS[0],
                 linewidth=2, capsize=4, label="Empirical MSE")
    ref = p_mse[0] * p_vals[0] / p_vals
    ax1.plot(p_vals, ref, "--", color="gray", alpha=0.6, label=r"$O(1/p)$ reference")
    ax1.set_xlabel("Number of In-Context Points (p)", fontsize=11)
    ax1.set_ylabel("MSE", fontsize=11)
    ax1.set_title("Scaling with Interpolation Points", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    h_vals = np.array(results_dict["h_values"], dtype=float)
    h_mse = np.array(results_dict["h_mse_mean"])
    h_std = np.array(results_dict["h_mse_std"])

    ax2.errorbar(h_vals, h_mse, yerr=h_std, fmt="s-", color=COLORS[1],
                 linewidth=2, capsize=4, label="Empirical MSE")
    ref_h = h_mse[0] * h_vals[0] / h_vals
    ax2.plot(h_vals, ref_h, "--", color="gray", alpha=0.6, label=r"$O(1/H)$ reference")
    ax2.set_xlabel("Number of Attention Heads (H)", fontsize=11)
    ax2.set_ylabel("MSE", fontsize=11)
    ax2.set_title("Scaling with Number of Heads", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Error Scaling — Theorems 3.1 & 3.2", fontsize=14, y=1.02)
    fig.tight_layout()
    return _save_fig(fig, "scaling_curves", figdir)


def plot_icl_comparison(
    results_dict: Dict[str, Dict[str, float]],
    figdir: Path = FIGDIR,
) -> Path:
    """Bar chart comparing attention-only vs standard transformer across tasks."""
    tasks = list(results_dict.keys())
    attn_mse = [results_dict[t]["attn_mse"] for t in tasks]
    attn_std = [results_dict[t].get("attn_std", 0) for t in tasks]
    std_mse = [results_dict[t]["std_mse"] for t in tasks]
    std_std = [results_dict[t].get("std_std", 0) for t in tasks]

    x = np.arange(len(tasks))
    width = 0.35

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.bar(x - width / 2, attn_mse, width, yerr=attn_std,
           label="Attention-Only", color=COLORS[0], capsize=4, alpha=0.85)
    ax.bar(x + width / 2, std_mse, width, yerr=std_std,
           label="Standard (w/ FFN)", color=COLORS[1], capsize=4, alpha=0.85)

    ax.set_xlabel("Task Family", fontsize=12)
    ax.set_ylabel("MSE", fontsize=12)
    ax.set_title("In-Context Learning: Attention-Only vs Standard Transformer", fontsize=13)
    ax.set_xticks(x)
    task_labels = [t.replace("_", " ").title() for t in tasks]
    ax.set_xticklabels(task_labels, rotation=15, ha="right")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    return _save_fig(fig, "icl_comparison", figdir)


def plot_attention_rank_analysis(
    model: torch.nn.Module,
    xs: torch.Tensor,
    ys: torch.Tensor,
    figdir: Path = FIGDIR,
) -> Path:
    """Plot effective rank of attention weight matrices."""
    model.eval()
    with torch.no_grad():
        _ = model(xs, ys, store_weights=True)
    weights = model.get_attention_weights()

    layer_names = []
    head_indices = []
    ranks = []

    for layer_key in sorted(weights.keys()):
        attn = weights[layer_key]
        n_heads = attn.shape[1]
        attn_mean = attn.mean(dim=0)
        for h in range(n_heads):
            r = effective_rank(attn_mean[h])
            layer_names.append(layer_key)
            head_indices.append(h)
            ranks.append(r)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    labels = [f"{ln.replace('layer_', 'L')}-H{hi}" for ln, hi in zip(layer_names, head_indices)]
    ax.bar(range(len(ranks)), ranks, color=COLORS[2], alpha=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Effective Rank", fontsize=11)
    ax.set_title("Effective Rank of Attention Matrices", fontsize=13)
    ax.grid(True, axis="y", alpha=0.3)

    if len(ranks) > 0:
        seq_len = list(weights.values())[0].shape[-1]
        ax.axhline(y=seq_len, color="red", linestyle="--", alpha=0.5,
                    label=f"Full rank ({seq_len})")
        ax.legend(fontsize=9)

    fig.tight_layout()
    return _save_fig(fig, "attention_rank_analysis", figdir)


def plot_entropy_distribution(
    model: torch.nn.Module,
    xs: torch.Tensor,
    ys: torch.Tensor,
    figdir: Path = FIGDIR,
) -> Path:
    """Histogram of per-position attention entropy across all layers/heads."""
    model.eval()
    with torch.no_grad():
        _ = model(xs, ys, store_weights=True)
    weights = model.get_attention_weights()

    all_entropies = []
    for attn in weights.values():
        ent = attention_entropy(attn)
        all_entropies.append(ent.cpu().numpy().flatten())

    all_entropies = np.concatenate(all_entropies)

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.hist(all_entropies, bins=50, color=COLORS[0], alpha=0.7, edgecolor="white")
    ax.axvline(x=np.median(all_entropies), color="red", linestyle="--",
               label=f"Median = {np.median(all_entropies):.2f}")
    ax.set_xlabel("Attention Entropy (nats)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Distribution of Attention Entropy", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return _save_fig(fig, "entropy_distribution", figdir)


def plot_training_curves(
    train_losses: List[float],
    eval_losses: List[float],
    eval_steps: List[int],
    log_every: int = 200,
    figdir: Path = FIGDIR,
    label_prefix: str = "",
) -> Path:
    """Plot training and evaluation loss curves."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    train_steps = [(i + 1) * log_every for i in range(len(train_losses))]
    ax.plot(train_steps, train_losses, alpha=0.5, color=COLORS[0], label="Train Loss")
    ax.plot(eval_steps, eval_losses, "o-", color=COLORS[1], linewidth=2, label="Eval MSE")

    ax.set_xlabel("Training Step", fontsize=11)
    ax.set_ylabel("MSE Loss", fontsize=11)
    title = "Training Curves"
    if label_prefix:
        title = f"{label_prefix} — {title}"
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    fig.tight_layout()
    name = "training_curves"
    if label_prefix:
        name = f"training_curves_{label_prefix.lower().replace(' ', '_')}"
    return _save_fig(fig, name, figdir)
