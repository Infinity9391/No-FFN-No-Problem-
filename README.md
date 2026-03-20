# AttentionPure

> **Empirical validation that softmax attention alone — without FFN/MLP blocks — is a universal in-context learner.**

This project is based on  [Universal Approximation with Softmax Attention (Hu et al., 2025)](https://arxiv.org/abs/2504.15956) through controlled experiments comparing attention-only transformers against standard transformers across multiple in-context learning task families.

---

## Motivation

Standard transformers interleave two components: multi-head self-attention (MHA) and feed-forward network (FFN/MLP) blocks. Ussually both are thought to be necessary attention for token mixing and then the FFNs for computation. Is this REALLY necessary? 

Hu et al. (2025) prove a surprising result: **softmax attention is not just a mixing mechanism  it is inherently a piecewise linear approximator**. The key insight is that softmax acts as a near-argmax selector over learned "anchor" points, enabling attention layers alone to approximate any continuous function without FFN blocks.

The paper establishes three core theorems:

| Theorem | Result | Implication |
|---|---|---|
| Theorem 3.1 | Single-head attention approximates generalized ReLUs with error O(1/p) | More context points = better approximation |
| Theorem 3.2 | Multi-head attention achieves O(1/H) error scaling | More heads = proportionally better |
| Theorem 3.3 & 4.1 | Two-layer attention-only transformers are universal sequence-to-sequence approximators | FFN blocks are unnecessary for ICL |

Goal of this project is  building attention only transformers from scratch and comparing them head-to-head against standard transformers on in context learning tasks.

---

## How It Works

### The In-Context Learning Setup

Rather than training on a fixed dataset, models learn to learn from context. Each training batch is a fresh regression episode:

```
Episode = [(x₁, y₁), (x₂, y₂), ..., (x₂₀, y₂₀), (x_query, ?)]
                                                        ↑
                                             Model must predict this
```

1. **Sample a random function** — e.g., random weights for a linear function, random frequency for a sine wave
2. **Generate 20 labeled examples** — inputs drawn uniformly from [−3, 3]⁵, labels computed from the random function + small noise
3. **Mask the query label** — the model sees the context pairs and must predict the query output

Since the function changes every batch, the model cannot memorize — it must perform genuine in-context learning by examining the examples and generalizing.

### Two Models, Fair Comparison

| | Attention-Only | Standard (w/ FFN) |
|---|---|---|
| **Architecture** | MHA → LayerNorm (×2 layers) | MHA → FFN → LayerNorm (×2 layers) |
| **FFN blocks** | None | GELU, hidden dim 256 |
| **Parameters** | Matched via scaled d_model | Baseline |
| **Training** | 5,000 steps, Adam, cosine LR | Identical |

The attention-only model's `d_model` is scaled by √(12/5) ≈ 1.55× so both models have roughly the same parameter count so its fair 

### Task Families

All tasks generate continuous functions on compact domains, matching the paper's theoretical assumptions:

| Task | Function Class | Paper Reference |
|---|---|---|
| Linear | y = wᵀx + b | Theorem 4.1 |
| Nonlinear (ReLU) | y = ReLU(wᵀx + b) | Theorem 3.1 |
| Sine | y = a · sin(wᵀx + φ) | Theorem 3.3 |
| Two-Layer NN | y = W₂ · ReLU(W₁x + b₁) + b₂ | Lemma 3.1 |

---

## Results

### Attention-Only Matches or Beats Standard Transformers

<p align="center">
  <img src="figures/icl_comparison.png" alt="ICL Comparison" width="700"/>
</p>

The attention only model outperforms the standard transformer on every single task:

| Task | Attn-Only MSE | Attn-Only R² | Standard MSE | Standard R² | MSE Reduction |
|---|---|---|---|---|---|
| Linear | 0.0719 | 0.9817 | 0.1678 | 0.9572 | **57%** |
| Nonlinear | 0.1881 | 0.8589 | 0.2085 | 0.8437 | **10%** |
| Sine | 0.4044 | 0.6850 | 0.4273 | 0.6672 | **5%** |
| Two-Layer NN | 0.0332 | 0.8352 | 0.0356 | 0.8235 | **7%** |

The attention only model achieves lower MSE and higher R² across all tasks, confirming that FFN blocks are not necessary for in-context learning.

The **linear task shows the most dramatic gap**   the attention only model cuts error by more than half. This aligns with Theorem 4.1, which proves attention can implement in-context gradient descent on convex losses.

### Training Dynamics

<p align="center">
  <img src="figures/training_curves.png" alt="Training Curves" width="700"/>
</p>

- **Left:** Training loss and eval MSE both converge smoothly over 5,000 steps
- **Right:** R² climbs from ~0.3 to ~0.88, showing the model progressively learns to regress from context

The attention only model trains stably with just **46,721 parameters** and converges in ~32 seconds on a CUDA GPU.

---

## Getting Started

### Installation

```bash
git clone https://github.com/Infinity9391/AttentionPure.git
cd AttentionPure
pip install -e .
```

**Requirements:** Python 3.9+, PyTorch 2.0+, NumPy, Matplotlib, tqdm, SciPy

### Quick Benchmark (~5 min on T4 GPU)

```bash
python -m experiments.run_icl_benchmark --quick
```

### Full Benchmark (50k steps per model)

```bash
python -m experiments.run_icl_benchmark
```

### Scaling Experiments

```bash
# Validate O(1/p) and O(1/H) scaling laws
python -m experiments.run_scaling --quick
```

### Interpolation Analysis

```bash
# Visualize attention weight structure after training
python -m experiments.run_interpolation_analysis \
    --checkpoint checkpoints/demo_linear/final.pt \
    --task nonlinear
```

---

## Repository Structure

```
AttentionPure/
├── attnpure/
│   ├── models.py          # Attention-only & standard transformer implementations
│   ├── tasks.py           # 6 ICL task generators (random function families)
│   ├── train.py           # Training loop with evaluation and checkpointing
│   ├── visualize.py       # Publication-quality plotting utilities
│   └── utils.py           # Seeding, metrics, device setup
├── experiments/
│   ├── configs.py                     # Hyperparameter configurations
│   ├── run_icl_benchmark.py           # Main ICL performance comparison
│   ├── run_interpolation_analysis.py  # Attention weight interpretability
│   └── run_scaling.py                 # O(1/p) and O(1/H) scaling ablation
├── figures/                           # Generated plots
├── requirements.txt
└── setup.py
```

---

## Key Takeaways

- **FFN blocks are not necessary for in-context learning.** Attention only transformers match or exceed standard transformers across linear, nonlinear, sinusoidal, and neural network regression tasks.

- **Attention is more than mixing.** Softmax attention functions as an inherent piecewise linear approximator, using a near-one-hot interpolation selection mechanism to approximate arbitrary continuous functions.

- **The extra parameters from FFN blocks may not be necessary.** The standard transformer consistently underperforms despite having the same parameter budget  my guess is that  the FFN parameters may make optimization harder without adding useful expressivity for ICL.

---



## License

MIT
