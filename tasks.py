"""ICL task generators for in-context learning experiments.

Each task family generates batches of ICL episodes. Per episode a new
random function is sampled, ensuring genuine in-context learning.
All tasks operate on compact domains with bounded outputs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Type

import torch


class ICLTask(ABC):
    """Abstract base class for an ICL task generator."""

    def __init__(
        self,
        d_x: int = 5,
        n_points: int = 20,
        x_range: float = 3.0,
        noise_std: float = 0.1,
    ) -> None:
        self.d_x = d_x
        self.n_points = n_points
        self.x_range = x_range
        self.noise_std = noise_std

    @abstractmethod
    def sample_batch(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ...

    def _sample_xs(self, batch_size: int, device: torch.device) -> torch.Tensor:
        total = self.n_points + 1
        return (
            torch.rand(batch_size, total, self.d_x, device=device) * 2 * self.x_range
            - self.x_range
        )

    def _add_noise(self, ys: torch.Tensor) -> torch.Tensor:
        if self.noise_std > 0:
            ys = ys + torch.randn_like(ys) * self.noise_std
        return ys

    def _clip_outputs(self, ys: torch.Tensor, clip_val: float = 10.0) -> torch.Tensor:
        return torch.clamp(ys, -clip_val, clip_val)

    def _mask_query_label(self, ys: torch.Tensor) -> torch.Tensor:
        ys_masked = ys.clone()
        ys_masked[:, -1, :] = 0.0
        return ys_masked


class LinearRegression(ICLTask):
    """y = w^T x + b + noise. w, b sampled per episode."""

    def sample_batch(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        xs = self._sample_xs(batch_size, device)
        w = torch.randn(batch_size, self.d_x, 1, device=device) * 0.5
        b = torch.randn(batch_size, 1, 1, device=device) * 0.5
        ys = torch.bmm(xs, w) + b
        ys = self._add_noise(ys)
        ys = self._clip_outputs(ys)
        ys_input = self._mask_query_label(ys)
        return xs, ys_input


class RidgeRegression(ICLTask):
    """y = w_ridge^T x + noise with L2 shrinkage."""

    def __init__(self, *args, ridge_alpha: float = 1.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ridge_alpha = ridge_alpha

    def sample_batch(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        xs = self._sample_xs(batch_size, device)
        w_true = torch.randn(batch_size, self.d_x, 1, device=device) * 0.5
        shrinkage = 1.0 / (1.0 + self.ridge_alpha / self.n_points)
        w_ridge = w_true * shrinkage
        b = torch.randn(batch_size, 1, 1, device=device) * 0.3
        ys = torch.bmm(xs, w_ridge) + b
        ys = self._add_noise(ys)
        ys = self._clip_outputs(ys)
        ys_input = self._mask_query_label(ys)
        return xs, ys_input


class NonlinearRegression(ICLTask):
    """y = ReLU(w^T x + b) + noise. Matches the truncated linear model (Theorem 3.1)."""

    def sample_batch(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        xs = self._sample_xs(batch_size, device)
        w = torch.randn(batch_size, self.d_x, 1, device=device) * 0.5
        b = torch.randn(batch_size, 1, 1, device=device) * 0.5
        ys = torch.relu(torch.bmm(xs, w) + b)
        ys = self._add_noise(ys)
        ys = self._clip_outputs(ys)
        ys_input = self._mask_query_label(ys)
        return xs, ys_input


class SineRegression(ICLTask):
    """y = a * sin(w^T x + phase) + noise. Smooth, continuous, bounded."""

    def sample_batch(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        xs = self._sample_xs(batch_size, device)
        w = torch.randn(batch_size, self.d_x, 1, device=device) * 0.3
        phase = torch.rand(batch_size, 1, 1, device=device) * 2 * 3.14159
        amplitude = torch.rand(batch_size, 1, 1, device=device) * 2.0 + 0.5
        linear = torch.bmm(xs, w) + phase
        ys = amplitude * torch.sin(linear)
        ys = self._add_noise(ys)
        ys = self._clip_outputs(ys)
        ys_input = self._mask_query_label(ys)
        return xs, ys_input


class PolynomialRegression(ICLTask):
    """y = sum of low-degree polynomial terms + noise."""

    def __init__(self, *args, max_degree: int = 3, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_degree = max_degree

    def sample_batch(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        xs = self._sample_xs(batch_size, device)
        B, S, D = xs.shape

        ys = torch.zeros(B, S, 1, device=device)
        for deg in range(1, self.max_degree + 1):
            coeff = torch.randn(B, D, 1, device=device) * (0.3 ** deg)
            xs_pow = xs ** deg
            ys = ys + torch.bmm(xs_pow, coeff)

        bias = torch.randn(B, 1, 1, device=device) * 0.3
        ys = ys + bias
        ys = self._add_noise(ys)
        ys = self._clip_outputs(ys)
        ys_input = self._mask_query_label(ys)
        return xs, ys_input


class TwoLayerNNRegression(ICLTask):
    """y = W2 @ ReLU(W1 @ x + b1) + b2 + noise.
    The paper's Lemma 3.1 says attention can approximate this."""

    def __init__(self, *args, hidden_dim: int = 16, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim

    def sample_batch(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        xs = self._sample_xs(batch_size, device)
        B, S, D = xs.shape
        H = self.hidden_dim

        W1 = torch.randn(B, H, D, device=device) * (0.5 / D**0.5)
        b1 = torch.randn(B, H, 1, device=device) * 0.3
        W2 = torch.randn(B, 1, H, device=device) * (0.5 / H**0.5)
        b2 = torch.randn(B, 1, 1, device=device) * 0.3

        hidden = torch.relu(torch.bmm(W1, xs.transpose(1, 2)) + b1)
        ys = torch.bmm(W2, hidden) + b2
        ys = ys.transpose(1, 2)

        ys = self._add_noise(ys)
        ys = self._clip_outputs(ys)
        ys_input = self._mask_query_label(ys)
        return xs, ys_input


TASK_REGISTRY: Dict[str, Type[ICLTask]] = {
    "linear": LinearRegression,
    "ridge": RidgeRegression,
    "nonlinear": NonlinearRegression,
    "sine": SineRegression,
    "polynomial": PolynomialRegression,
    "twolayer_nn": TwoLayerNNRegression,
}


def get_task(name: str, **kwargs) -> ICLTask:
    """Instantiate a task by name."""
    if name not in TASK_REGISTRY:
        raise ValueError(
            f"Unknown task '{name}'. Choose from: {list(TASK_REGISTRY.keys())}"
        )
    return TASK_REGISTRY[name](**kwargs)
