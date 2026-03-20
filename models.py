"""Attention-only and standard transformer models for in-context learning."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """Standard multi-head self-attention with optional weight caching."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self._attn_weights: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, store_weights: bool = False) -> torch.Tensor:
        B, S, _ = x.shape
        qkv = self.W_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, S, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        if store_weights:
            self._attn_weights = attn.detach()

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, S, self.d_model)
        return self.out_proj(out)

    def get_cached_weights(self) -> Optional[torch.Tensor]:
        return self._attn_weights


class FeedForward(nn.Module):
    """Two-layer FFN with GELU activation."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AttentionOnlyBlock(nn.Module):
    """Single attention-only layer: MHA -> skip -> LayerNorm -> Linear projection."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, store_weights: bool = False) -> torch.Tensor:
        h = x + self.attn(x, store_weights=store_weights)
        h = self.norm(h)
        h = h + self.proj(h)
        return h


class AttentionOnlyTransformer(nn.Module):
    """Attention-only Transformer for in-context learning.

    Architecture per layer: Multi-Head Attention -> Skip -> LayerNorm -> Linear.
    No MLP / FFN blocks whatsoever.

    The paper (Hu et al., 2025) proves this architecture is a universal
    sequence-to-sequence approximator (Theorem 3.3) and can perform
    in-context gradient descent on convex losses (Theorem 4.1).
    """

    def __init__(
        self,
        d_x: int,
        d_model: int = 64,
        n_heads: int = 8,
        n_layers: int = 2,
        d_out: int = 1,
        max_seq_len: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_x = d_x
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_out = d_out

        self.input_proj = nn.Linear(d_x + 1, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        self.layers = nn.ModuleList([
            AttentionOnlyBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        self.output_norm = nn.LayerNorm(d_model)
        self.readout = nn.Linear(d_model, d_out)

    def forward(
        self,
        xs: torch.Tensor,
        ys: torch.Tensor,
        store_weights: bool = False,
    ) -> torch.Tensor:
        tokens = torch.cat([xs, ys], dim=-1)
        h = self.input_proj(tokens)

        B, S, _ = h.shape
        positions = torch.arange(S, device=h.device).unsqueeze(0)
        h = h + self.pos_embed(positions)
        h = self.input_norm(h)

        for layer in self.layers:
            h = layer(h, store_weights=store_weights)

        h = self.output_norm(h)
        query_repr = h[:, -1, :]
        return self.readout(query_repr)

    def get_attention_weights(self) -> Dict[str, torch.Tensor]:
        weights = {}
        for i, layer in enumerate(self.layers):
            w = layer.attn.get_cached_weights()
            if w is not None:
                weights[f"layer_{i}"] = w
        return weights


class StandardBlock(nn.Module):
    """Standard Transformer block: MHA -> skip -> LN -> FFN -> skip -> LN."""

    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, store_weights: bool = False) -> torch.Tensor:
        h = x + self.attn(x, store_weights=store_weights)
        h = self.norm1(h)
        h = h + self.ffn(h)
        h = self.norm2(h)
        return h


class StandardTransformer(nn.Module):
    """Standard Transformer (with FFN) baseline for in-context learning."""

    def __init__(
        self,
        d_x: int,
        d_model: int = 64,
        n_heads: int = 8,
        n_layers: int = 2,
        d_out: int = 1,
        max_seq_len: int = 128,
        d_ff: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_x = d_x
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_out = d_out

        self.input_proj = nn.Linear(d_x + 1, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        self.layers = nn.ModuleList([
            StandardBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.output_norm = nn.LayerNorm(d_model)
        self.readout = nn.Linear(d_model, d_out)

    def forward(
        self,
        xs: torch.Tensor,
        ys: torch.Tensor,
        store_weights: bool = False,
    ) -> torch.Tensor:
        tokens = torch.cat([xs, ys], dim=-1)
        h = self.input_proj(tokens)

        B, S, _ = h.shape
        positions = torch.arange(S, device=h.device).unsqueeze(0)
        h = h + self.pos_embed(positions)
        h = self.input_norm(h)

        for layer in self.layers:
            h = layer(h, store_weights=store_weights)

        h = self.output_norm(h)
        query_repr = h[:, -1, :]
        return self.readout(query_repr)

    def get_attention_weights(self) -> Dict[str, torch.Tensor]:
        weights = {}
        for i, layer in enumerate(self.layers):
            w = layer.attn.get_cached_weights()
            if w is not None:
                weights[f"layer_{i}"] = w
        return weights


def build_matched_models(
    d_x: int,
    d_model_attn: int = 64,
    n_heads: int = 8,
    n_layers: int = 2,
    d_out: int = 1,
    max_seq_len: int = 128,
    dropout: float = 0.0,
) -> Tuple[AttentionOnlyTransformer, StandardTransformer]:
    """Build attention-only and standard transformer with approximately
    matched parameter counts."""
    d_ff = 4 * d_model_attn
    scale_factor = math.sqrt(12.0 / 5.0)
    d_model_scaled = int(round(d_model_attn * scale_factor / n_heads)) * n_heads
    d_model_scaled = max(d_model_scaled, n_heads)

    attn_model = AttentionOnlyTransformer(
        d_x=d_x, d_model=d_model_scaled, n_heads=n_heads,
        n_layers=n_layers, d_out=d_out, max_seq_len=max_seq_len, dropout=dropout,
    )
    std_model = StandardTransformer(
        d_x=d_x, d_model=d_model_attn, n_heads=n_heads,
        n_layers=n_layers, d_out=d_out, max_seq_len=max_seq_len,
        d_ff=d_ff, dropout=dropout,
    )
    return attn_model, std_model
