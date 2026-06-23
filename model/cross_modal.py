"""
EdgeGuard: Cross-Modal Attention Module

Cross-Attention fusion layer where visual features serve as Query
and text features serve as Key/Value, enabling modality interaction.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class CrossModalAttention(nn.Module):
    """
    Cross-Modal Attention for visual-text feature fusion.

    Uses visual features as Query (Q) and text features as Key/Value (K/V),
    allowing visual tokens to attend to relevant text information.
    This enables the model to leverage semantic text context when
    processing visual features.

    Architecture:
        Q = visual_proj(x)          # (B, L_v, D)
        K = text_proj(y)            # (B, L_t, D)
        V = text_proj(y)            # (B, L_t, D)
        Attention = softmax(QK^T / sqrt(D)) V

    Args:
        query_dim: Dimension of query features (visual).
        key_value_dim: Dimension of key/value features (text).
        num_heads: Number of attention heads.
        dropout: Dropout probability.

    Example:
        >>> cross_attn = CrossModalAttention(query_dim=384, key_value_dim=384, num_heads=8)
        >>> visual = torch.randn(2, 16, 384)  # per-frame visual features
        >>> text = torch.randn(2, 1, 384)     # per-clip text feature
        >>> fused = cross_attn(visual, text)  # (2, 16, 384)
    """

    def __init__(
        self,
        query_dim: int = 384,
        key_value_dim: int = 384,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        assert query_dim % num_heads == 0, "query_dim must be divisible by num_heads"
        assert key_value_dim % num_heads == 0, "key_value_dim must be divisible by num_heads"

        self.query_dim = query_dim
        self.key_value_dim = key_value_dim
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.query_proj = nn.Linear(query_dim, query_dim)
        self.key_proj = nn.Linear(key_value_dim, query_dim)
        self.value_proj = nn.Linear(key_value_dim, query_dim)

        self.output_proj = nn.Sequential(
            nn.Linear(query_dim, query_dim),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(query_dim)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Apply cross-modal attention.

        Args:
            query: Visual features of shape (B, L_v, query_dim).
            key_value: Text features of shape (B, L_t, key_value_dim).
            mask: Optional attention mask of shape (B, L_v, L_t).

        Returns:
            Fused features of shape (B, L_v, query_dim) with residual connection applied.
        """
        residual = query

        B, L_v, _ = query.shape
        L_t = key_value.shape[1]

        Q = self.query_proj(query)
        K = self.key_proj(key_value)
        V = self.value_proj(key_value)

        Q = Q.view(B, L_v, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, L_t, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, L_t, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(B, L_v, self.query_dim)

        output = self.output_proj(attn_output)
        output = self.layer_norm(output + residual)

        return output


class CrossModalPooler(nn.Module):
    """
    Pools cross-attended features across the temporal dimension.

    Combines mean pooling and [CLS]-like pooling for robust representation.

    Args:
        feature_dim: Input feature dimension.
    """

    def __init__(self, feature_dim: int = 384) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.weight = nn.Parameter(torch.ones(feature_dim) * 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pool temporal features.

        Args:
            x: Features of shape (B, T, feature_dim).

        Returns:
            Pooled features of shape (B, feature_dim).
        """
        mean_pool = x.mean(dim=1)
        cls_pool = x[:, 0, :]
        pooled = self.weight * mean_pool + (1 - self.weight) * cls_pool
        return pooled
