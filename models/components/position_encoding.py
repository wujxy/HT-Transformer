"""
Relative Position Encoding (RPE) via bucket-based learnable bias.

Supports:
- Angular distance buckets (spherical angular separation)
- Time difference buckets
"""

import torch
import torch.nn as nn
import math


class BucketRPE(nn.Module):
    """
    Bucket-based Relative Position Encoding.

    Computes learnable bias tables for angular distance and time difference,
    added to attention logits.
    """

    def __init__(self, num_heads: int = 4, num_angle_buckets: int = 64,
                 num_time_buckets: int = 64):
        super().__init__()
        self.num_heads = num_heads
        self.num_angle_buckets = num_angle_buckets
        self.num_time_buckets = num_time_buckets

        # Learnable bias tables: (num_buckets, num_heads)
        self.angle_bias = nn.Embedding(num_angle_buckets * num_heads, 1)
        self.time_bias = nn.Embedding(num_time_buckets * num_heads, 1)

    def _bucketize(self, values: torch.Tensor, num_buckets: int,
                   max_val: float) -> torch.Tensor:
        """
        Map continuous values to discrete bucket indices.

        Args:
            values: (...) tensor of values
            num_buckets: number of buckets
            max_val: maximum expected value for scaling

        Returns:
            (...) long tensor of bucket indices
        """
        # Scale to [0, 1] then to [0, num_buckets-1]
        normalized = values / (max_val + 1e-8)
        normalized = normalized.clamp(0.0, 1.0)
        buckets = (normalized * (num_buckets - 1)).long()
        return buckets

    def forward(self, angle_ij: torch.Tensor, delta_t_ij: torch.Tensor,
                max_angle: float = math.pi, max_dt: float = 1.0) -> torch.Tensor:
        """
        Compute RPE bias for attention logits.

        Args:
            angle_ij: (B, H, S_q, S_k) angular distances between tokens
            delta_t_ij: (B, H, S_q, S_k) time differences between tokens
            max_angle: maximum angular distance (pi for unit sphere)
            max_dt: maximum time difference (already normalized to ~1.0)

        Returns:
            (B, H, S_q, S_k) bias to add to attention logits
        """
        B, H, Sq, Sk = angle_ij.shape

        # Bucketize
        a_buckets = self._bucketize(angle_ij, self.num_angle_buckets, max_angle)
        t_buckets = self._bucketize(delta_t_ij, self.num_time_buckets, max_dt)

        # Add head offset for embedding lookup
        head_offsets = torch.arange(H, device=angle_ij.device).view(1, H, 1, 1)
        a_indices = a_buckets * H + head_offsets
        t_indices = t_buckets * H + head_offsets

        # Lookup biases
        a_bias = self.angle_bias(a_indices).squeeze(-1)  # (B, H, Sq, Sk)
        t_bias = self.time_bias(t_indices).squeeze(-1)   # (B, H, Sq, Sk)

        return a_bias + t_bias
