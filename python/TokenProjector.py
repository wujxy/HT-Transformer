"""
Token Projectors for WP and CD tokens.

Maps raw token features to unified d_model dimension.
Includes type embedding (WP, CD, GLOBAL, QUERY) and Fourier position encoding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# Token type IDs
TOKEN_WP = 0
TOKEN_CD = 1
TOKEN_GLOBAL = 2
TOKEN_QUERY = 3


class WPProjector(nn.Module):
    """Project WP hit token [ux, uy, uz, q, t] -> d_model."""

    def __init__(self, input_dim: int = 5, d_model: int = 128):
        super().__init__()
        self.linear = nn.Linear(input_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class CDProjector(nn.Module):
    """
    Project CD patch token to d_model.

    Input: pixel_unit_vec (3) + first-level stats (4) + time-bin embedding
    Time-bin encoder: Conv1d -> GELU -> Conv1d -> GELU -> GAP -> Linear
    """

    def __init__(self, num_time_bins: int = 32, d_model: int = 128,
                 d_stats: int = 32, d_time: int = 32):
        super().__init__()
        # Stats projector: [ux, uy, uz, sumQ, count, t_min, t_mean] = 7 -> d_stats
        self.stats_proj = nn.Sequential(
            nn.Linear(7, d_stats),
            nn.GELU(),
        )

        # Time-bin encoder: Conv1d pipeline
        self.time_encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(16, 16, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.time_gap = nn.AdaptiveAvgPool1d(1)
        self.time_proj = nn.Linear(16, d_time)

        # Fusion
        self.fusion = nn.Linear(d_stats + d_time, d_model)

    def forward(self, patch_unit_vecs: torch.Tensor, patch_stats: torch.Tensor,
                patch_time_bins: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_unit_vecs: (B, N_patches, 3) pixel center unit vectors
            patch_stats: (B, N_patches, 4) [sumQ, count, t_min, t_mean]
            patch_time_bins: (B, N_patches, B_bins) time-bin charge histograms

        Returns:
            (B, N_patches, d_model)
        """
        # Concatenate unit vec + stats
        stats_input = torch.cat([patch_unit_vecs, patch_stats], dim=-1)  # (B, N, 7)
        stats_emb = self.stats_proj(stats_input)  # (B, N, d_stats)

        # Time-bin encoder
        B, N, B_bins = patch_time_bins.shape
        tb = patch_time_bins.view(B * N, 1, B_bins)  # (B*N, 1, B_bins)
        tb = self.time_encoder(tb)  # (B*N, 16, B_bins)
        tb = self.time_gap(tb).squeeze(-1)  # (B*N, 16)
        tb = self.time_proj(tb)  # (B*N, d_time)
        tb = tb.view(B, N, -1)  # (B, N, d_time)

        # Fusion
        combined = torch.cat([stats_emb, tb], dim=-1)  # (B, N, d_stats + d_time)
        return self.fusion(combined)


class TokenTypeEmbedding(nn.Module):
    """Learnable type embedding for WP, CD, GLOBAL, QUERY tokens."""

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(4, d_model)

    def forward(self, type_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            type_ids: (...) integer tensor of token type IDs

        Returns:
            (..., d_model) type embeddings
        """
        return self.embedding(type_ids)


class FourierPositionEncoding(nn.Module):
    """
    Fourier features on unit vector for absolute position encoding.

    Random frequency matrix B, then [sin(uB), cos(uB)] -> linear -> d_model.
    """

    def __init__(self, d_model: int = 128, num_frequencies: int = 32):
        super().__init__()
        self.num_freq = num_frequencies
        # Fixed random frequencies
        B = torch.randn(3, num_frequencies) * 2.0
        self.register_buffer('B', B)
        self.linear = nn.Linear(num_frequencies * 2, d_model)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u: (..., 3) unit vectors

        Returns:
            (..., d_model) Fourier position encoding
        """
        proj = u @ self.B  # (..., num_freq)
        features = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # (..., num_freq*2)
        return self.linear(features)
