"""
Token Projectors for WP and CD tokens (V2 Architecture).

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
    """
    Enhanced WP token projector with dual-branch feature extraction.

    Branch 1 (Geometry): [ux, uy, uz] -> MLP
    Branch 2 (Optical-Time): [q, t] -> MLP
    Concat -> Fusion MLP -> d_model

    Input: (B, N_wp, 5) - [ux, uy, uz, q, t]
    Output: (B, N_wp, d_model)
    """

    def __init__(self, d_model: int = 128, d_geo: int = 32, d_qt: int = 32,
                 hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_geo = d_geo
        self.d_qt = d_qt

        # Geometry branch: [ux, uy, uz]
        self.geo_branch = nn.Sequential(
            nn.Linear(3, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_geo),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Optical-time branch: [q, t]
        self.qt_branch = nn.Sequential(
            nn.Linear(2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_qt),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(d_geo + d_qt, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N_wp, 5) - [ux, uy, uz, q, t]

        Returns:
            (B, N_wp, d_model)
        """
        geo_features = self.geo_branch(x[..., :3])   # (B, N, d_geo)
        qt_features = self.qt_branch(x[..., 3:])     # (B, N, d_qt)
        combined = torch.cat([geo_features, qt_features], dim=-1)  # (B, N, d_geo+d_qt)
        return self.fusion(combined)


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


__all__ = [
    'WPProjector', 'CDProjector',
    'TokenTypeEmbedding', 'FourierPositionEncoding',
    'TOKEN_WP', 'TOKEN_CD', 'TOKEN_GLOBAL', 'TOKEN_QUERY',
]
