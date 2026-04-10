"""
Token Projectors for WP and CD tokens (V3 Architecture).

Maps raw token features to unified d_model dimension.
Includes type embedding (WP, CD, GLOBAL, QUERY) and Fourier position encoding.
"""

import torch
import torch.nn as nn


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
    'WPProjector', 'CDHitProjector', 'CDTimeEmbedding',
    'TokenTypeEmbedding', 'FourierPositionEncoding',
    'TOKEN_WP', 'TOKEN_CD', 'TOKEN_GLOBAL', 'TOKEN_QUERY',
]


class CDHitProjector(nn.Module):
    """
    Project sparse CD PMT tokens (10-dim) to d_model (v3).

    Dual-branch:
      geo branch: [ux, uy, uz] -> MLP -> d_geo
      physics-time branch: [q_sum, q_max, n_hits, t_first, t_mean, t_late, t_span] -> MLP -> d_pt
    Concat -> fusion MLP -> d_model

    Input:  (B, K_cd, 10)
    Output: (B, K_cd, d_model)
    """

    def __init__(self, d_model: int = 128, d_geo: int = 32, d_pt: int = 64,
                 hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        # Geometry branch: [ux, uy, uz]
        self.geo_branch = nn.Sequential(
            nn.Linear(3, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_geo),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Physics-time branch: [q_sum, q_max, n_hits, t_first, t_mean, t_late, t_span]
        self.pt_branch = nn.Sequential(
            nn.Linear(7, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_pt),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(d_geo + d_pt, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, cd_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cd_tokens: (B, K_cd, 10) [ux,uy,uz,q_sum,q_max,n_hits,t_first,t_mean,t_late,t_span]

        Returns:
            (B, K_cd, d_model)
        """
        geo_features = self.geo_branch(cd_tokens[..., :3])    # (B, K, d_geo)
        pt_features = self.pt_branch(cd_tokens[..., 3:])      # (B, K, d_pt)
        combined = torch.cat([geo_features, pt_features], dim=-1)
        return self.fusion(combined)


class CDTimeEmbedding(nn.Module):
    """
    Additive time embedding for CD PMT tokens (v3).

    Input: [t_first, t_mean, t_late, t_span] (4-dim, normalized to [0,1])
    Output: (B, K_cd, d_model)

    Uses Fourier features per time dimension + MLP, same philosophy as WPTimeEncoding.
    SDPA-compatible: time is injected as additive token-level embedding.
    """

    def __init__(self, d_model: int = 128, hidden: int = 32, fourier_dim: int = 16):
        super().__init__()
        self.d_model = d_model
        self.hidden = hidden
        self.fourier_dim = fourier_dim

        # Learnable frequency scales for Fourier encoding (4 time features)
        freqs = torch.randn(1, 1, 4, fourier_dim) * 4.0
        self.register_buffer("freqs", freqs)

        # MLP: [4 raw, 4*2*fourier_dim sin/cos] -> d_model
        self.mlp = nn.Sequential(
            nn.Linear(4 + 4 * 2 * fourier_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, cd_time_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cd_time_features: (B, K_cd, 4) [t_first, t_mean, t_late, t_span]

        Returns:
            (B, K_cd, d_model) time embeddings
        """
        # (B, K, 4) -> (B, K, 4, 1) * (1, 1, 4, F) -> (B, K, 4, F)
        t = cd_time_features.unsqueeze(-1)
        proj = t * self.freqs

        # (B, K, 4, F) -> sin/cos
        sin_feat = torch.sin(proj)  # (B, K, 4, F)
        cos_feat = torch.cos(proj)  # (B, K, 4, F)

        # Flatten Fourier features: (B, K, 4*2*F)
        fourier_flat = torch.cat([sin_feat, cos_feat], dim=-1).reshape(
            *cd_time_features.shape[:2], -1)

        # Concat raw + Fourier: (B, K, 4 + 4*2*F)
        feat = torch.cat([cd_time_features, fourier_flat], dim=-1)

        return self.mlp(feat)
