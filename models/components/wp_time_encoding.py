"""
WP Time Encoding module for injecting time information into WP token embeddings.

Provides:
- WPTimeEncoding: Token-level time encoding using Fourier features + MLP

This module replaces the deprecated SignedTimeBucketBias (which built explicit
pairwise time bias tensors) with a more efficient token-level encoding that
is SDPA-compatible.
"""

import torch
import torch.nn as nn


class WPTimeEncoding(nn.Module):
    """
    Token-level time encoding for WP hits.

    Replaces explicit pairwise time bias by encoding normalized hit time
directly into token embeddings. This approach:
    - Avoids building expensive (B, H, N, N) pairwise tensors
    - Is fully compatible with SDPA/FlashAttention fast path
    - Provides expressive time representation via Fourier features

    Input:
        wp_times: (B, N_wp), normalized to [0, 1]

    Output:
        time_emb: (B, N_wp, d_model)
    """

    def __init__(self, d_model: int, hidden: int = 32, fourier_dim: int = 16):
        """
        Args:
            d_model: Output embedding dimension
            hidden: Hidden dimension of the MLP
            fourier_dim: Number of Fourier frequency components
        """
        super().__init__()
        self.d_model = d_model
        self.hidden = hidden
        self.fourier_dim = fourier_dim

        # Learnable frequency scales for Fourier encoding
        # Using random scales provides good coverage of different time scales
        freqs = torch.randn(1, 1, fourier_dim) * 4.0
        self.register_buffer("freqs", freqs)

        # MLP: [time, sin(freq*time), cos(freq*time)] -> d_model
        # Input: 1 (raw time) + 2*fourier_dim (sin/cos) = 1 + 2*F
        self.mlp = nn.Sequential(
            nn.Linear(1 + 2 * fourier_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, wp_times: torch.Tensor) -> torch.Tensor:
        """
        Encode normalized WP hit times into embeddings.

        Args:
            wp_times: (B, N_wp) normalized hit times in [0, 1]

        Returns:
            (B, N_wp, d_model) time embeddings to add to token embeddings
        """
        # Add feature dimension: (B, N) -> (B, N, 1)
        t = wp_times.unsqueeze(-1)

        # Project time onto frequency axes: (B, N, 1) * (1, 1, F) -> (B, N, F)
        proj = t * self.freqs

        # Concatenate raw time with Fourier features
        # [t, sin(proj), cos(proj)] -> (B, N, 1 + 2*F)
        feat = torch.cat([t, torch.sin(proj), torch.cos(proj)], dim=-1)

        # MLP to d_model
        return self.mlp(feat)

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, hidden={self.hidden}, fourier_dim={self.fourier_dim}"
