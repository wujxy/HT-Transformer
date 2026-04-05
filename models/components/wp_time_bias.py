"""
WP Time Bias module for injecting signed relative time bias into WP self-attention.

**DEPRECATION NOTICE:**
This module is deprecated for the main training path. It builds explicit pairwise
bias tensors of shape (B, H, N, N) which prevents using SDPA/FlashAttention fast path.

For new training, use WPTimeEncoding (token-level time encoding) instead,
which is fully SDPA-compatible and significantly faster.

This module is kept for:
- Legacy compatibility
- Ablation studies
- Research experiments

Provides:
- SignedTimeBucketBias: Converts relative time differences to bucketed bias values

Performance Warning:
- Explicit (B, H, N, N) bias tensors are memory and compute intensive
- Disables SDPA fast path, falling back to manual attention
- Consider using token-level time encoding (WPTimeEncoding) for production
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class SignedTimeBucketBias(nn.Module):
    """
    Signed relative time bias for WP self-attention.

    **DEPRECATED for main training path.** Use WPTimeEncoding instead.

    This module builds explicit (B, num_heads, N_wp, N_wp) bias tensors which:
    - Are memory intensive for large N_wp
    - Prevent using SDPA/FlashAttention fast path
    - Require manual attention computation

    Converts time differences between WP hits into bucketed bias values
    that can be added to attention logits.

    Key features:
    - Signed buckets (not just absolute time difference)
    - Head-specific biases (unless heads_shared=True)
    - Configurable number of buckets

    Input: wp_times (B, N_wp)
    Output: time_bias (B, num_heads, N_wp, N_wp)

    For production use, prefer WPTimeEncoding which uses token-level encoding
    and is fully SDPA-compatible.
    """

    def __init__(self, num_buckets: int = 64, num_heads: int = 4,
                 heads_shared: bool = False, max_time: float = 1.0):
        """
        Args:
            num_buckets: Number of time difference buckets (should be odd for symmetry)
            num_heads: Number of attention heads
            heads_shared: If True, all heads share the same bias table
            max_time: Maximum expected normalized time value (default 1.0)
        """
        super().__init__()
        self.num_buckets = num_buckets
        self.num_heads = num_heads
        self.heads_shared = heads_shared
        self.max_time = max_time

        # Performance note:
        # heads_shared=True is cheaper and usually recommended for large N_wp.
        # It reduces the number of bias lookups from H separate tables to 1 shared table.

        # Create bucket boundaries
        # We want symmetric buckets around 0 for signed time differences
        # delta_t ranges from -max_time to +max_time

        if num_buckets % 2 == 0:
            # Even number of buckets - add a small epsilon to avoid boundary issues
            self._has_center_bucket = False
        else:
            # Odd number of buckets - middle bucket is centered at 0
            self._has_center_bucket = True

        # Bucket boundaries in original time scale
        # Linear spacing from -max_time to +max_time
        boundaries = torch.linspace(-max_time, max_time, num_buckets + 1)
        self.register_buffer('boundaries', boundaries)

        # Learnable bias values per bucket (and per head if not shared)
        if heads_shared:
            # Single bias table shared across all heads
            self.bias_table = nn.Parameter(torch.zeros(num_buckets))
        else:
            # Separate bias table per head
            self.bias_table = nn.Parameter(torch.zeros(num_heads, num_buckets))

        self._init_weights()

    def _init_weights(self):
        """Initialize bias values to small random values."""
        nn.init.normal_(self.bias_table, mean=0.0, std=0.02)

    def _time_to_bucket(self, delta_t: torch.Tensor) -> torch.Tensor:
        """
        Convert time differences to bucket indices.

        Args:
            delta_t: (..., ) time differences

        Returns:
            (..., ) bucket indices (0 to num_buckets-1)
        """
        # Clamp to valid range
        delta_t = delta_t.clamp(-self.max_time, self.max_time)

        # Digitize: find which bucket each value falls into
        # boundaries[i] <= value < boundaries[i+1] goes to bucket i
        # Use searchsorted for efficiency
        bucket_ids = torch.searchsorted(self.boundaries, delta_t, right=False) - 1
        bucket_ids = bucket_ids.clamp(0, self.num_buckets - 1)

        return bucket_ids

    def forward(self, wp_times: torch.Tensor) -> torch.Tensor:
        """
        Compute time bias matrix for WP self-attention.

        Args:
            wp_times: (B, N_wp) normalized hit times

        Returns:
            time_bias: (B, num_heads, N_wp, N_wp) additive bias for attention logits
        """
        B, N = wp_times.shape

        # Fast disable mode: return zero bias for ablation/profiling
        if self.num_buckets <= 1:
            return torch.zeros(
                B, self.num_heads, N, N,
                device=wp_times.device,
                dtype=wp_times.dtype,
            )

        # Compute pairwise time differences: t_i - t_j
        # delta_t[i, j] = t_i - t_j (signed)
        # Positive means query (i) is later than key (j)
        t_i = wp_times.unsqueeze(2)  # (B, N, 1)
        t_j = wp_times.unsqueeze(1)  # (B, 1, N)
        delta_t = t_i - t_j  # (B, N, N)

        # Convert to bucket indices
        bucket_ids = self._time_to_bucket(delta_t)  # (B, N, N)

        # Lookup bias values
        if self.heads_shared:
            # bias_table: (num_buckets,) -> lookup gives (B, N, N)
            bias = self.bias_table[bucket_ids]  # (B, N, N)
            # Expand to all heads
            bias = bias.unsqueeze(1).expand(-1, self.num_heads, -1, -1)  # (B, H, N, N)
        else:
            # bias_table: (num_heads, num_buckets)
            # Need to lookup per head: (B, H, N, N)
            bias = self.bias_table[:, bucket_ids]  # (H, B, N, N)
            bias = bias.permute(1, 0, 2, 3)  # (B, H, N, N)

        return bias

    def get_bucket_info(self) -> dict:
        """Get information about bucket assignments for debugging."""
        info = {
            'num_buckets': self.num_buckets,
            'boundaries': self.boundaries.tolist(),
            'has_center_bucket': self._has_center_bucket,
            'heads_shared': self.heads_shared,
        }

        # Show some example mappings
        example_times = torch.linspace(-self.max_time, self.max_time, 11)
        example_buckets = self._time_to_bucket(example_times)
        info['example_mappings'] = [
            {'time': t.item(), 'bucket': b.item()}
            for t, b in zip(example_times, example_buckets)
        ]

        return info


class LogTimeBucketBias(SignedTimeBucketBias):
    """
    Log-scaled time bucket bias for handling wide time ranges.

    Uses logarithmic bucketing for time differences, which may be more
    appropriate when time differences span multiple orders of magnitude.
    """

    def __init__(self, num_buckets: int = 64, num_heads: int = 4,
                 heads_shared: bool = False, max_time: float = 1.0,
                 min_time: float = 1e-6):
        """
        Args:
            min_time: Minimum time difference for log scaling (to avoid log(0))
        """
        super().__init__(num_buckets, num_heads, heads_shared, max_time)
        self.min_time = min_time

    def _time_to_bucket(self, delta_t: torch.Tensor) -> torch.Tensor:
        """
        Convert time differences to bucket indices using log spacing.

        For signed time differences, we handle positive and negative separately:
        - Negative values: log(|delta_t|) with negative sign
        - Zero: center bucket
        - Positive values: log(delta_t)
        """
        # Separate sign and magnitude
        sign = torch.sign(delta_t)
        abs_t = delta_t.abs().clamp(self.min_time, self.max_time)

        # Log scale the magnitude
        log_t = torch.log(abs_t / self.min_time) / math.log(self.max_time / self.min_time)

        # Reapply sign (now in [-1, 1] range, log-scaled)
        scaled_t = sign * log_t

        # Map from [-1, 1] to bucket indices
        # Linear spacing from -1 to 1
        bucket_ids = ((scaled_t + 1.0) / 2.0 * self.num_buckets).long()
        bucket_ids = bucket_ids.clamp(0, self.num_buckets - 1)

        return bucket_ids
