"""
Normalization utilities for JUNO CDWP HT-Transformer.

Adapted from muon_track_reco_transformer/python/DataLoader.py Normalizer class.

Key differences from reference project:
- Time normalization: clip + divide by t_max (NOT subtract-min then divide-by-range)
  because input hittime is already relative trigger time.
- Charge normalization: separate CD/WP normalization for CDWP mode.
"""

import numpy as np
from loguru import logger


PMT_RADIUS = 19433.975  # mm


class Normalizer:
    """Normalization utilities for hit-level and endpoint data."""

    @staticmethod
    def normalize_time(times: np.ndarray, t_max: float) -> np.ndarray:
        """
        Normalize hit times by clipping and dividing by t_max.

        Input times are already relative trigger times. No subtract-min needed.

        Args:
            times: (N,) array of relative hit times in ns
            t_max: maximum time for clipping and normalization

        Returns:
            (N,) normalized times in [0, 1]
        """
        if len(times) == 0:
            return times
        if t_max <= 0:
            raise ValueError(f"t_max must be positive, got {t_max}")
        clipped = np.clip(times, 0.0, t_max)
        return clipped / t_max

    @staticmethod
    def estimate_t_max(times: np.ndarray, percentile: float = 99.0) -> float:
        """
        Estimate t_max from data using a percentile.

        Args:
            times: (N,) array of hit times
            percentile: percentile to use (default 99)

        Returns:
            Estimated t_max value
        """
        if len(times) == 0:
            return 1000.0  # fallback
        return float(np.percentile(times, percentile))

    @staticmethod
    def normalize_charge(charges: np.ndarray, apply_log: bool = True) -> np.ndarray:
        """
        Normalize charges: log10(q+1) → clip p99 → min-max [0,1].

        Args:
            charges: (N,) array of charge values in PE
            apply_log: whether to apply log10 transform

        Returns:
            (N,) normalized charges in [0, 1]
        """
        if len(charges) == 0:
            return charges

        if apply_log:
            charges = np.log10(charges + 1)

        p99 = np.percentile(charges, 99)
        charges = np.clip(charges, 0, p99)

        cmin, cmax = charges.min(), charges.max()
        if cmax > cmin:
            return (charges - cmin) / (cmax - cmin)
        else:
            return np.zeros_like(charges)

    @staticmethod
    def normalize_charge_cdwp(cd_charges: np.ndarray, wp_charges: np.ndarray,
                              apply_log: bool = True):
        from typing import Tuple
        """
        Normalize CD and WP charges independently.

        Args:
            cd_charges: (N_cd,) CD charge values
            wp_charges: (N_wp,) WP charge values
            apply_log: whether to apply log10 transform

        Returns:
            (norm_cd, norm_wp) tuple of normalized charges
        """
        norm_cd = Normalizer.normalize_charge(cd_charges, apply_log=apply_log)
        norm_wp = Normalizer.normalize_charge(wp_charges, apply_log=apply_log)
        return norm_cd, norm_wp

    @staticmethod
    def normalize_position(positions: np.ndarray) -> np.ndarray:
        """Normalize positions by PMT_RADIUS."""
        return positions / PMT_RADIUS

    @staticmethod
    def normalize_endpoint(xyz: np.ndarray) -> np.ndarray:
        """
        Normalize endpoint to unit vector.

        Args:
            xyz: (..., 3) array of endpoint coordinates

        Returns:
            (..., 3) unit vectors on sphere
        """
        norms = np.linalg.norm(xyz, axis=-1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        return xyz / norms
