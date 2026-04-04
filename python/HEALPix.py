"""
HEALPix Patch Mapping Module for CD PMTs.

Provides:
- CD PMT copyno -> HEALPix pixel mapping
- Precomputed kNN adjacency for CD patches
- Time-bin construction for patch tokens
"""

import numpy as np
import healpy as hp
from typing import Tuple
from loguru import logger


class HEALPixMapper:
    """
    Maps CD PMT positions to HEALPix pixels and provides patch utilities.
    """

    def __init__(self, nside: int = 8, cd_unit_vecs: np.ndarray = None):
        """
        Args:
            nside: HEALPix nside parameter (default 8 -> 768 pixels)
            cd_unit_vecs: (N_cd_max, 3) array of CD PMT unit direction vectors,
                          indexed by copyno. Can be None for lazy init.
        """
        self.nside = nside
        self.npix = hp.nside2npix(nside)
        logger.info(f"HEALPix initialized: nside={nside}, npix={self.npix}")

        self._copyno_to_pixel = None  # lookup table
        self._pixel_center_vecs = None  # (npix, 3) pixel center unit vectors
        self._knn_adj = None  # (npix, k) kNN adjacency

        # Compute pixel center vectors (fixed for given nside)
        self._compute_pixel_centers()

        if cd_unit_vecs is not None:
            self.build_lookup(cd_unit_vecs)

    def _compute_pixel_centers(self):
        """Precompute HEALPix pixel center unit vectors."""
        # hp.pix2vec returns (x,y,z) in Cartesian for ring ordering
        vecs = hp.pix2vec(self.nside, np.arange(self.npix), nest=False)
        self._pixel_center_vecs = np.stack(vecs, axis=-1).astype(np.float64)  # (npix, 3)

    def build_lookup(self, cd_unit_vecs: np.ndarray):
        """
        Build copyno -> pixel_id lookup table.

        Args:
            cd_unit_vecs: (N_cd_max, 3) array of CD PMT unit vectors, indexed by copyno
        """
        n_cd = len(cd_unit_vecs)
        self._copyno_to_pixel = np.zeros(n_cd, dtype=np.int32)

        for i in range(n_cd):
            vec = cd_unit_vecs[i]
            if np.abs(np.linalg.norm(vec) - 1.0) < 0.01:  # valid unit vector
                # hp.vec2pix expects (x,y,z) with norm ~1
                theta = np.arccos(np.clip(vec[2], -1.0, 1.0))
                phi = np.arctan2(vec[1], vec[0]) % (2 * np.pi)
                self._copyno_to_pixel[i] = hp.ang2pix(self.nside, theta, phi, nest=False)
            # else: pixel = 0 (will be masked out)

        active_pixels = np.unique(self._copyno_to_pixel[self._copyno_to_pixel >= 0])
        logger.info(f"HEALPix lookup built: {n_cd} CD PMTs -> {len(active_pixels)} active pixels")

    def lookup(self, copynos: np.ndarray) -> np.ndarray:
        """
        Map CD PMT copynos to HEALPix pixel IDs.

        Args:
            copynos: (N,) array of CD copyno values

        Returns:
            (N,) array of pixel IDs
        """
        if self._copyno_to_pixel is None:
            raise RuntimeError("Call build_lookup() first")
        clipped = np.clip(copynos, 0, len(self._copyno_to_pixel) - 1)
        return self._copyno_to_pixel[clipped]

    def build_knn_adjacency(self, k: int = 16) -> np.ndarray:
        """
        Precompute kNN adjacency based on angular distance between pixel centers.

        Args:
            k: number of nearest neighbors

        Returns:
            (npix, k) array of neighbor pixel indices
        """
        if self._pixel_center_vecs is None:
            raise RuntimeError("Pixel centers not computed")

        npix = self.npix
        adj = np.zeros((npix, k), dtype=np.int32)

        for i in range(npix):
            # Angular distance from pixel i to all pixels
            dots = self._pixel_center_vecs @ self._pixel_center_vecs[i]
            dots = np.clip(dots, -1.0, 1.0)
            ang_dist = np.arccos(dots)

            # Get k+1 nearest (excluding self)
            nearest = np.argpartition(ang_dist, k + 1)[:k + 1]
            nearest = nearest[nearest != i][:k]
            if len(nearest) < k:
                # Pad with self if not enough neighbors (shouldn't happen for nside>=8)
                nearest = np.pad(nearest, (0, k - len(nearest)), constant_values=i)
            adj[i] = nearest[:k]

        self._knn_adj = adj
        logger.info(f"Built kNN adjacency: k={k}, shape={adj.shape}")
        return adj

    def get_knn_adjacency(self, k: int = 16) -> np.ndarray:
        """Get or build kNN adjacency table."""
        if self._knn_adj is None or self._knn_adj.shape[1] != k:
            self.build_knn_adjacency(k)
        return self._knn_adj

    def get_pixel_center(self, pixel_id: int) -> np.ndarray:
        """Get unit vector for a pixel center."""
        return self._pixel_center_vecs[pixel_id]

    def get_pixel_centers_batch(self, pixel_ids: np.ndarray) -> np.ndarray:
        """Get unit vectors for multiple pixel centers."""
        return self._pixel_center_vecs[pixel_ids]

    @staticmethod
    def build_time_bins(times: np.ndarray, charges: np.ndarray,
                        t_max: float, num_bins: int) -> np.ndarray:
        """
        Build time-bin charge histogram for a patch.

        Args:
            times: (N,) hit times within patch
            charges: (N,) hit charges within patch
            t_max: maximum time for binning
            num_bins: number of time bins

        Returns:
            (num_bins,) array of summed charges per bin
        """
        bins = np.zeros(num_bins, dtype=np.float32)
        if len(times) == 0:
            return bins

        bin_edges = np.linspace(0, t_max, num_bins + 1)
        bin_indices = np.clip(np.digitize(times, bin_edges) - 1, 0, num_bins - 1)
        np.add.at(bins, bin_indices, charges)
        return bins

    @staticmethod
    def aggregate_patch(cd_copynos: np.ndarray, cd_unit_vecs: np.ndarray,
                        cd_times: np.ndarray, cd_charges: np.ndarray,
                        pixel_ids: np.ndarray, t_max: float, num_bins: int
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Aggregate CD hits into patch tokens.

        For each active HEALPix pixel, compute:
        - Pixel center unit vector (ux, uy, uz)
        - First-level stats: sumQ, count, t_min, t_mean
        - Second-level time bins: (num_bins,) charge histogram

        Args:
            cd_copynos: (N_cd,) copyno of CD hits
            cd_unit_vecs: (N_cd, 3) unit vectors of CD hits
            cd_times: (N_cd,) normalized times of CD hits
            cd_charges: (N_cd,) normalized charges of CD hits
            pixel_ids: (N_cd,) HEALPix pixel IDs for each hit
            t_max: max time for binning
            num_bins: number of time bins

        Returns:
            active_pixels: (N_patches,) unique pixel IDs with hits
            patch_stats: (N_patches, 4) [sumQ, count, t_min, t_mean]
            patch_time_bins: (N_patches, num_bins) time-bin sequences
            patch_unit_vecs: (N_patches, 3) pixel center unit vectors
        """
        active_pixels = np.unique(pixel_ids)
        n_patches = len(active_pixels)
        patch_stats = np.zeros((n_patches, 4), dtype=np.float32)
        patch_time_bins = np.zeros((n_patches, num_bins), dtype=np.float32)
        patch_unit_vecs = np.zeros((n_patches, 3), dtype=np.float32)

        # Build reverse index
        pixel_to_idx = {p: i for i, p in enumerate(active_pixels)}

        for p_idx, pixel in enumerate(active_pixels):
            mask = pixel_ids == pixel
            hit_q = cd_charges[mask]
            hit_t = cd_times[mask]

            patch_stats[p_idx, 0] = hit_q.sum()           # sumQ
            patch_stats[p_idx, 1] = len(hit_q)             # count
            patch_stats[p_idx, 2] = hit_t.min() if len(hit_t) > 0 else 0.0  # t_min
            patch_stats[p_idx, 3] = (hit_q * hit_t).sum() / (hit_q.sum() + 1e-10)  # t_mean (charge-weighted)

            # Time bins (use original times, not normalized)
            patch_time_bins[p_idx] = HEALPixMapper.build_time_bins(
                hit_t, hit_q, t_max, num_bins)

            # Pixel center from lookup
            theta = np.arccos(np.clip(
                cd_unit_vecs[mask][0, 2] if len(cd_unit_vecs[mask]) > 0 else 1.0, -1, 1))
            phi = np.arctan2(
                cd_unit_vecs[mask][0, 1] if len(cd_unit_vecs[mask]) > 0 else 0.0,
                cd_unit_vecs[mask][0, 0] if len(cd_unit_vecs[mask]) > 0 else 1.0)
            patch_unit_vecs[p_idx] = cd_unit_vecs[mask].mean(axis=0)
            norm = np.linalg.norm(patch_unit_vecs[p_idx])
            if norm > 1e-10:
                patch_unit_vecs[p_idx] /= norm

        return active_pixels, patch_stats, patch_time_bins, patch_unit_vecs
