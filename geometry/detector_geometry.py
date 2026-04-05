"""
PMT Geometry Lookup Module for JUNO CDWP.

Provides DualPMTPositionLookup for combined CD + WP hit position lookups.
Extracted and adapted from muon_track_reco_transformer/python/DataLoader.py.

Routing logic:
- copyno < 50000  → CD hits (PMTPos_CD_LPMT.csv, key=CopyNo)
- copyno >= 50000 → WP hits (PMTPos_WP_LPMT.csv, key=CopyNo)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict
from loguru import logger


PMT_RADIUS = 19433.975  # mm
PMT_COPYNO_OFFSET = 50000  # CD < 50000, WP >= 50000


class DualPMTPositionLookup:
    """
    Dual lookup table for CDWP detector type (CD + WP mixed hits).

    Provides vectorized position lookups for both CD and WP PMTs,
    plus subsystem tagging and unit direction vectors.
    """

    def __init__(self, cd_csv_path: str, wp_csv_path: str):
        self.cd_csv_path = cd_csv_path
        self.wp_csv_path = wp_csv_path
        self.cd_count = 0
        self.wp_count = 0
        self._load_pmt_positions()

    def _load_pmt_positions(self):
        """Load and index both CD and WP PMT position tables."""
        logger.info(f"Loading CDWP dual PMT positions:")
        logger.info(f"  CD: {self.cd_csv_path}")
        logger.info(f"  WP: {self.wp_csv_path}")

        # --- CD positions ---
        cd_df = pd.read_csv(self.cd_csv_path, sep=r'\s+', skiprows=4,
                            names=['CopyNo', 'X', 'Y', 'Z', 'Theta', 'Phi'])
        self.cd_position_map: Dict[int, Tuple[float, float, float]] = {}
        for _, row in cd_df.iterrows():
            pmt_id = int(row['CopyNo'])
            self.cd_position_map[pmt_id] = (row['X'], row['Y'], row['Z'])

        max_cd_id = max(self.cd_position_map.keys()) + 1
        self.cd_position_array = np.zeros((max_cd_id, 3), dtype=np.float32)
        for pmt_id, (x, y, z) in self.cd_position_map.items():
            self.cd_position_array[pmt_id] = [x, y, z]
        self.cd_count = len(self.cd_position_map)

        # --- WP positions ---
        wp_df = pd.read_csv(self.wp_csv_path, sep=r'\s+')
        self.wp_position_map: Dict[int, Tuple[float, float, float]] = {}
        for _, row in wp_df.iterrows():
            copyno = int(row['CopyNo'])
            self.wp_position_map[copyno] = (row['X'], row['Y'], row['Z'])

        max_wp_id = max(self.wp_position_map.keys()) + 1
        self.wp_position_array = np.zeros((max_wp_id, 3), dtype=np.float32)
        for copyno, (x, y, z) in self.wp_position_map.items():
            self.wp_position_array[copyno] = [x, y, z]
        self.wp_count = len(self.wp_position_map)

        logger.info(f"  CD: {self.cd_count} PMTs (max ID: {max_cd_id - 1})")
        logger.info(f"  WP: {self.wp_count} PMTs (max ID: {max_wp_id - 1})")

    def get_position(self, copyno: int) -> Tuple[float, float, float]:
        """Get PMT position for a single copyno."""
        if copyno < PMT_COPYNO_OFFSET:
            return self.cd_position_map.get(copyno, (0.0, 0.0, 0.0))
        else:
            return self.wp_position_map.get(copyno, (0.0, 0.0, 0.0))

    def get_positions_batch(self, copynos: np.ndarray) -> np.ndarray:
        """
        Vectorized position lookup for a batch of copyno values.

        Args:
            copynos: (N,) array of PMT copyno values

        Returns:
            (N, 3) array of PMT (X, Y, Z) positions in mm
        """
        positions = np.zeros((len(copynos), 3), dtype=np.float32)

        cd_mask = copynos < PMT_COPYNO_OFFSET
        wp_mask = copynos >= PMT_COPYNO_OFFSET

        if np.any(cd_mask):
            cd_copynos = copynos[cd_mask].astype(np.int32)
            cd_copynos_clipped = np.clip(cd_copynos, 0, len(self.cd_position_array) - 1)
            positions[cd_mask] = self.cd_position_array[cd_copynos_clipped]

        if np.any(wp_mask):
            wp_copynos = copynos[wp_mask].astype(np.int32)
            wp_copynos_clipped = np.clip(wp_copynos, 0, len(self.wp_position_array) - 1)
            positions[wp_mask] = self.wp_position_array[wp_copynos_clipped]

        return positions

    def get_subsystem_tags(self, copynos: np.ndarray) -> np.ndarray:
        """
        Return subsystem tag for each copyno.

        Args:
            copynos: (N,) array of PMT copyno values

        Returns:
            (N,) array of strings: 'CD' or 'WP'
        """
        tags = np.where(copynos < PMT_COPYNO_OFFSET, 'CD', 'WP')
        return tags

    def get_unit_vectors(self, copynos: np.ndarray) -> np.ndarray:
        """
        Get unit direction vectors for PMTs.

        Args:
            copynos: (N,) array of PMT copyno values

        Returns:
            (N, 3) array of unit vectors
        """
        positions = self.get_positions_batch(copynos)
        norms = np.linalg.norm(positions, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        return positions / norms

    def get_spherical_coords(self, copynos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get spherical coordinates (theta, phi) in radians for PMTs.
        Useful for HEALPix mapping.

        Args:
            copynos: (N,) array of PMT copyno values

        Returns:
            theta: (N,) polar angle [0, pi]
            phi:   (N,) azimuthal angle [0, 2*pi)
        """
        unit_vecs = self.get_unit_vectors(copynos)
        # unit_vecs: (N, 3) = (sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta))
        # But actually XYZ, so:
        x, y, z = unit_vecs[:, 0], unit_vecs[:, 1], unit_vecs[:, 2]
        theta = np.arccos(np.clip(z, -1.0, 1.0))
        phi = np.arctan2(y, x) % (2 * np.pi)
        return theta.astype(np.float64), phi.astype(np.float64)
