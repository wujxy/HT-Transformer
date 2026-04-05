"""
DataLoader for JUNO CDWP HT-Transformer.

Reads h5 files, separates CD/WP hits, performs:
- WP tokenization: [ux, uy, uz, q, t] per hit
- CD tokenization: HEALPix patch aggregation with two-level time representation

Produces batch dicts compatible with Model.py.
"""

import os
import glob as glob_module
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import Dict, List, Optional, Tuple, Union
import h5py
import healpy as hp
from loguru import logger

from geometry.detector_geometry import DualPMTPositionLookup, PMT_COPYNO_OFFSET
from data.normalization import Normalizer
from geometry.healpix_mapper import HEALPixMapper


def build_dense_cd_patches(
    pixel_ids: np.ndarray,
    cd_unit_vecs_hits: np.ndarray,
    cd_times: np.ndarray,
    cd_charges: np.ndarray,
    pixel_center_vecs: np.ndarray,
    npix: int,
    num_time_bins: int,
    t_max: float,
):
    """
    Build dense HEALPix patch representation for Stage B.

    Stage B optimization: Returns fixed-shape dense HEALPix grid instead of
    variable-length active patch list. This enables:
    - Fixed graph topology for DeepSphere
    - No runtime local graph construction
    - Better SDPA/compile compatibility

    Args:
        pixel_ids: (N_hits,) HEALPix pixel id per CD hit
        cd_unit_vecs_hits: (N_hits, 3) CD hit directions
        cd_times: (N_hits,) normalized times [0, 1]
        cd_charges: (N_hits,) normalized charges
        pixel_center_vecs: (npix, 3) fixed HEALPix pixel center unit vectors
        npix: total number of pixels for nside (12 * nside^2)
        num_time_bins: number of time bins
        t_max: max time for binning (used for time bin calculation)

    Returns:
        cd_unit_vecs_dense: (npix, 3) pixel center unit vectors
        cd_stats_dense: (npix, 4) [sumQ, count, t_min, t_mean]
        cd_time_bins_dense: (npix, num_time_bins) time histograms
        cd_mask_dense: (npix,) True = inactive (no hits)
        cd_times_mean: (npix,) charge-weighted mean time per pixel
    """
    # Initialize with fixed shape
    cd_unit_vecs_dense = pixel_center_vecs.astype(np.float32).copy()
    cd_stats_dense = np.zeros((npix, 4), dtype=np.float32)
    cd_time_bins_dense = np.zeros((npix, num_time_bins), dtype=np.float32)
    cd_mask_dense = np.ones((npix,), dtype=bool)  # True = inactive

    if len(pixel_ids) == 0:
        cd_times_mean = np.zeros((npix,), dtype=np.float32)
        return cd_unit_vecs_dense, cd_stats_dense, cd_time_bins_dense, cd_mask_dense, cd_times_mean

    # Group hits by pixel and aggregate
    unique_pixels = np.unique(pixel_ids)
    for pix in unique_pixels:
        mask = (pixel_ids == pix)
        hit_q = cd_charges[mask]
        hit_t = cd_times[mask]

        # Aggregate stats
        cd_stats_dense[pix, 0] = hit_q.sum()  # sumQ
        cd_stats_dense[pix, 1] = len(hit_q)   # count
        cd_stats_dense[pix, 2] = hit_t.min() if len(hit_t) > 0 else 0.0  # t_min
        cd_stats_dense[pix, 3] = (hit_q * hit_t).sum() / (hit_q.sum() + 1e-10)  # t_mean

        # Time bins (times already normalized to [0, 1])
        bin_idx = np.clip(
            (hit_t * num_time_bins).astype(np.int32),
            0, num_time_bins - 1
        )
        np.add.at(cd_time_bins_dense[pix], bin_idx, hit_q)

        # Mark as active
        cd_mask_dense[pix] = False

    # Extract t_mean as separate array for convenience
    cd_times_mean = cd_stats_dense[:, 3].copy()

    return cd_unit_vecs_dense, cd_stats_dense, cd_time_bins_dense, cd_mask_dense, cd_times_mean


def discover_h5_files(path: Union[str, List[str]]) -> List[str]:
    """
    Expand a path specification to a sorted list of H5 files.

    Supports:
    - List[str]: return sorted list directly
    - str + is directory: recursive search for *.h5
    - str + contains glob chars (*?): glob match
    - str + is single file: return [path]

    Returns:
        Sorted list of absolute file paths
    """
    if isinstance(path, list):
        return sorted(path)

    # Directory
    if os.path.isdir(path):
        files = sorted(glob_module.glob(os.path.join(path, '**/*.h5'), recursive=True))
        if not files:
            # Also try non-recursive
            files = sorted(glob_module.glob(os.path.join(path, '*.h5')))
        return files

    # Glob pattern
    if '*' in path or '?' in path:
        return sorted(glob_module.glob(path))

    # Single file
    if os.path.isfile(path):
        return [path]

    # Fallback: try appending *.h5
    files = sorted(glob_module.glob(path + '*.h5'))
    if files:
        return files

    raise FileNotFoundError(f"No H5 files found for path: {path}")


class H5EndpointDataset(Dataset):
    """
    H5 dataset for endpoint reconstruction.

    Supports multiple H5 files via a (file_id, local_idx) dual-index mapping.
    Reads events from h5, tokenizes into WP hits + CD patches,
    returns batch dict for HT-Transformer.
    """

    def __init__(self, h5_files: Union[str, List[str]], config: dict,
                 geometry: DualPMTPositionLookup,
                 healpix: HEALPixMapper,
                 indices: Optional[np.ndarray] = None,
                 is_train: bool = True,
                 events_per_file: Optional[List[int]] = None,
                 apply_rotation_aug: bool = False):
        """
        Args:
            h5_files: path(s) to h5 file(s). Can be:
                - single file path (str)
                - directory path (str) - will auto-discover *.h5
                - glob pattern (str) e.g. "/path/run_*.h5"
                - list of file paths
            config: full config dict
            geometry: DualPMTPositionLookup instance
            healpix: HEALPixMapper instance
            indices: global event indices to use (for train/val/test split)
            is_train: whether this is training mode
            events_per_file: pre-computed list of event counts per file (avoids re-scanning)
            apply_rotation_aug: whether to apply per-event random SO(3) rotation augmentation
        """
        # Discover files
        if isinstance(h5_files, str):
            h5_files = discover_h5_files(h5_files)
        self._file_paths: List[str] = sorted(h5_files)

        if len(self._file_paths) == 0:
            raise FileNotFoundError("No H5 files found")

        self.cfg = config
        self.data_cfg = config['data']
        self.model_cfg = config['model']
        self.geo = geometry
        self.healpix = healpix
        self.is_train = is_train
        self._apply_rotation_aug = apply_rotation_aug  # per-event random SO(3) rotation

        # Key mapping
        self.km = self.data_cfg['h5_key_map']

        # Normalization params
        self.t_max = self.data_cfg['t_max']
        self.num_time_bins = self.data_cfg['num_time_bins']
        self.nside = self.data_cfg['nside']

        # Build file index (global_idx -> file_id, local_idx)
        self._build_file_index(events_per_file)

        if indices is not None:
            self.indices = indices
        else:
            self.indices = np.arange(self.n_events)

        # Precompute global t_max if auto
        if self.data_cfg.get('auto_t_max', False) and self.t_max is None:
            self.t_max = self._estimate_t_max()
            logger.info(f"Auto-estimated t_max = {self.t_max:.1f} ns")

        n_files = len(self._file_paths)
        logger.info(f"H5EndpointDataset: {len(self.indices)} events from {n_files} file(s)")

        # Tokenization cache (keyed by global_idx)
        self._use_cache = self.data_cfg.get('use_cache', True)
        self._token_cache: Dict[int, Dict[str, torch.Tensor]] = {}
        if self._use_cache:
            logger.info(f"Tokenization cache enabled")

    def _build_file_index(self, events_per_file: Optional[List[int]] = None):
        """Build mapping from global event index to (file_id, local_idx)."""
        self._sample_file_ids = np.zeros(0, dtype=np.int32)
        self._sample_local_indices = np.zeros(0, dtype=np.int32)

        offset = 0
        file_ids_list = []
        local_idx_list = []

        for file_id, fp in enumerate(self._file_paths):
            if events_per_file is not None and file_id < len(events_per_file):
                n = events_per_file[file_id]
            else:
                with h5py.File(fp, 'r') as f:
                    n = f[self.km['nhits']].shape[0]

            file_ids_list.append(np.full(n, file_id, dtype=np.int32))
            local_idx_list.append(np.arange(n, dtype=np.int32))
            offset += n

        self._sample_file_ids = np.concatenate(file_ids_list)
        self._sample_local_indices = np.concatenate(local_idx_list)
        self.n_events = offset

    def _get_h5_file(self, file_path: str) -> h5py.File:
        """Get cached H5 file handle, opening lazily."""
        if not hasattr(self, '_h5_cache'):
            self._h5_cache: Dict[str, h5py.File] = {}
        if file_path not in self._h5_cache:
            self._h5_cache[file_path] = h5py.File(file_path, 'r')
        return self._h5_cache[file_path]

    def _estimate_t_max(self) -> float:
        """Estimate t_max from data p99, sampling across files."""
        all_times = []
        km = self.km
        sampled = 0
        for fp in self._file_paths:
            if sampled >= 100:
                break
            f = self._get_h5_file(fp)
            n = f[km['nhits']].shape[0]
            for i in range(min(n, 100 - sampled)):
                nhits = int(f[km['nhits']][i])
                times = f[km['time']][i][:nhits]
                all_times.append(times)
                sampled += 1
                if sampled >= 100:
                    break
        all_times = np.concatenate(all_times)
        return float(np.percentile(all_times, 99)) * 1.2  # 20% margin

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load and tokenize one event."""
        global_idx = self.indices[idx]

        # --- Cache lookup ---
        if self._use_cache and global_idx in self._token_cache:
            return self._token_cache[global_idx]

        km = self.km

        # Dual-index mapping: global_idx -> (file_id, local_idx)
        file_id = self._sample_file_ids[global_idx]
        local_idx = self._sample_local_indices[global_idx]
        f = self._get_h5_file(self._file_paths[file_id])

        # Read raw data
        nhits = int(f[km['nhits']][local_idx])
        copyno = f[km['copyno']][local_idx][:nhits].astype(np.int32)
        charge = f[km['charge']][local_idx][:nhits].astype(np.float32)
        time = f[km['time']][local_idx][:nhits].astype(np.float32)

        # Read labels
        enter = np.array([
            f[km['enter_x']][local_idx],
            f[km['enter_y']][local_idx],
            f[km['enter_z']][local_idx],
        ], dtype=np.float32)
        exit_pt = np.array([
            f[km['exit_x']][local_idx],
            f[km['exit_y']][local_idx],
            f[km['exit_z']][local_idx],
        ], dtype=np.float32)

        # --- Geometry lookup ---
        positions = self.geo.get_positions_batch(copyno)  # (N, 3)
        unit_vecs = self.geo.get_unit_vectors(copyno)      # (N, 3)
        subsystem = self.geo.get_subsystem_tags(copyno)     # (N,) 'CD' or 'WP'

        # --- Per-event random SO(3) rotation augmentation ---
        if self._apply_rotation_aug:
            from data.augmentation import random_rotation_matrix
            R = random_rotation_matrix()
            unit_vecs = (R @ unit_vecs.T).T          # (N, 3)
            positions = (R @ positions.T).T          # (N, 3)
            enter = R @ enter                        # (3,)
            exit_pt = R @ exit_pt                    # (3,)

        # Normalize labels to unit vectors
        u1 = Normalizer.normalize_endpoint(enter.reshape(1, 3)).squeeze(0)  # (3,)
        u2 = Normalizer.normalize_endpoint(exit_pt.reshape(1, 3)).squeeze(0)  # (3,)

        # --- Separate CD / WP ---
        cd_mask = subsystem == 'CD'
        wp_mask = subsystem == 'WP'

        cd_copyno = copyno[cd_mask]
        cd_unit = unit_vecs[cd_mask]     # (N_cd, 3)
        cd_charge = charge[cd_mask]
        cd_time = time[cd_mask]

        wp_unit = unit_vecs[wp_mask]     # (N_wp, 3)
        wp_charge = charge[wp_mask]
        wp_time = time[wp_mask]

        # --- Normalize ---
        # Charge: CD/WP separate
        norm_cd_q = Normalizer.normalize_charge(cd_charge, apply_log=True) if len(cd_charge) > 0 else np.array([], dtype=np.float32)
        norm_wp_q = Normalizer.normalize_charge(wp_charge, apply_log=True) if len(wp_charge) > 0 else np.array([], dtype=np.float32)

        # Time: unified clip + normalize
        norm_cd_t = Normalizer.normalize_time(cd_time, self.t_max) if len(cd_time) > 0 else np.array([], dtype=np.float32)
        norm_wp_t = Normalizer.normalize_time(wp_time, self.t_max) if len(wp_time) > 0 else np.array([], dtype=np.float32)

        # --- WP tokens: [ux, uy, uz, q, t] ---
        N_wp = len(wp_unit)
        if N_wp > 0:
            wp_tokens = np.stack([wp_unit[:, 0], wp_unit[:, 1], wp_unit[:, 2],
                                  norm_wp_q, norm_wp_t], axis=-1)  # (N_wp, 5)
        else:
            wp_tokens = np.zeros((0, 5), dtype=np.float32)

        # --- CD patch tokens: Stage B dense HEALPix grid representation ---
        N_cd = len(cd_unit)
        npix = self.healpix.npix  # Fixed: 12 * nside^2

        if N_cd > 0:
            # When rotation is applied, PMT positions changed → HEALPix pixel assignment changes → re-map
            if self._apply_rotation_aug:
                theta_cd = np.arccos(np.clip(cd_unit[:, 2], -1.0, 1.0))
                phi_cd = np.arctan2(cd_unit[:, 1], cd_unit[:, 0]) % (2 * np.pi)
                pixel_ids = hp.ang2pix(self.healpix.nside, theta_cd, phi_cd, nest=False)
            else:
                pixel_ids = self.healpix.lookup(cd_copyno)  # (N_cd,)

            # Build dense HEALPix grid (fixed shape: npix)
            cd_patch_unit, cd_stats, cd_time_bins, cd_mask_dense, cd_patch_t_mean = build_dense_cd_patches(
                pixel_ids=pixel_ids,
                cd_unit_vecs_hits=cd_unit.astype(np.float32),
                cd_times=norm_cd_t,
                cd_charges=norm_cd_q,
                pixel_center_vecs=self.healpix.cd_unit_vecs,  # (npix, 3)
                npix=npix,
                num_time_bins=self.num_time_bins,
                t_max=self.t_max,
            )
        else:
            # No CD hits: return empty dense grid (all masked)
            cd_patch_unit = self.healpix.cd_unit_vecs.astype(np.float32).copy()
            cd_stats = np.zeros((npix, 4), dtype=np.float32)
            cd_time_bins = np.zeros((npix, self.num_time_bins), dtype=np.float32)
            cd_mask_dense = np.ones((npix,), dtype=bool)

        # Fixed shape for Stage B: (npix, ...)
        n_patches = npix
        cd_patch_t_mean = cd_stats[:, 3]  # t_mean is 4th column

        # Convert to tensors
        # Note: cd_pixel_ids is deprecated in Stage B (fixed grid, index = pixel id)
        cd_pixel_ids = torch.arange(npix, dtype=torch.int64)  # 0, 1, ..., npix-1

        # Stage B: CD uses fixed dense HEALPix grid representation
        # Shape is now fixed (npix, ...) instead of variable (N_active_patches, ...)
        result = {
            'wp_tokens': torch.from_numpy(wp_tokens),           # (N_wp, 5)
            'wp_mask': torch.zeros(N_wp, dtype=torch.bool),     # False = valid
            'wp_unit_vecs': torch.from_numpy(wp_unit.astype(np.float32)),  # (N_wp, 3)
            'wp_times': torch.from_numpy(norm_wp_t.astype(np.float32)),    # (N_wp,)
            # Stage B: CD uses fixed dense HEALPix grid (npix = 12 * nside^2)
            'cd_unit_vecs': torch.from_numpy(cd_patch_unit),    # (npix, 3)
            'cd_stats': torch.from_numpy(cd_stats),             # (npix, 4)
            'cd_time_bins': torch.from_numpy(cd_time_bins),     # (npix, B_bins)
            'cd_mask': torch.from_numpy(cd_mask_dense),         # (npix,) True = inactive/no hits
            'cd_times_mean': torch.from_numpy(cd_patch_t_mean),  # (npix,)
            # DEPRECATED in Stage B: kept for compatibility, use index as pixel id
            'cd_pixel_ids': cd_pixel_ids,                        # (npix,) = [0, 1, ..., npix-1]
            'u1': torch.from_numpy(u1),                          # (3,)
            'u2': torch.from_numpy(u2),                          # (3,)
            'p1': torch.from_numpy(enter),                       # (3,) raw xyz
            'p2': torch.from_numpy(exit_pt),                     # (3,) raw xyz
        }

        # --- Cache store ---
        if self._use_cache:
            self._token_cache[global_idx] = result

        return result

    def __del__(self):
        if hasattr(self, '_h5_cache'):
            for f in self._h5_cache.values():
                try:
                    f.close()
                except Exception:
                    pass
            self._h5_cache.clear()


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate: pad variable-length WP and CD tokens to batch max.
    """
    # Stage B: CD uses fixed dense HEALPix grid, so no padding needed for CD
    # Only WP needs padding due to variable hit counts
    max_wp = max(b['wp_tokens'].shape[0] for b in batch)
    B = len(batch)
    npix = batch[0]['cd_unit_vecs'].shape[0]  # Fixed: 12 * nside^2
    num_time_bins = batch[0]['cd_time_bins'].shape[1] if batch[0]['cd_time_bins'].dim() > 1 else 0

    # Allocate tensors
    # WP: variable length, needs padding
    wp_tokens = torch.zeros(B, max_wp, 5)
    wp_mask = torch.ones(B, max_wp, dtype=torch.bool)  # True = padding
    wp_unit_vecs = torch.zeros(B, max_wp, 3)
    wp_times = torch.zeros(B, max_wp)

    # Stage B: CD fixed shape (npix, ...), no padding needed, just stack
    cd_unit_vecs = torch.zeros(B, npix, 3)
    cd_stats = torch.zeros(B, npix, 4)
    cd_time_bins = torch.zeros(B, npix, num_time_bins)
    cd_mask = torch.ones(B, npix, dtype=torch.bool)   # True = inactive/no hits
    cd_times_mean = torch.zeros(B, npix)
    # DEPRECATED in Stage B: kept for compatibility
    cd_pixel_ids = torch.arange(npix, dtype=torch.long).unsqueeze(0).expand(B, -1)

    u1 = torch.zeros(B, 3)
    u2 = torch.zeros(B, 3)
    p1 = torch.zeros(B, 3)
    p2 = torch.zeros(B, 3)

    for i, b in enumerate(batch):
        n_wp = b['wp_tokens'].shape[0]

        if n_wp > 0:
            wp_tokens[i, :n_wp] = b['wp_tokens']
            wp_mask[i, :n_wp] = False
            wp_unit_vecs[i, :n_wp] = b['wp_unit_vecs']
            wp_times[i, :n_wp] = b['wp_times']

        # Stage B: CD fixed shape, direct copy
        cd_unit_vecs[i] = b['cd_unit_vecs']
        cd_stats[i] = b['cd_stats']
        cd_time_bins[i] = b['cd_time_bins']
        cd_mask[i] = b['cd_mask']  # Use the event's cd_mask (True = inactive)
        cd_times_mean[i] = b['cd_times_mean']
        # cd_pixel_ids[i] = b['cd_pixel_ids']  # Now fixed [0, 1, ..., npix-1]

        u1[i] = b['u1']
        u2[i] = b['u2']
        p1[i] = b['p1']
        p2[i] = b['p2']

    result = {
        'wp_tokens': wp_tokens,
        'wp_mask': wp_mask,
        'wp_unit_vecs': wp_unit_vecs,
        'wp_times': wp_times,
        'cd_unit_vecs': cd_unit_vecs,
        'cd_stats': cd_stats,
        'cd_time_bins': cd_time_bins,
        'cd_mask': cd_mask,
        'cd_times_mean': cd_times_mean,
        'cd_pixel_ids': cd_pixel_ids,
        'u1': u1,
        'u2': u2,
        'p1': p1,
        'p2': p2,
    }

    return result


def create_dataloaders(config: dict, geometry: DualPMTPositionLookup,
                       healpix: HEALPixMapper) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test DataLoaders from config.

    Auto-detects preprocessed data if available, otherwise uses raw H5 files.

    Supports h5_path as:
    - Single file path
    - Directory (auto-discover *.h5)
    - Glob pattern (e.g. "/path/run_*.h5")

    Returns:
        (train_loader, val_loader, test_loader)
    """
    # --- Check for preprocessed data first ---
    mission = config.get('mission_name', 'ht_transformer_v1')
    output_base = config.get('output_path', 'output')
    preprocessed_dir = os.path.join(output_base, mission, 'preprocessed')
    manifest_path = os.path.join(preprocessed_dir, 'manifest.json')

    if os.path.exists(manifest_path):
        logger.info(f"Found preprocessed data at {preprocessed_dir}, using directly")
        from data.preprocess import create_preprocessed_dataloaders
        return create_preprocessed_dataloaders(config, preprocessed_dir)

    logger.info("No preprocessed data found, using raw H5 files")

    data_cfg = config['data']
    h5_path = data_cfg['h5_path']

    # Discover H5 files
    h5_files = discover_h5_files(h5_path)
    if len(h5_files) == 0:
        raise FileNotFoundError(f"No H5 files found for: {h5_path}")
    logger.info(f"Discovered {len(h5_files)} H5 file(s) from: {h5_path}")

    # Count total events across all files
    km = data_cfg['h5_key_map']
    total_events = 0
    events_per_file = []
    for fp in h5_files:
        with h5py.File(fp, 'r') as f:
            n = f[km['nhits']].shape[0]
            events_per_file.append(n)
            total_events += n
    logger.info(f"Total events: {total_events} across {len(h5_files)} files")

    # Split indices
    ratios = [data_cfg['train_ratio'], data_cfg['val_ratio'], data_cfg['test_ratio']]
    indices = np.arange(total_events)
    np.random.seed(config['train'].get('seed', 42))
    np.random.shuffle(indices)

    n_train = int(ratios[0] * total_events)
    n_val = int(ratios[1] * total_events)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    logger.info(f"Data split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    batch_size = config['train']['batch_size']

    train_ds = H5EndpointDataset(h5_files, config, geometry, healpix, train_idx,
                                  is_train=True, events_per_file=events_per_file)
    val_ds = H5EndpointDataset(h5_files, config, geometry, healpix, val_idx,
                                is_train=False, events_per_file=events_per_file)
    test_ds = H5EndpointDataset(h5_files, config, geometry, healpix, test_idx,
                                 is_train=False, events_per_file=events_per_file)

    # Disable num_workers when using accelerate to avoid fork/deadlock issues
    use_accelerate = config.get('train', {}).get('use_accelerate', False)
    if use_accelerate:
        num_workers = 0
        prefetch = None
        # Let accelerator.handle data distribution automatically
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  collate_fn=collate_fn, num_workers=num_workers,
                                  pin_memory=True, persistent_workers=num_workers > 0,
                                  prefetch_factor=prefetch)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                collate_fn=collate_fn, num_workers=num_workers,
                                pin_memory=True, persistent_workers=num_workers > 0,
                                prefetch_factor=prefetch)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                 collate_fn=collate_fn, num_workers=num_workers,
                                 pin_memory=True, persistent_workers=num_workers > 0,
                                 prefetch_factor=prefetch)
    else:
        num_workers = config['data'].get('num_workers', 4)
        prefetch = config['data'].get('prefetch_factor', 2) if num_workers > 0 else None

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  collate_fn=collate_fn, num_workers=num_workers,
                                  pin_memory=True, persistent_workers=num_workers > 0,
                                  prefetch_factor=prefetch)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                collate_fn=collate_fn, num_workers=num_workers,
                                pin_memory=True, persistent_workers=num_workers > 0,
                                prefetch_factor=prefetch)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                 collate_fn=collate_fn, num_workers=num_workers,
                                 pin_memory=True, persistent_workers=num_workers > 0,
                                 prefetch_factor=prefetch)

    return train_loader, val_loader, test_loader
