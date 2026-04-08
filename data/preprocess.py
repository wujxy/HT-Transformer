"""
Preprocessing: tokenize all H5 events and save to disk as HDF5 chunked storage.

Usage:
    python RunModule.py --config configs/default.yaml --Preprocess

This reads raw H5 files, performs full tokenization (WP + CD aggregation),
and saves results as a single HDF5 file with chunked datasets for O(1) random access.

Training will auto-detect the preprocessed data and skip tokenization.

Storage layout (data.h5):
    /metadata/cd_unit_vecs  (npix, 3)     shared across all events
    /metadata/cd_pixel_ids  (npix,)        shared
    /cd/stats               (N, npix, 4)   chunked per event
    /cd/time_bins           (N, npix, 32)  chunked per event, gzip
    /cd/mask                (N, npix)      chunked per event
    /wp/tokens_flat         (total*5,)     flat concatenated WP tokens
    /wp/offsets             (N, 2)         [start, n_hits] per event
    /labels/u1              (N, 3)
    /labels/u2              (N, 3)
    /labels/p1              (N, 3)
    /labels/p2              (N, 3)
"""

import os
import json
import hashlib
import numpy as np
import torch
import h5py
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
from loguru import logger
from tqdm import tqdm

from data.dataset import (
    H5EndpointDataset, collate_fn, discover_h5_files,
)
from data.augmentation import random_rotation_matrix


def identity_collate_fn(batch):
    """Identity collate for preprocessing: return events as-is without batching."""
    return batch


def _config_hash(config: dict) -> str:
    """Simple hash of config keys that affect tokenization."""
    relevant = json.dumps({
        'nside': config['data'].get('nside', 8),
        'num_time_bins': config['data'].get('num_time_bins', 32),
        't_max': config['data'].get('t_max', 800.0),
    }, sort_keys=True)
    return hashlib.md5(relevant.encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------
# HDF5 write helpers
# ---------------------------------------------------------------------------

def _write_events_to_hdf5(h5f, tokenized_events, start_idx, npix, num_time_bins):
    """Write a list of tokenized event dicts into an HDF5 file.

    Args:
        h5f: open h5py.File (write mode)
        tokenized_events: list of dicts from H5EndpointDataset.__getitem__
        start_idx: global event index to start writing at
        npix: number of HEALPix pixels
        num_time_bins: number of CD time bins
    """
    n = len(tokenized_events)
    end_idx = start_idx + n

    # Collect arrays for batch write
    cd_stats = np.zeros((n, npix, 4), dtype=np.float32)
    cd_time_bins = np.zeros((n, npix, num_time_bins), dtype=np.float32)
    cd_mask = np.zeros((n, npix), dtype=bool)
    u1 = np.zeros((n, 3), dtype=np.float32)
    u2 = np.zeros((n, 3), dtype=np.float32)
    p1 = np.zeros((n, 3), dtype=np.float32)
    p2 = np.zeros((n, 3), dtype=np.float32)

    wp_flat_parts = []
    wp_offsets = np.zeros((n, 2), dtype=np.int64)
    wp_flat_offset = int(h5f['wp/tokens_flat'].shape[0]) if 'wp/tokens_flat' in h5f else 0

    for i, ev in enumerate(tokenized_events):
        # CD data (numpy)
        cd_stats[i] = ev['cd_stats'].numpy() if isinstance(ev['cd_stats'], torch.Tensor) else ev['cd_stats']
        cd_time_bins[i] = ev['cd_time_bins'].numpy() if isinstance(ev['cd_time_bins'], torch.Tensor) else ev['cd_time_bins']
        cd_mask[i] = ev['cd_mask'].numpy() if isinstance(ev['cd_mask'], torch.Tensor) else ev['cd_mask']

        # Labels
        for field, arr in [('u1', u1), ('u2', u2), ('p1', p1), ('p2', p2)]:
            val = ev[field]
            arr[i] = val.numpy() if isinstance(val, torch.Tensor) else val

        # WP tokens (variable length)
        wp_tok = ev['wp_tokens']
        wp_tok_np = wp_tok.numpy() if isinstance(wp_tok, torch.Tensor) else wp_tok
        n_wp = wp_tok_np.shape[0]
        wp_offsets[i] = [wp_flat_offset // 5 if wp_tok_np.shape[-1] == 5 else wp_flat_offset, n_wp]

        if n_wp > 0:
            flat = wp_tok_np.reshape(-1).astype(np.float32)
            wp_flat_parts.append(flat)
            wp_flat_offset += flat.size
        # else: offset stays, n_wp = 0

    # Batch write fixed-shape datasets
    h5f['cd/stats'][start_idx:end_idx] = cd_stats
    h5f['cd/time_bins'][start_idx:end_idx] = cd_time_bins
    h5f['cd/mask'][start_idx:end_idx] = cd_mask
    h5f['labels/u1'][start_idx:end_idx] = u1
    h5f['labels/u2'][start_idx:end_idx] = u2
    h5f['labels/p1'][start_idx:end_idx] = p1
    h5f['labels/p2'][start_idx:end_idx] = p2

    # Append WP flat tokens and write offsets
    if wp_flat_parts:
        all_flat = np.concatenate(wp_flat_parts)
        old_size = h5f['wp/tokens_flat'].shape[0]
        h5f['wp/tokens_flat'].resize(old_size + all_flat.size, axis=0)
        h5f['wp/tokens_flat'][old_size:old_size + all_flat.size] = all_flat

    # Recompute WP offsets based on actual flat position
    wp_offset_pos = 0
    for i, ev in enumerate(tokenized_events):
        wp_tok = ev['wp_tokens']
        wp_tok_np = wp_tok.numpy() if isinstance(wp_tok, torch.Tensor) else wp_tok
        n_wp = wp_tok_np.shape[0]
        if n_wp > 0:
            wp_offsets[i] = [wp_offset_pos, n_wp]
            wp_offset_pos += n_wp
        else:
            wp_offsets[i] = [wp_offset_pos, 0]

    h5f['wp/offsets'][start_idx:end_idx] = wp_offsets


def _create_hdf5_datasets(h5f, total_events, npix, num_time_bins):
    """Create HDF5 dataset placeholders for all fields."""
    # Metadata (shared, written once)
    g_meta = h5f.create_group('metadata')

    # CD fixed-shape datasets: chunked per event for O(1) random access
    g_cd = h5f.create_group('cd')
    g_cd.create_dataset(
        'stats', shape=(total_events, npix, 4), dtype='float32',
        chunks=(1, npix, 4), compression='gzip', compression_opts=1,
    )
    g_cd.create_dataset(
        'time_bins', shape=(total_events, npix, num_time_bins), dtype='float32',
        chunks=(1, npix, num_time_bins), compression='gzip', compression_opts=1,
    )
    g_cd.create_dataset(
        'mask', shape=(total_events, npix), dtype='bool',
        chunks=(1, npix),
    )

    # WP variable-length: flat array + offset index
    g_wp = h5f.create_group('wp')
    g_wp.create_dataset(
        'tokens_flat', shape=(0,), dtype='float32',
        maxshape=(None,), chunks=(1024 * 1024,),  # 1M float32 chunks
    )
    g_wp.create_dataset(
        'offsets', shape=(total_events, 2), dtype='int64',
        chunks=(256, 2),
    )

    # Labels
    g_labels = h5f.create_group('labels')
    for name in ('u1', 'u2', 'p1', 'p2'):
        g_labels.create_dataset(
            name, shape=(total_events, 3), dtype='float32',
            chunks=(256, 3),
        )


# ---------------------------------------------------------------------------
# Preprocessing entry point
# ---------------------------------------------------------------------------

def preprocess(config: dict):
    """Preprocess all H5 events into tokenized HDF5 chunked storage."""
    data_cfg = config['data']

    # Setup dirs
    mission = config.get('mission_name', 'ht_transformer_v1')
    output_base = config.get('output_path', 'output')
    preprocessed_dir = os.path.join(output_base, mission, 'preprocessed')
    os.makedirs(preprocessed_dir, exist_ok=True)

    manifest_path = os.path.join(preprocessed_dir, 'manifest.json')
    h5_path = os.path.join(preprocessed_dir, 'data.h5')

    # Check existing cache
    if os.path.exists(h5_path) and os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            existing_manifest = json.load(f)
        existing_repr = existing_manifest.get('cd_representation', 'active_patch')
        if existing_repr != 'dense_healpix':
            raise RuntimeError(
                f"Existing preprocessed cache is old '{existing_repr}' format. "
                f"Stage B requires 'dense_healpix' format. "
                f"Please delete {preprocessed_dir} and re-run preprocessing."
            )
        logger.info(f"Preprocessed data already exists at {preprocessed_dir}")
        logger.info("Delete the directory to re-run preprocessing.")
        return

    # --- Initialize geometry + HEALPix ---
    from geometry.detector_geometry import DualPMTPositionLookup
    from geometry.healpix_mapper import HEALPixMapper

    geometry = DualPMTPositionLookup(
        data_cfg['geometry_cd'], data_cfg['geometry_wp'])

    cd_unit_vecs = geometry.cd_position_array.copy()
    norms = np.linalg.norm(cd_unit_vecs, axis=1, keepdims=True)
    valid = (norms.squeeze() > 0)
    cd_unit_vecs[valid] = cd_unit_vecs[valid] / norms[valid]

    healpix = HEALPixMapper(
        nside=data_cfg['nside'],
        cd_unit_vecs=cd_unit_vecs,
    )
    healpix.build_knn_adjacency(k=config['model']['cd_knn_k'])

    npix = 12 * data_cfg['nside'] * data_cfg['nside']
    num_time_bins = data_cfg['num_time_bins']

    # --- Discover H5 files ---
    h5_files = discover_h5_files(data_cfg['h5_path'])
    if len(h5_files) == 0:
        raise FileNotFoundError(f"No H5 files found for: {data_cfg['h5_path']}")
    logger.info(f"Discovered {len(h5_files)} H5 file(s)")

    # Count events per file
    km = data_cfg['h5_key_map']
    events_per_file = []
    total_events = 0
    for fp in h5_files:
        with h5py.File(fp, 'r') as f:
            n = f[km['nhits']].shape[0]
            events_per_file.append(n)
            total_events += n
    logger.info(f"Total events in H5 files: {total_events}")

    # Apply max_events limit
    max_events = data_cfg.get('max_events', -1)
    if max_events > 0 and max_events < total_events:
        logger.info(f"Limiting to {max_events} events (out of {total_events})")
        total_events = max_events
        # Truncate events_per_file to match the limited total
        truncated = []
        remaining = max_events
        for n in events_per_file:
            take = min(n, remaining)
            truncated.append(take)
            remaining -= take
            if remaining <= 0:
                break
        events_per_file = truncated
    else:
        logger.info(f"Processing all {total_events} events")

    # --- Create dataset with selected events (no split) ---
    all_indices = np.arange(total_events)
    dataset = H5EndpointDataset(
        h5_files, config, geometry, healpix,
        indices=all_indices, is_train=False,
        events_per_file=events_per_file,
    )

    # --- Tokenize and save ---
    expand_times = config.get('augmentation', {}).get('rotation_expand_times', 0)
    total_with_aug = total_events * (1 + expand_times)

    # Create HDF5 file with pre-allocated datasets
    logger.info(f"Creating HDF5 file: {h5_path} ({total_with_aug} events)")
    with h5py.File(h5_path, 'w') as h5f:
        _create_hdf5_datasets(h5f, total_with_aug, npix, num_time_bins)

        # Write shared metadata
        h5f['metadata'].create_dataset(
            'cd_unit_vecs', data=healpix.cd_unit_vecs.astype(np.float32))
        h5f['metadata'].create_dataset(
            'cd_pixel_ids', data=np.arange(npix, dtype=np.int64))
        h5f['metadata'].attrs['nside'] = data_cfg['nside']
        h5f['metadata'].attrs['npix'] = npix
        h5f['metadata'].attrs['num_time_bins'] = num_time_bins

        # --- Phase 1: Write original events ---
        num_workers = data_cfg.get('num_workers', 4)
        prefetch = data_cfg.get('prefetch_factor', 2) if num_workers > 0 else None

        loader = DataLoader(
            dataset,
            batch_size=data_cfg.get('preprocess_batch_size', 512),
            shuffle=False,
            num_workers=num_workers,
            collate_fn=identity_collate_fn,
            pin_memory=False,
            prefetch_factor=prefetch,
            persistent_workers=num_workers > 0,
        )

        logger.info(f"Tokenizing original {total_events} events...")
        event_idx = 0
        for batch_data in tqdm(loader, desc="Tokenizing"):
            n = len(batch_data)
            _write_events_to_hdf5(h5f, batch_data, event_idx, npix, num_time_bins)
            event_idx += n

        # --- Phase 2: Rotation augmentation expansions ---
        rotation_matrices = []
        for exp in range(expand_times):
            R = random_rotation_matrix()
            rotation_matrices.append(R)
            logger.info(f"Rotation expansion {exp+1}/{expand_times}")

            rot_dataset = H5EndpointDataset(
                h5_files, config, geometry, healpix,
                indices=all_indices, is_train=False,
                events_per_file=events_per_file,
                apply_rotation_aug=True,
            )

            rot_loader = DataLoader(
                rot_dataset,
                batch_size=data_cfg.get('preprocess_batch_size', 512),
                shuffle=False,
                num_workers=num_workers,
                collate_fn=identity_collate_fn,
                pin_memory=False,
                prefetch_factor=prefetch,
                persistent_workers=num_workers > 0,
            )

            for batch_data in tqdm(rot_loader, desc=f"Rotation {exp+1}/{expand_times}"):
                n = len(batch_data)
                _write_events_to_hdf5(h5f, batch_data, event_idx, npix, num_time_bins)
                event_idx += n

    # --- Save manifest ---
    manifest = {
        'original_events': total_events,
        'expand_times': expand_times,
        'total_events': total_with_aug,
        'batch_size': data_cfg.get('preprocess_batch_size', 512),
        'config_hash': _config_hash(config),
        'cd_representation': 'dense_healpix',
        'format': 'hdf5',
        'nside': data_cfg['nside'],
        'npix': npix,
        'num_time_bins': num_time_bins,
        'h5_file': 'data.h5',
        'rotation_matrices': [R.tolist() for R in rotation_matrices],
    }

    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Preprocessing complete: {h5_path}")
    logger.info(f"  Original: {total_events}, expand_times={expand_times}, "
                f"total={total_with_aug}")
    logger.info(f"Manifest saved: {manifest_path}")


# ---------------------------------------------------------------------------
# HDF5-backed Dataset for training
# ---------------------------------------------------------------------------

class PreprocessedDataset(Dataset):
    """
    Dataset that reads pre-tokenized events from HDF5 chunked storage.

    Each __getitem__ reads exactly one HDF5 chunk per field (~132 KB total),
    instead of loading an entire .pt batch file (~83 MB).

    Memory usage is bounded by the HDF5 chunk cache (rdcc_nbytes, default 256 MB).
    """

    def __init__(self, h5_path: str, indices: Optional[np.ndarray] = None,
                 cache_nbytes: int = 256 * 1024 * 1024):
        """
        Args:
            h5_path: path to data.h5 file
            indices: event-level indices to use (for train/val/test split)
            cache_nbytes: max HDF5 chunk cache size in bytes (default 256 MB)
        """
        self._h5_path = h5_path
        self._file = h5py.File(h5_path, 'r', rdcc_nbytes=cache_nbytes, rdcc_w0=0.5)

        # Load shared metadata once (tiny, ~9 KB)
        self._cd_unit_vecs = torch.from_numpy(
            np.array(self._file['metadata/cd_unit_vecs']))  # (npix, 3)
        self._npix = self._file['metadata'].attrs['npix']

        # WP offset index (tiny, ~3.8 MB for 240K events)
        self._wp_offsets = self._file['wp/offsets'][:]  # (N, 2) int64

        # Total events
        self._total_events = self._file['cd/stats'].shape[0]

        # Apply indices
        if indices is not None:
            self._indices = indices
        else:
            self._indices = np.arange(self._total_events)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        global_idx = int(self._indices[idx])
        f = self._file

        # --- CD data: read one chunk per field ---
        cd_stats = torch.from_numpy(np.array(f['cd/stats'][global_idx]))        # (npix, 4)
        cd_time_bins = torch.from_numpy(np.array(f['cd/time_bins'][global_idx]))# (npix, 32)
        cd_mask = torch.from_numpy(np.array(f['cd/mask'][global_idx]))          # (npix,)

        # --- WP tokens: variable-length via offset index ---
        wp_start, n_wp = self._wp_offsets[global_idx]
        n_wp = int(n_wp)
        wp_start = int(wp_start)

        if n_wp > 0:
            wp_flat = np.array(f['wp/tokens_flat'][wp_start:wp_start + n_wp * 5])
            wp_tokens = torch.from_numpy(wp_flat.reshape(n_wp, 5))
            wp_mask = torch.zeros(n_wp, dtype=torch.bool)
            wp_unit_vecs = wp_tokens[:, :3]   # view, no copy
            wp_times = wp_tokens[:, 4]         # view, no copy
        else:
            wp_tokens = torch.zeros((0, 5), dtype=torch.float32)
            wp_mask = torch.ones(0, dtype=torch.bool)
            wp_unit_vecs = torch.zeros((0, 3), dtype=torch.float32)
            wp_times = torch.zeros(0, dtype=torch.float32)

        # --- Labels ---
        u1 = torch.from_numpy(np.array(f['labels/u1'][global_idx]))
        u2 = torch.from_numpy(np.array(f['labels/u2'][global_idx]))
        p1 = torch.from_numpy(np.array(f['labels/p1'][global_idx]))
        p2 = torch.from_numpy(np.array(f['labels/p2'][global_idx]))

        return {
            'wp_tokens': wp_tokens,
            'wp_mask': wp_mask,
            'wp_unit_vecs': wp_unit_vecs,
            'wp_times': wp_times,
            # CD: shared unit vecs + per-event data
            'cd_unit_vecs': self._cd_unit_vecs,    # shared tensor, no per-event storage
            'cd_stats': cd_stats,
            'cd_time_bins': cd_time_bins,
            'cd_mask': cd_mask,
            'cd_times_mean': cd_stats[:, 3],       # derived view
            'cd_pixel_ids': torch.arange(self._npix, dtype=torch.long),  # generated
            # Labels
            'u1': u1, 'u2': u2, 'p1': p1, 'p2': p2,
        }

    def __del__(self):
        if hasattr(self, '_file') and self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def create_preprocessed_dataloaders(
    config: dict, preprocessed_dir: str
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders from preprocessed HDF5 data."""
    data_cfg = config['data']
    train_cfg = config['train']

    # Load manifest
    manifest_path = os.path.join(preprocessed_dir, 'manifest.json')
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    # Check format
    cd_repr = manifest.get('cd_representation', 'active_patch')
    if cd_repr != 'dense_healpix':
        raise RuntimeError(
            f"Preprocessed cache is '{cd_repr}' format, "
            f"but Stage B model requires 'dense_healpix'. "
            f"Please delete {preprocessed_dir} and re-run preprocessing."
        )

    # Detect HDF5 vs old .pt format
    fmt = manifest.get('format', 'pt')
    h5_path = os.path.join(preprocessed_dir, 'data.h5')

    if fmt == 'hdf5' and os.path.exists(h5_path):
        return _create_hdf5_dataloaders(config, h5_path, manifest)
    else:
        # Old .pt format — prompt user to re-preprocess
        if any(fn.startswith('batch_') and fn.endswith('.pt')
               for fn in os.listdir(preprocessed_dir)):
            logger.warning(
                "Found old .pt format preprocessed data. "
                "Please delete the preprocessed directory and re-run --Preprocess "
                "to generate the new HDF5 format for better performance."
            )
            raise RuntimeError(
                "Old .pt format detected. Please re-run preprocessing: "
                f"rm -rf {preprocessed_dir} && python -m cli.run --config configs/default.yaml --Preprocess"
            )
        raise FileNotFoundError(f"No preprocessed data found in {preprocessed_dir}")


def _create_hdf5_dataloaders(
    config: dict, h5_path: str, manifest: dict
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders backed by HDF5 chunked storage."""
    data_cfg = config['data']
    train_cfg = config['train']

    original_count = manifest.get('original_events', manifest['total_events'])
    expand_times = manifest.get('expand_times', 0)
    total_events = manifest['total_events']
    logger.info(f"Preprocessed HDF5 data: {total_events} events from {h5_path} "
                f"(original={original_count}, expand_times={expand_times})")

    # Split only the ORIGINAL event indices, then expand each split
    ratios = [data_cfg['train_ratio'], data_cfg['val_ratio'], data_cfg['test_ratio']]
    orig_indices = np.arange(original_count)
    np.random.seed(train_cfg.get('seed', 42))
    np.random.shuffle(orig_indices)

    n_train = int(ratios[0] * original_count)
    n_val = int(ratios[1] * original_count)

    train_orig = orig_indices[:n_train]
    val_orig = orig_indices[n_train:n_train + n_val]
    test_orig = orig_indices[n_train + n_val:]

    # Expand indices: each split gets its original events + all rotated copies
    def _expand_split(orig_idx, original_count, expand_times):
        result = orig_idx.tolist()
        for exp in range(expand_times):
            offset = original_count * (exp + 1)
            result.extend((orig_idx + offset).tolist())
        return np.array(result, dtype=np.int64)

    train_idx = _expand_split(train_orig, original_count, expand_times)
    val_idx = _expand_split(val_orig, original_count, expand_times)
    test_idx = _expand_split(test_orig, original_count, expand_times)

    logger.info(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    batch_size = train_cfg['batch_size']
    cache_nbytes = data_cfg.get('h5_cache_nbytes', 256 * 1024 * 1024)

    train_ds = PreprocessedDataset(h5_path, train_idx, cache_nbytes=cache_nbytes)
    val_ds = PreprocessedDataset(h5_path, val_idx, cache_nbytes=cache_nbytes)
    test_ds = PreprocessedDataset(h5_path, test_idx, cache_nbytes=cache_nbytes)

    # DataLoader config: disable num_workers when using accelerate
    use_accelerate = train_cfg.get('use_accelerate', False)
    if use_accelerate:
        num_workers = 0
        prefetch = None
    else:
        num_workers = data_cfg.get('num_workers', 4)
        prefetch = data_cfg.get('prefetch_factor', 2) if num_workers > 0 else None

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
