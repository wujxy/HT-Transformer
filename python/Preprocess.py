"""
Preprocessing: tokenize all H5 events and save to disk as .pt batch files.

Usage:
    python RunModule.py --config configs/default.yaml --Preprocess

This reads raw H5 files, performs full tokenization (WP + CD aggregation),
and saves results as .pt files in {output_dir}/{mission_name}/preprocessed/.

Training will auto-detect the preprocessed data and skip tokenization.
"""

import os
import json
import bisect
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
from loguru import logger
from tqdm import tqdm

from DataLoader import (
    H5EndpointDataset, collate_fn, discover_h5_files,
)
from Augmentation import random_rotation_matrix


def identity_collate_fn(batch):
    """Identity collate for preprocessing: return events as-is without batching."""
    return batch


def preprocess(config: dict):
    """Preprocess all H5 events into tokenized .pt batch files."""
    data_cfg = config['data']
    train_cfg = config['train']

    # Setup dirs
    mission = config.get('mission_name', 'ht_transformer_v1')
    output_base = config.get('output_path', 'output')
    preprocessed_dir = os.path.join(output_base, mission, 'preprocessed')
    os.makedirs(preprocessed_dir, exist_ok=True)

    manifest_path = os.path.join(preprocessed_dir, 'manifest.json')
    if os.path.exists(manifest_path):
        logger.info(f"Preprocessed data already exists at {preprocessed_dir}")
        logger.info("Delete the directory to re-run preprocessing.")
        return

    # --- Initialize geometry + HEALPix ---
    from Geometry import DualPMTPositionLookup
    from HEALPix import HEALPixMapper

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

    # --- Discover H5 files ---
    h5_path = data_cfg['h5_path']
    h5_files = discover_h5_files(h5_path)
    if len(h5_files) == 0:
        raise FileNotFoundError(f"No H5 files found for: {h5_path}")
    logger.info(f"Discovered {len(h5_files)} H5 file(s)")

    # Count events per file
    km = data_cfg['h5_key_map']
    events_per_file = []
    total_events = 0
    for fp in h5_files:
        import h5py
        with h5py.File(fp, 'r') as f:
            n = f[km['nhits']].shape[0]
            events_per_file.append(n)
            total_events += n
    logger.info(f"Total events: {total_events}")

    # --- Create dataset with ALL events (no split) ---
    all_indices = np.arange(total_events)
    dataset = H5EndpointDataset(
        h5_files, config, geometry, healpix,
        indices=all_indices, is_train=False,
        events_per_file=events_per_file,
    )

    # --- Tokenize and save with multi-worker DataLoader ---
    batch_size = data_cfg.get('preprocess_batch_size', 512)
    batch_idx = 0
    events_per_batch = []

    # Get num_workers config for parallel data loading
    num_workers = data_cfg.get('num_workers', 4)
    prefetch = data_cfg.get('prefetch_factor', 2) if num_workers > 0 else None

    # Create DataLoader with identity collate for parallel processing
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=identity_collate_fn,
        pin_memory=False,  # Saving to disk, no need for pin_memory
        prefetch_factor=prefetch,
        persistent_workers=num_workers > 0
    )

    logger.info(f"Starting preprocessing: {total_events} events, batch_size={batch_size}, num_workers={num_workers}")
    for batch_data in tqdm(loader, desc="Tokenizing"):
        save_path = os.path.join(preprocessed_dir, f"batch_{batch_idx:04d}.pt")
        torch.save(batch_data, save_path)
        events_per_batch.append(len(batch_data))
        batch_idx += 1

    # --- Save manifest ---
    expand_times = config.get('augmentation', {}).get('rotation_expand_times', 0)
    rotation_matrices = []

    manifest = {
        'original_events': total_events,
        'expand_times': expand_times,
        'total_events': total_events * (1 + expand_times),
        'batch_size': batch_size,
        'num_files': batch_idx,
        'events_per_file': events_per_batch,
        'config_hash': _config_hash(config),
    }

    # --- Rotation augmentation expansion ---
    # Each expansion applies ONE fixed random SO(3) rotation to ALL events,
    # re-tokenizing from scratch (HEALPix mapping changes naturally).
    for exp in range(expand_times):
        R = random_rotation_matrix()
        rotation_matrices.append(R)
        logger.info(f"Rotation expansion {exp+1}/{expand_times}: applying random SO(3) rotation")

        rot_dataset = H5EndpointDataset(
            h5_files, config, geometry, healpix,
            indices=all_indices, is_train=False,
            events_per_file=events_per_file,
            apply_rotation_aug=True,      # triggers per-event rotation inside __getitem__
        )

        # Create DataLoader for rotated data
        rot_loader = DataLoader(
            rot_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=identity_collate_fn,
            pin_memory=False,
            prefetch_factor=prefetch,
            persistent_workers=num_workers > 0
        )

        for batch_data in tqdm(rot_loader, desc=f"Rotation {exp+1}/{expand_times}"):
            save_path = os.path.join(preprocessed_dir, f"batch_{batch_idx:04d}.pt")
            torch.save(batch_data, save_path)
            events_per_batch.append(len(batch_data))
            batch_idx += 1

    # Update manifest with final info
    manifest['num_files'] = batch_idx
    manifest['events_per_file'] = events_per_batch
    manifest['rotation_matrices'] = [R.tolist() for R in rotation_matrices]

    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Preprocessing complete: {batch_idx} files saved to {preprocessed_dir}")
    logger.info(f"  Original: {total_events}, expand_times={expand_times}, "
                f"total={manifest['total_events']}")
    logger.info(f"Manifest saved: {manifest_path}")


def _config_hash(config: dict) -> str:
    """Simple hash of config keys that affect tokenization."""
    import hashlib
    relevant = json.dumps({
        'nside': config['data'].get('nside', 8),
        'num_time_bins': config['data'].get('num_time_bins', 32),
        't_max': config['data'].get('t_max', 800.0),
    }, sort_keys=True)
    return hashlib.md5(relevant.encode()).hexdigest()[:8]


class PreprocessedDataset(Dataset):
    """
    Dataset that reads pre-tokenized events from .pt batch files.

    Each .pt file contains a list of dicts (same format as H5EndpointDataset.__getitem__).
    """

    def __init__(self, preprocessed_dir: str, indices: Optional[np.ndarray] = None):
        """
        Args:
            preprocessed_dir: directory containing batch_XXXX.pt files and manifest.json
            indices: event-level indices to use (for train/val/test split)
        """
        self._dir = preprocessed_dir

        # Load manifest
        manifest_path = os.path.join(preprocessed_dir, 'manifest.json')
        with open(manifest_path, 'r') as f:
            self._manifest = json.load(f)

        self._total_events = self._manifest['total_events']
        self._events_per_file = self._manifest['events_per_file']

        # Build cumulative sizes for file-level indexing
        self._cumulative_sizes = np.cumsum([0] + self._events_per_file)

        # Apply indices
        if indices is not None:
            self._indices = indices
        else:
            self._indices = np.arange(self._total_events)

        # File cache (LRU)
        self._file_cache: Dict[int, list] = {}
        self._cache_max = max(16, len(self._events_per_file))  # cache all batch files

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        global_idx = self._indices[idx]

        # Find which batch file this event belongs to
        file_idx = bisect.bisect_right(self._cumulative_sizes, global_idx) - 1
        local_idx = global_idx - self._cumulative_sizes[file_idx]

        # Load batch file (with cache)
        if file_idx not in self._file_cache:
            if len(self._file_cache) >= self._cache_max:
                # Evict oldest
                oldest_key = next(iter(self._file_cache))
                del self._file_cache[oldest_key]
            batch_path = os.path.join(self._dir, f"batch_{file_idx:04d}.pt")
            self._file_cache[file_idx] = torch.load(batch_path, weights_only=False)

        return self._file_cache[file_idx][local_idx]


def create_preprocessed_dataloaders(
    config: dict, preprocessed_dir: str
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders from preprocessed .pt files."""
    data_cfg = config['data']
    train_cfg = config['train']

    # Load manifest to get total events
    manifest_path = os.path.join(preprocessed_dir, 'manifest.json')
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    original_count = manifest.get('original_events', manifest['total_events'])
    expand_times = manifest.get('expand_times', 0)
    total_events = manifest['total_events']
    logger.info(f"Preprocessed data: {total_events} events from {preprocessed_dir} "
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

    train_ds = PreprocessedDataset(preprocessed_dir, train_idx)
    val_ds = PreprocessedDataset(preprocessed_dir, val_idx)
    test_ds = PreprocessedDataset(preprocessed_dir, test_idx)

    # Disable num_workers when using accelerate to avoid fork/deadlock issues
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
