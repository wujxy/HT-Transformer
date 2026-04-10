"""
Preprocessing: tokenize H5 events → split-level HDF5 files (v3).

Offline build produces:
    output/{mission}/preprocessed/
    ├── meta.json
    ├── train.h5
    ├── val.h5
    └── test.h5

Each HDF5 file layout (v3 sparse PMT token):
    /cd_tokens      (N, K_cd, 10)   float16  [ux,uy,uz,q_sum,q_max,n_hits,t_first,t_mean,t_late,t_span]
    /cd_mask        (N, K_cd)       bool     True = padding
    /wp_offsets     (N+1,)          int64    cumulative
    /wp_tokens_flat (total_wp*5,)   float16
    /labels         (N, 12)         float16  [u1(3), u2(3), p1(3), p2(3)]
"""

import os
import json
import hashlib
import functools
import numpy as np
import torch
import h5py
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
from loguru import logger
from tqdm import tqdm

from data.dataset import (
    H5EndpointDataset, discover_h5_files,
)


def _config_hash(config: dict) -> str:
    relevant = json.dumps({
        't_max': config['data'].get('t_max', 800.0),
        'cd_max_tokens': config['model'].get('cd_max_tokens', 640),
    }, sort_keys=True)
    return hashlib.md5(relevant.encode()).hexdigest()[:8]


def _identity_collate(batch):
    return batch


# ---------------------------------------------------------------------------
# Write helper: tokenized events → one split HDF5 file (v3 format)
# ---------------------------------------------------------------------------

def _write_split_hdf5(h5_path, tokenized_events, K_cd):
    """Write a list of tokenized event dicts into one split HDF5 file (v3 format).

    Stores: cd_tokens, cd_mask, wp tokens, labels.
    All float data stored as float16.
    """
    N = len(tokenized_events)
    if N == 0:
        return 0

    cd_tokens = np.zeros((N, K_cd, 10), dtype=np.float16)
    cd_mask = np.ones((N, K_cd), dtype=bool)
    labels = np.zeros((N, 12), dtype=np.float16)

    wp_flat_parts = []
    wp_counts = np.zeros(N, dtype=np.int64)

    for i, ev in enumerate(tokenized_events):
        # CD v3: cd_tokens + cd_mask
        ct = ev['cd_tokens']
        cd_tokens[i] = ct.numpy().astype(np.float16) if isinstance(ct, torch.Tensor) else ct.astype(np.float16)
        cm = ev['cd_mask']
        cd_mask[i] = cm.numpy() if isinstance(cm, torch.Tensor) else cm

        # Labels: [u1, u2, p1, p2]
        for j, key in enumerate(['u1', 'u2', 'p1', 'p2']):
            v = ev[key]
            labels[i, j*3:(j+1)*3] = v.numpy().astype(np.float16) if isinstance(v, torch.Tensor) else v.astype(np.float16)

        # WP tokens
        wp = ev['wp_tokens']
        wp_np = wp.numpy() if isinstance(wp, torch.Tensor) else wp
        n_wp = wp_np.shape[0]
        wp_counts[i] = n_wp
        if n_wp > 0:
            wp_flat_parts.append(wp_np.reshape(-1).astype(np.float16))

    # Cumulative offsets (N+1,)
    wp_offsets = np.zeros(N + 1, dtype=np.int64)
    np.cumsum(wp_counts, out=wp_offsets[1:])

    # Write HDF5 with maxshape for resize support (needed for augmentation append)
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('cd_tokens', data=cd_tokens, dtype='float16',
                         maxshape=(None, K_cd, 10))
        f.create_dataset('cd_mask', data=cd_mask, dtype='bool',
                         maxshape=(None, K_cd))
        f.create_dataset('labels', data=labels, dtype='float16',
                         maxshape=(None, 12))
        f.create_dataset('wp_offsets', data=wp_offsets, dtype='int64',
                         maxshape=(None,))
        wp_flat = np.concatenate(wp_flat_parts) if wp_flat_parts else np.zeros(0, dtype=np.float16)
        f.create_dataset('wp_tokens_flat', data=wp_flat, dtype='float16',
                         maxshape=(None,))

    return N


# ---------------------------------------------------------------------------
# Append helper: add more events to an existing split HDF5 file (v3 format)
# ---------------------------------------------------------------------------

def _append_to_split_hdf5(h5_path, tokenized_events, K_cd):
    """Append more events to an existing split HDF5 file (v3 format)."""
    if not tokenized_events:
        return
    N = len(tokenized_events)

    cd_tokens = np.zeros((N, K_cd, 10), dtype=np.float16)
    cd_mask = np.ones((N, K_cd), dtype=bool)
    labels = np.zeros((N, 12), dtype=np.float16)
    wp_flat_parts = []
    wp_counts = np.zeros(N, dtype=np.int64)

    for i, ev in enumerate(tokenized_events):
        ct = ev['cd_tokens']
        cd_tokens[i] = ct.numpy().astype(np.float16) if isinstance(ct, torch.Tensor) else ct.astype(np.float16)
        cm = ev['cd_mask']
        cd_mask[i] = cm.numpy() if isinstance(cm, torch.Tensor) else cm
        for j, key in enumerate(['u1', 'u2', 'p1', 'p2']):
            v = ev[key]
            labels[i, j*3:(j+1)*3] = v.numpy().astype(np.float16) if isinstance(v, torch.Tensor) else v.astype(np.float16)
        wp = ev['wp_tokens']
        wp_np = wp.numpy() if isinstance(wp, torch.Tensor) else wp
        n_wp = wp_np.shape[0]
        wp_counts[i] = n_wp
        if n_wp > 0:
            wp_flat_parts.append(wp_np.reshape(-1).astype(np.float16))

    new_offsets = np.zeros(N + 1, dtype=np.int64)
    np.cumsum(wp_counts, out=new_offsets[1:])

    with h5py.File(h5_path, 'a') as f:
        old_N = f['cd_tokens'].shape[0]
        # Resize and append fixed-shape datasets
        f['cd_tokens'].resize(old_N + N, axis=0)
        f['cd_tokens'][old_N:] = cd_tokens
        f['cd_mask'].resize(old_N + N, axis=0)
        f['cd_mask'][old_N:] = cd_mask
        f['labels'].resize(old_N + N, axis=0)
        f['labels'][old_N:] = labels

        # Append WP tokens
        old_flat_size = f['wp_tokens_flat'].shape[0]
        if wp_flat_parts:
            new_flat = np.concatenate(wp_flat_parts)
            f['wp_tokens_flat'].resize(old_flat_size + new_flat.size, axis=0)
            f['wp_tokens_flat'][old_flat_size:] = new_flat

        # Rewrite offsets (need to merge old + new)
        old_offsets = f['wp_offsets'][:]
        total_wp_old = int(old_offsets[-1])
        new_offsets_shifted = new_offsets + total_wp_old
        merged = np.concatenate([old_offsets[:-1], new_offsets_shifted])
        del f['wp_offsets']
        f.create_dataset('wp_offsets', data=merged, dtype='int64')


# ---------------------------------------------------------------------------
# Preprocessing entry point
# ---------------------------------------------------------------------------

def preprocess(config: dict):
    """Build split-level HDF5 files from raw H5 events (v3 sparse PMT token format)."""
    data_cfg = config['data']
    model_cfg = config['model']

    mission = config.get('mission_name', 'ht_transformer_v3')
    output_base = config.get('output_path', 'output')
    preprocessed_dir = os.path.join(output_base, mission, 'preprocessed')
    os.makedirs(preprocessed_dir, exist_ok=True)

    meta_path = os.path.join(preprocessed_dir, 'meta.json')

    # Check existing
    if os.path.exists(meta_path):
        logger.info(f"Preprocessed data already exists at {preprocessed_dir}")
        logger.info("Delete the directory to re-run preprocessing.")
        return

    # --- Geometry ---
    from geometry.detector_geometry import DualPMTPositionLookup

    geometry = DualPMTPositionLookup(data_cfg['geometry_cd'], data_cfg['geometry_wp'])

    # v3 config
    K_cd = model_cfg.get('cd_max_tokens', 640)

    # --- Discover + count events ---
    h5_files = discover_h5_files(data_cfg['h5_path'])
    if len(h5_files) == 0:
        raise FileNotFoundError(f"No H5 files found for: {data_cfg['h5_path']}")
    logger.info(f"Discovered {len(h5_files)} H5 file(s)")

    km = data_cfg['h5_key_map']
    events_per_file = []
    total_events = 0
    for fp in h5_files:
        with h5py.File(fp, 'r') as f:
            n = f[km['nhits']].shape[0]
            events_per_file.append(n)
            total_events += n

    # Apply max_events limit
    max_events = data_cfg.get('max_events', -1)
    if max_events > 0 and max_events < total_events:
        logger.info(f"Limiting to {max_events} events (out of {total_events})")
        total_events = max_events
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

    # --- Split indices ---
    ratios = [data_cfg['train_ratio'], data_cfg['val_ratio'], data_cfg['test_ratio']]
    all_indices = np.arange(total_events)
    np.random.seed(config['train'].get('seed', 42))
    np.random.shuffle(all_indices)

    n_train = int(ratios[0] * total_events)
    n_val = int(ratios[1] * total_events)
    train_idx = all_indices[:n_train]
    val_idx = all_indices[n_train:n_train + n_val]
    test_idx = all_indices[n_train + n_val:]
    logger.info(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # --- Augmentation ---
    expand_times = config.get('augmentation', {}).get('rotation_expand_times', 0)

    # --- Tokenize and write each split ---
    num_workers = data_cfg.get('num_workers', 4)
    prefetch = data_cfg.get('prefetch_factor', 2) if num_workers > 0 else None
    batch_size = data_cfg.get('preprocess_batch_size', 512)

    def _tokenize_and_write(split_name, indices_list, h5_path, apply_rotation=False):
        """Tokenize events for a list of index arrays and write to one HDF5 file."""
        all_events = []
        for idx_arr in indices_list:
            ds = H5EndpointDataset(
                h5_files, config, geometry,
                indices=idx_arr, is_train=False,
                events_per_file=events_per_file,
                apply_rotation_aug=apply_rotation,
            )
            loader = DataLoader(
                ds, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, collate_fn=_identity_collate,
                pin_memory=False, prefetch_factor=prefetch,
                persistent_workers=False,
            )
            for batch_data in tqdm(loader, desc=f"Tokenizing {split_name}"):
                all_events.extend(batch_data)

        n = _write_split_hdf5(h5_path, all_events, K_cd)
        logger.info(f"  {split_name}: {n} events → {h5_path}")
        return n

    # Write train (original only first) — skip if already done
    train_h5_path = os.path.join(preprocessed_dir, 'train.h5')
    expected_train_total = len(train_idx) * (1 + expand_times)
    skip_train = False

    if os.path.exists(train_h5_path):
        with h5py.File(train_h5_path, 'r') as f:
            existing_n = f['labels'].shape[0]
        if existing_n == expected_train_total:
            logger.info(f"train.h5 exists with {existing_n} events (expected {expected_train_total}), skipping train tokenization")
            n_train_total = existing_n
            skip_train = True
        else:
            logger.warning(f"train.h5 exists but has {existing_n} events (expected {expected_train_total}), re-processing train")

    if not skip_train:
        n_train_total = _tokenize_and_write(
            'train', [train_idx],
            train_h5_path,
            apply_rotation=False,
        )
        # Append rotation-augmented copies to train.h5
        for exp in range(expand_times):
            logger.info(f"Rotation augmentation {exp+1}/{expand_times}...")
            aug_events = []
            ds = H5EndpointDataset(
                h5_files, config, geometry,
                indices=train_idx, is_train=False,
                events_per_file=events_per_file,
                apply_rotation_aug=True,
            )
            loader = DataLoader(
                ds, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, collate_fn=_identity_collate,
                pin_memory=False, prefetch_factor=prefetch,
                persistent_workers=False,
            )
            for batch_data in tqdm(loader, desc=f"Rotation {exp+1}/{expand_times}"):
                aug_events.extend(batch_data)

            _append_to_split_hdf5(train_h5_path, aug_events, K_cd)
            n_train_total += len(aug_events)
        if expand_times > 0:
            logger.info(f"  train (with aug): {n_train_total} events")

    # --- Val: original + rotation copies (same pattern as train) ---
    val_h5_path = os.path.join(preprocessed_dir, 'val.h5')
    n_val_total = _tokenize_and_write(
        'val', [val_idx],
        val_h5_path,
        apply_rotation=False,
    )
    for exp in range(expand_times):
        logger.info(f"Val rotation augmentation {exp+1}/{expand_times}...")
        aug_events = []
        ds = H5EndpointDataset(
            h5_files, config, geometry,
            indices=val_idx, is_train=False,
            events_per_file=events_per_file,
            apply_rotation_aug=True,
        )
        loader = DataLoader(
            ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, collate_fn=_identity_collate,
            pin_memory=False, prefetch_factor=prefetch,
            persistent_workers=False,
        )
        for batch_data in tqdm(loader, desc=f"Val rotation {exp+1}/{expand_times}"):
            aug_events.extend(batch_data)
        _append_to_split_hdf5(val_h5_path, aug_events, K_cd)
        n_val_total += len(aug_events)
    if expand_times > 0:
        logger.info(f"  val (with aug): {n_val_total} events")

    # --- Test: original + rotation copies (same pattern as train) ---
    test_h5_path = os.path.join(preprocessed_dir, 'test.h5')
    n_test_total = _tokenize_and_write(
        'test', [test_idx],
        test_h5_path,
        apply_rotation=False,
    )
    for exp in range(expand_times):
        logger.info(f"Test rotation augmentation {exp+1}/{expand_times}...")
        aug_events = []
        ds = H5EndpointDataset(
            h5_files, config, geometry,
            indices=test_idx, is_train=False,
            events_per_file=events_per_file,
            apply_rotation_aug=True,
        )
        loader = DataLoader(
            ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, collate_fn=_identity_collate,
            pin_memory=False, prefetch_factor=prefetch,
            persistent_workers=False,
        )
        for batch_data in tqdm(loader, desc=f"Test rotation {exp+1}/{expand_times}"):
            aug_events.extend(batch_data)
        _append_to_split_hdf5(test_h5_path, aug_events, K_cd)
        n_test_total += len(aug_events)
    if expand_times > 0:
        logger.info(f"  test (with aug): {n_test_total} events")

    # --- Save metadata ---
    meta = {
        'format': 'split_hdf5_v3',
        'cd_representation': 'sparse_pmt_precomputed',
        'cd_max_tokens': K_cd,
        'cd_topk_score': 'q_sum',
        'cd_feature_dim': 10,
        'cd_feature_order': [
            'ux', 'uy', 'uz', 'q_sum', 'q_max', 'n_hits',
            't_first', 't_mean', 't_late', 't_span',
        ],
        'train_events': n_train_total,
        'val_events': n_val_total,
        'test_events': n_test_total,
        'original_events': total_events,
        'expand_times': expand_times,
        'config_hash': _config_hash(config),
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Preprocessing complete: {preprocessed_dir}")
    logger.info(f"  train={n_train_total}, val={n_val_total}, test={n_test_total}")


# ---------------------------------------------------------------------------
# Thin reader for training (v3 format)
# ---------------------------------------------------------------------------

class SplitDataset(Dataset):
    """Thin reader: opens one split HDF5 (v3 format), reads by sequential index.

    Each worker opens its own file handle lazily (fork-safe).
    """

    def __init__(self, h5_path: str):
        self._h5_path = h5_path
        self._file = None

        # Read metadata once (small)
        with h5py.File(h5_path, 'r') as f:
            self._n_events = f['cd_tokens'].shape[0]
            self._wp_offsets = f['wp_offsets'][:]  # (N+1,) int64

    def _ensure_open(self):
        if self._file is None:
            self._file = h5py.File(self._h5_path, 'r', rdcc_nbytes=64 * 1024 * 1024)

    def __len__(self) -> int:
        return self._n_events

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        self._ensure_open()
        f = self._file

        # CD v3: float16 → float32 on read
        cd_tokens = torch.from_numpy(np.array(f['cd_tokens'][idx])).float()   # (K_cd, 10)
        cd_mask = torch.from_numpy(np.array(f['cd_mask'][idx]))               # (K_cd,) bool

        # WP: cumulative offset → flat slice
        s = int(self._wp_offsets[idx])
        e = int(self._wp_offsets[idx + 1])
        if e > s:
            wp_tokens = torch.from_numpy(
                np.array(f['wp_tokens_flat'][s * 5:e * 5])).float().reshape(-1, 5)
        else:
            wp_tokens = torch.zeros((0, 5), dtype=torch.float32)

        # Labels: one read of 12 floats
        labels = torch.from_numpy(np.array(f['labels'][idx])).float()  # (12,)

        return {
            'wp_tokens': wp_tokens,
            'cd_tokens': cd_tokens,
            'cd_mask': cd_mask,
            'labels': labels,
        }

    def __del__(self):
        if hasattr(self, '_file') and self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Collate: minimal per-sample data → full batch dict for the model (v3)
# ---------------------------------------------------------------------------

def collate_fn(batch: list) -> Dict[str, torch.Tensor]:
    """Assemble batch: pad WP, stack CD tokens."""
    B = len(batch)
    max_wp = max(b['wp_tokens'].shape[0] for b in batch)

    wp_tokens = torch.zeros(B, max_wp, 5)
    wp_mask = torch.ones(B, max_wp, dtype=torch.bool)   # True = padding
    labels = torch.zeros(B, 12)

    # CD: fixed-shape (K_cd, 10), just stack
    cd_tokens = torch.stack([b['cd_tokens'] for b in batch])   # (B, K_cd, 10)
    cd_mask = torch.stack([b['cd_mask'] for b in batch])       # (B, K_cd) bool

    for i, b in enumerate(batch):
        n_wp = b['wp_tokens'].shape[0]
        if n_wp > 0:
            wp_tokens[i, :n_wp] = b['wp_tokens']
            wp_mask[i, :n_wp] = False
        labels[i] = b['labels']

    return {
        'wp_tokens': wp_tokens,
        'wp_mask': wp_mask,
        'wp_unit_vecs': wp_tokens[:, :, :3],    # view
        'wp_times': wp_tokens[:, :, 4],          # view
        'cd_tokens': cd_tokens,
        'cd_mask': cd_mask,
        'u1': labels[:, :3],
        'u2': labels[:, 3:6],
        'p1': labels[:, 6:9],
        'p2': labels[:, 9:12],
    }


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def create_preprocessed_dataloaders(
    config: dict, preprocessed_dir: str
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create thin DataLoaders from split-level HDF5 files (v3 format)."""
    data_cfg = config['data']
    train_cfg = config['train']

    # Load metadata
    with open(os.path.join(preprocessed_dir, 'meta.json'), 'r') as f:
        meta = json.load(f)

    cd_repr = meta.get('cd_representation', 'unknown')
    if cd_repr != 'sparse_pmt_precomputed':
        raise RuntimeError(
            f"Expected cd_representation='sparse_pmt_precomputed', got '{cd_repr}'. "
            "Delete the preprocessed directory and re-run --Preprocess."
        )

    logger.info(f"Split HDF5 v3 data: train={meta['train_events']}, "
                f"val={meta['val_events']}, test={meta['test_events']}")
    logger.info(f"  K_cd={meta.get('cd_max_tokens')}, feature_dim={meta.get('cd_feature_dim')}")

    batch_size = train_cfg['batch_size']
    val_batch_size = train_cfg.get('val_batch_size', -1)
    if val_batch_size <= 0:
        val_batch_size = batch_size

    num_workers = data_cfg.get('num_workers', 8)
    prefetch = data_cfg.get('prefetch_factor', 2) if num_workers > 0 else None

    train_ds = SplitDataset(os.path.join(preprocessed_dir, 'train.h5'))
    val_ds = SplitDataset(os.path.join(preprocessed_dir, 'val.h5'))
    test_ds = SplitDataset(os.path.join(preprocessed_dir, 'test.h5'))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=num_workers,
                              pin_memory=True, persistent_workers=num_workers > 0,
                              prefetch_factor=prefetch)
    val_loader = DataLoader(val_ds, batch_size=val_batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=num_workers,
                            pin_memory=True, persistent_workers=num_workers > 0,
                            prefetch_factor=prefetch)
    test_loader = DataLoader(test_ds, batch_size=val_batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=num_workers,
                             pin_memory=True, persistent_workers=num_workers > 0,
                             prefetch_factor=prefetch)

    return train_loader, val_loader, test_loader
