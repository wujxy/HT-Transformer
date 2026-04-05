"""
Configuration loader for HT-Transformer.

Loads YAML config + optional CLI argument overrides.
Follows reference project's argparse-based interface style.
"""

import os
import yaml
import argparse
import copy
from loguru import logger


def load_yaml(path: str) -> dict:
    """Load YAML config file."""
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def deep_update(base: dict, override: dict) -> dict:
    """Recursively update base dict with override dict."""
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_update(result[k], v)
        else:
            result[k] = v
    return result


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser compatible with reference project style."""
    parser = argparse.ArgumentParser(description="HT-Transformer for JUNO Endpoint Reconstruction")

    # Mode selection
    parser.add_argument('--InspectH5', action='store_true', help='Run h5 schema inspection')
    parser.add_argument('--TrainModel', action='store_true', help='Train model')
    parser.add_argument('--Predict', action='store_true', help='Run prediction')
    parser.add_argument('--Eval', action='store_true', help='Run evaluation + visualization')
    parser.add_argument('--Preprocess', action='store_true',
                        help='Preprocess H5 data into tokenized .pt files')

    # Config file
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to YAML config file')

    # Data overrides
    parser.add_argument('--h5_path', type=str, default=None)
    parser.add_argument('--h5_val_path', type=str, default=None)
    parser.add_argument('--max_hits', type=int, default=None)

    # Model overrides
    parser.add_argument('--d_model', type=int, default=None)
    parser.add_argument('--num_layers', type=int, default=None)
    parser.add_argument('--num_heads', type=int, default=None)
    parser.add_argument('--d_ff', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--num_epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)

    # Output
    parser.add_argument('--mission_name', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)

    return parser


def load_config(args=None) -> dict:
    """
    Load configuration from YAML + CLI overrides.

    Returns:
        Merged config dict
    """
    parser = build_parser()
    parsed = parser.parse_args(args)

    # Load YAML
    config_path = parsed.config
    if not os.path.exists(config_path):
        logger.warning(f"Config file not found: {config_path}, using defaults")
        config = _default_config()
    else:
        config = load_yaml(config_path)
        logger.info(f"Loaded config from: {config_path}")

    # Apply CLI overrides
    overrides = {}
    cli_map = {
        'h5_path': ('data', 'h5_path'),
        'h5_val_path': ('data', 'h5_val_path'),
        'max_hits': ('data', 'max_hits'),
        'd_model': ('model', 'd_model'),
        'num_layers': ('model', 'num_layers'),
        'num_heads': ('model', 'num_heads'),
        'd_ff': ('model', 'd_ff'),
        'batch_size': ('train', 'batch_size'),
        'num_epochs': ('train', 'num_epochs'),
        'lr': ('train', 'lr'),
        'mission_name': ('mission_name',),
        'output_path': ('output_path',),
    }

    for cli_key, path in cli_map.items():
        val = getattr(parsed, cli_key, None)
        if val is not None:
            d = overrides
            for p in path[:-1]:
                d = d.setdefault(p, {})
            d[path[-1]] = val

    if overrides:
        config = deep_update(config, overrides)
        logger.info(f"Applied CLI overrides: {overrides}")

    # Store mode flags
    config['_mode'] = {
        'inspect_h5': parsed.InspectH5,
        'preprocess': parsed.Preprocess,
        'train': parsed.TrainModel,
        'predict': parsed.Predict,
        'eval': parsed.Eval,
    }

    return config


def _default_config() -> dict:
    """Return hardcoded default config as fallback."""
    return {
        'mission_name': 'ht_transformer_v1',
        'output_path': 'output',
        'detector_type': 'CDWP',
        'data': {
            'h5_path': 'output_cdwp_train_10459.h5',
            'h5_val_path': None,
            'geometry_cd': '/datafs/users/wujxy/muon_reco/muon_track_reco_CDWP/muon_track_reco_transformer/PMTPos_CD_LPMT.csv',
            'geometry_wp': '/datafs/users/wujxy/muon_reco/muon_track_reco_CDWP/muon_track_reco_transformer/PMTPos_WP_LPMT.csv',
            'h5_key_map': {
                'copyno': 'copyno', 'charge': 'charge', 'time': 'time',
                'nhits': 'nhits',
                'enter_x': 'tt_enter_x', 'enter_y': 'tt_enter_y', 'enter_z': 'tt_enter_z',
                'exit_x': 'tt_exit_x', 'exit_y': 'tt_exit_y', 'exit_z': 'tt_exit_z',
                'track_time': 'track_time',
            },
            'max_hits': 20012,
            'train_ratio': 0.8, 'val_ratio': 0.1, 'test_ratio': 0.1,
            'nside': 8, 'num_time_bins': 32,
            't_max': 800.0, 'auto_t_max': False,
        },
        'model': {
            'd_model': 128, 'num_layers': 4, 'num_heads': 4, 'd_ff': 512,
            'num_queries': 2, 'num_global_tokens': 8,
            'cd_knn_k': 16,
            'abs_posenc': 'fourier', 'fourier_freq': 32, 'dropout': 0.1,
            # WP projector (dual-branch)
            'wp_geo_hidden': 32,
            'wp_qt_hidden': 32,
            'wp_projector_hidden': 64,
            # WP time bias
            'wp_time_bias': 'signed_bucket',
            'wp_num_time_buckets': 64,
            'wp_time_bias_heads_shared': False,
            # CD DeepSphere
            'cd_deepsphere_layers': 4,
            'cd_deepsphere_hidden': 256,
            'cd_compression': 'healpix_pool',
            'cd_fusion_tokens': 128,
            'cd_compression_nside': 4,
            # Global normalization
            'norm_type': 'rmsnorm',
        },
        'loss': {'lambda_ang': 1.0, 'lambda_len': 0.5, 'lambda_dir': 0.25},
        'train': {
            'optimizer': 'adamw', 'lr': 3e-4, 'weight_decay': 1e-2,
            # V2 scheduler: Warmup + ReduceLROnPlateau (fixed configuration)
            'warmup_epochs': 5,       # Linear warmup epochs at start
            'plateau_factor': 0.5,    # LR reduction factor on plateau
            'plateau_patience': 5,    # Epochs to wait before reducing LR
            'plateau_threshold': 1e-3, # Minimum change to qualify as improvement
            'plateau_min_lr': 1e-6,   # Minimum learning rate
            # Early stopping (V2)
            'early_stop_patience': 8,
            'early_stop_monitor': 'val_dir_ang_p68',  # Metric to monitor
            'early_stop_mode': 'min',
            # Training settings
            'precision': 'bf16', 'dropout': 0.1, 'grad_clip': 1.0,
            'batch_size': 2, 'num_epochs': 200,
            'save_every': 50, 'eval_every': 10,
            'use_accelerate': False, 'seed': 42,
        },
        'augmentation': {
            'use_rotation_aug': False, 'use_time_jitter': False,
            'use_drop_hits': False, 'use_noise_injection': False,
        },
    }
