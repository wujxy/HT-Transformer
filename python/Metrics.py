"""
Evaluation Metrics for Ordered Dual-Endpoint Reconstruction.

Computes angular errors, distance errors, direction consistency.
"""

import numpy as np
import torch
from typing import Dict, List


def compute_endpoint_metrics(pred_u1: np.ndarray, pred_u2: np.ndarray,
                            gt_u1: np.ndarray, gt_u2: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive endpoint reconstruction metrics.

    Args:
        pred_u1: (N, 3) predicted first endpoint unit vectors
        pred_u2: (N, 3) predicted second endpoint unit vectors
        gt_u1: (N, 3) ground truth first endpoint unit vectors
        gt_u2: (N, 3) ground truth second endpoint unit vectors

    Returns:
        Dict of metric name -> value
    """
    eps = 1e-10
    N = len(pred_u1)

    # --- Per-endpoint angular errors ---
    cos1 = np.sum(pred_u1 * gt_u1, axis=-1)
    cos2 = np.sum(pred_u2 * gt_u2, axis=-1)
    ang_err1 = np.degrees(np.arccos(np.clip(cos1, -1.0 + eps, 1.0 - eps)))
    ang_err2 = np.degrees(np.arccos(np.clip(cos2, -1.0 + eps, 1.0 - eps)))
    mean_ang_err = 0.5 * (ang_err1 + ang_err2)

    # --- Distance error ---
    pred_dist = np.linalg.norm(pred_u2 - pred_u1, axis=-1)
    gt_dist = np.linalg.norm(gt_u2 - gt_u1, axis=-1)
    dist_err = np.abs(pred_dist - gt_dist)

    # --- Direction consistency ---
    pred_dir = pred_u2 - pred_u1
    gt_dir = gt_u2 - gt_u1
    pred_dir_norm = pred_dir / (np.linalg.norm(pred_dir, axis=-1, keepdims=True) + eps)
    gt_dir_norm = gt_dir / (np.linalg.norm(gt_dir, axis=-1, keepdims=True) + eps)
    dir_cos = np.sum(pred_dir_norm * gt_dir_norm, axis=-1)
    dir_err = np.degrees(np.arccos(np.clip(dir_cos, -1.0 + eps, 1.0 - eps)))

    # --- Compile metrics ---
    def quantiles(arr):
        return {
            'mean': float(arr.mean()),
            'median': float(np.median(arr)),
            'std': float(arr.std()),
            'p68': float(np.percentile(arr, 68)),
            'p95': float(np.percentile(arr, 95)),
            'min': float(arr.min()),
            'max': float(arr.max()),
        }

    metrics = {
        'n_events': N,
        'endpoint1_ang_err': quantiles(ang_err1),
        'endpoint2_ang_err': quantiles(ang_err2),
        'mean_ang_err': quantiles(mean_ang_err),
        'distance_err': quantiles(dist_err),
        'direction_err': quantiles(dir_err),
        'direction_cos_mean': float(dir_cos.mean()),
    }

    return metrics


def compute_training_metrics(pred_u1: np.ndarray, pred_u2: np.ndarray,
                             gt_u1: np.ndarray, gt_u2: np.ndarray,
                             sphere_radius: float = 25000.0) -> Dict:
    """
    Compute training-time reconstruction metrics for eval visualization.

    Designed for use during training (called every eval_every epochs).
    Returns raw arrays for plotting plus scalar quantile summaries.

    Args:
        pred_u1: (N, 3) predicted first endpoint unit vectors
        pred_u2: (N, 3) predicted second endpoint unit vectors
        gt_u1: (N, 3) ground truth first endpoint unit vectors
        gt_u2: (N, 3) ground truth second endpoint unit vectors
        sphere_radius: radius in mm for converting unit vectors to physical coords

    Returns:
        Dict with raw arrays and scalar metric summaries
    """
    eps = 1e-10

    # --- Per-endpoint angular errors ---
    cos1 = np.sum(pred_u1 * gt_u1, axis=-1)
    cos2 = np.sum(pred_u2 * gt_u2, axis=-1)
    ep1_angle = np.degrees(np.arccos(np.clip(cos1, -1.0 + eps, 1.0 - eps)))
    ep2_angle = np.degrees(np.arccos(np.clip(cos2, -1.0 + eps, 1.0 - eps)))

    # --- Direction angular error (ignoring sign, acute angle) ---
    pred_dir = pred_u2 - pred_u1
    gt_dir = gt_u2 - gt_u1
    pred_dir_norm = np.linalg.norm(pred_dir, axis=-1, keepdims=True)
    gt_dir_norm = np.linalg.norm(gt_dir, axis=-1, keepdims=True)
    pred_dir = pred_dir / (pred_dir_norm + eps)
    gt_dir = gt_dir / (gt_dir_norm + eps)
    dir_cos = np.abs(np.sum(pred_dir * gt_dir, axis=-1))  # abs for acute angle
    dir_angle = np.degrees(np.arccos(np.clip(dir_cos, 0.0, 1.0 - eps)))

    # --- Midpoint distance (physical mm) ---
    pred_mid = (pred_u1 + pred_u2) / 2.0 * sphere_radius
    gt_mid = (gt_u1 + gt_u2) / 2.0 * sphere_radius
    midpoint_dist = np.linalg.norm(pred_mid - gt_mid, axis=-1)

    # --- Quantile summaries ---
    def _quantile_summary(arr, prefix):
        return {
            f'{prefix}_mean': float(arr.mean()),
            f'{prefix}_median': float(np.median(arr)),
            f'{prefix}_std': float(arr.std()),
            f'{prefix}_p50': float(np.percentile(arr, 50)),
            f'{prefix}_p68': float(np.percentile(arr, 68)),
            f'{prefix}_p90': float(np.percentile(arr, 90)),
            f'{prefix}_p95': float(np.percentile(arr, 95)),
            f'{prefix}_p99': float(np.percentile(arr, 99)),
        }

    result = {
        'n_events': len(pred_u1),
        'dir_angle': dir_angle,
        'midpoint_dist': midpoint_dist,
        'ep1_angle': ep1_angle,
        'ep2_angle': ep2_angle,
    }
    result.update(_quantile_summary(dir_angle, 'dir_ang'))
    result.update(_quantile_summary(midpoint_dist, 'mid_dist'))
    result.update(_quantile_summary(ep1_angle, 'ep1_ang'))
    result.update(_quantile_summary(ep2_angle, 'ep2_ang'))

    return result


def format_metrics(metrics: Dict) -> str:
    """Format metrics dict into readable string."""
    lines = []
    lines.append(f"Events: {metrics['n_events']}")
    lines.append("")
    lines.append("Endpoint 1 Angular Error (deg):")
    for k, v in metrics['endpoint1_ang_err'].items():
        lines.append(f"  {k}: {v:.4f}")
    lines.append("")
    lines.append("Endpoint 2 Angular Error (deg):")
    for k, v in metrics['endpoint2_ang_err'].items():
        lines.append(f"  {k}: {v:.4f}")
    lines.append("")
    lines.append("Mean Angular Error (deg):")
    for k, v in metrics['mean_ang_err'].items():
        lines.append(f"  {k}: {v:.4f}")
    lines.append("")
    lines.append("Distance Error:")
    for k, v in metrics['distance_err'].items():
        lines.append(f"  {k}: {v:.4f}")
    lines.append("")
    lines.append("Direction Error (deg):")
    for k, v in metrics['direction_err'].items():
        lines.append(f"  {k}: {v:.4f}")
    lines.append(f"  direction_cos_mean: {metrics['direction_cos_mean']:.4f}")

    return "\n".join(lines)


def evaluate_model(model, dataloader, device, config) -> Dict:
    """
    Run full evaluation of model on dataloader.

    Returns:
        Dict with all predictions, ground truths, and metrics
    """
    import torch
    from tqdm import tqdm
    from LossFunction import EndpointLoss

    model.eval()
    criterion = EndpointLoss(
        lambda_ang=config['loss']['lambda_ang'],
        lambda_len=config['loss']['lambda_len'],
        lambda_dir=config['loss']['lambda_dir'],
    )

    all_pred_u1 = []
    all_pred_u2 = []
    all_gt_u1 = []
    all_gt_u2 = []
    all_gt_p1 = []
    all_gt_p2 = []
    total_loss = 0
    n_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

            # Add kNN adj
            adj = dataloader.dataset.healpix.get_knn_adjacency(k=config['model']['cd_knn_k'])
            batch_gpu['cd_knn_adj'] = torch.from_numpy(adj).long().to(device)

            outputs = model(batch_gpu)
            loss, _ = criterion(
                outputs['pred_u1'], outputs['pred_u2'],
                batch_gpu['u1'], batch_gpu['u2']
            )

            all_pred_u1.append(outputs['pred_u1'].cpu().numpy())
            all_pred_u2.append(outputs['pred_u2'].cpu().numpy())
            all_gt_u1.append(batch['u1'].numpy())
            all_gt_u2.append(batch['u2'].numpy())
            if 'p1' in batch:
                all_gt_p1.append(batch['p1'].numpy())
                all_gt_p2.append(batch['p2'].numpy())
            total_loss += loss.item()
            n_batches += 1

    pred_u1 = np.concatenate(all_pred_u1)
    pred_u2 = np.concatenate(all_pred_u2)
    gt_u1 = np.concatenate(all_gt_u1)
    gt_u2 = np.concatenate(all_gt_u2)

    metrics = compute_endpoint_metrics(pred_u1, pred_u2, gt_u1, gt_u2)
    metrics['val_loss'] = total_loss / max(1, n_batches)

    # Save raw arrays for plotting
    metrics['_pred_u1'] = pred_u1
    metrics['_pred_u2'] = pred_u2
    metrics['_gt_u1'] = gt_u1
    metrics['_gt_u2'] = gt_u2
    if all_gt_p1:
        metrics['_gt_p1'] = np.concatenate(all_gt_p1)
        metrics['_gt_p2'] = np.concatenate(all_gt_p2)

    return metrics
