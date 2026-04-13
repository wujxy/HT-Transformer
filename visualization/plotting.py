"""
Visualization for HT-Transformer Endpoint Reconstruction.

Generates training curves, result distributions, and event-level 3D visualizations.
"""

import os
import numpy as np
import json
from typing import Dict, Optional


def plot_training_curves(history_path: str, output_dir: str):
    """Plot training loss curves from history JSON."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    with open(history_path, 'r') as f:
        history = json.load(f)

    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Total loss
    axes[0, 0].plot(epochs, history['train_loss'], label='Train')
    axes[0, 0].plot(epochs, history['val_loss'], label='Val')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].set_xlabel('Epoch')

    # Angle loss
    axes[0, 1].plot(epochs, history['train_ang'], label='Train')
    axes[0, 1].plot(epochs, history['val_ang'], label='Val')
    axes[0, 1].set_title('Angle Loss')
    axes[0, 1].legend()

    # Length loss
    axes[0, 2].plot(epochs, history['train_len'], label='Train')
    axes[0, 2].plot(epochs, history['val_len'], label='Val')
    axes[0, 2].set_title('Length Loss')
    axes[0, 2].legend()

    # Direction loss
    axes[1, 0].plot(epochs, history['train_dir'], label='Train')
    axes[1, 0].plot(epochs, history['val_dir'], label='Val')
    axes[1, 0].set_title('Direction Loss')
    axes[1, 0].legend()

    # LR curve
    axes[1, 1].plot(epochs, history['lr'])
    axes[1, 1].set_title('Learning Rate')
    axes[1, 1].set_xlabel('Epoch')

    # Combined loss components
    axes[1, 2].plot(epochs, history['train_ang'], label='Angle')
    axes[1, 2].plot(epochs, history['train_len'], label='Length')
    axes[1, 2].plot(epochs, history['train_dir'], label='Direction')
    axes[1, 2].set_title('Train Loss Components')
    axes[1, 2].legend()

    plt.tight_layout()
    path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_result_distributions(metrics: Dict, output_dir: str):
    """Plot result distribution histograms and violin plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    pred_u1 = metrics['_pred_u1']
    pred_u2 = metrics['_pred_u2']
    gt_u1 = metrics['_gt_u1']
    gt_u2 = metrics['_gt_u2']

    eps = 1e-10

    # Compute errors
    cos1 = np.sum(pred_u1 * gt_u1, axis=-1)
    cos2 = np.sum(pred_u2 * gt_u2, axis=-1)
    ang_err1 = np.degrees(np.arccos(np.clip(cos1, -1 + eps, 1 - eps)))
    ang_err2 = np.degrees(np.arccos(np.clip(cos2, -1 + eps, 1 - eps)))
    mean_ang_err = 0.5 * (ang_err1 + ang_err2)

    pred_dist = np.linalg.norm(pred_u2 - pred_u1, axis=-1)
    gt_dist = np.linalg.norm(gt_u2 - gt_u1, axis=-1)
    dist_err = np.abs(pred_dist - gt_dist)

    pred_dir = pred_u2 - pred_u1
    gt_dir = gt_u2 - gt_u1
    pred_dir_n = pred_dir / (np.linalg.norm(pred_dir, axis=-1, keepdims=True) + eps)
    gt_dir_n = gt_dir / (np.linalg.norm(gt_dir, axis=-1, keepdims=True) + eps)
    dir_cos = np.sum(pred_dir_n * gt_dir_n, axis=-1)
    dir_err_deg = np.degrees(np.arccos(np.clip(dir_cos, -1 + eps, 1 - eps)))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Angular error histogram
    bins = np.linspace(0, max(mean_ang_err.max(), 30), 50)
    axes[0, 0].hist(ang_err1, bins=bins, alpha=0.5, label=f'EP1 (median={np.median(ang_err1):.2f}°)')
    axes[0, 0].hist(ang_err2, bins=bins, alpha=0.5, label=f'EP2 (median={np.median(ang_err2):.2f}°)')
    axes[0, 0].hist(mean_ang_err, bins=bins, alpha=0.5, label=f'Mean (median={np.median(mean_ang_err):.2f}°)')
    axes[0, 0].set_xlabel('Angular Error (deg)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Endpoint Angular Error Distribution')
    axes[0, 0].legend()

    # Angular error box plot
    axes[0, 1].boxplot([ang_err1, ang_err2, mean_ang_err],
                       labels=['EP1', 'EP2', 'Mean'])
    axes[0, 1].set_ylabel('Angular Error (deg)')
    axes[0, 1].set_title('Angular Error Box Plot')

    # Distance error histogram
    axes[1, 0].hist(dist_err, bins=50, alpha=0.7, color='steelblue')
    axes[1, 0].axvline(np.median(dist_err), color='red', linestyle='--',
                       label=f'Median={np.median(dist_err):.4f}')
    axes[1, 0].set_xlabel('Chord Distance Error')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Endpoint Distance Error Distribution')
    axes[1, 0].legend()

    # Direction consistency histogram
    axes[1, 1].hist(dir_err_deg, bins=50, alpha=0.7, color='coral')
    axes[1, 1].axvline(np.median(dir_err_deg), color='red', linestyle='--',
                       label=f'Median={np.median(dir_err_deg):.2f}°')
    axes[1, 1].set_xlabel('Direction Error (deg)')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Direction Consistency Distribution')
    axes[1, 1].legend()

    plt.tight_layout()
    path = os.path.join(output_dir, 'result_distributions.png')
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_training_eval_distributions(eval_results: Dict, output_dir: str,
                                      epoch: int, prefix: str = "val"):
    """
    Generate training-time evaluation distribution plots.

    Called every eval_every epochs during training. Produces three plots:
    1. Direction angular error distribution (with CD-in/CD-out overlay)
    2. Midpoint distance distribution (with CD-in/CD-out overlay)
    3. Per-endpoint angular error comparison (with CD-in/CD-out overlay)

    Style follows reference project ModelTrainingPlotter conventions:
    bins=200, histtype='step', linewidth=2, quantile vertical lines.

    Args:
        eval_results: dict from compute_training_metrics()
        output_dir: directory to save plots
        epoch: current epoch number (used in filename and title)
        prefix: filename prefix (e.g. "val")
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    is_in_cd = eval_results.get('is_in_cd', None)
    has_cd_split = is_in_cd is not None and len(is_in_cd) == len(eval_results['dir_angle'])
    cd_in_mask = is_in_cd.astype(bool) if has_cd_split else None
    cd_out_mask = ~cd_in_mask if has_cd_split else None

    # --- Plot 1: Direction angular error distribution ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(eval_results['dir_angle'], bins=200, range=(0, 180),
            histtype='step', linewidth=2, color='steelblue',
            label=f'All (N={len(eval_results["dir_angle"])})')
    if has_cd_split:
        n_in = int(cd_in_mask.sum())
        n_out = int(cd_out_mask.sum())
        if n_in > 0:
            ax.hist(eval_results['dir_angle'][cd_in_mask], bins=200, range=(0, 180),
                    histtype='step', linewidth=1.5, color='crimson', alpha=0.8,
                    label=f'CD-in (N={n_in})')
        if n_out > 0:
            ax.hist(eval_results['dir_angle'][cd_out_mask], bins=200, range=(0, 180),
                    histtype='step', linewidth=1.5, color='forestgreen', alpha=0.8,
                    label=f'CD-out (N={n_out})')
    for q, c, ls in [('p68', 'orange', '--'), ('p90', 'red', '-.'),
                     ('p99', 'darkred', ':')]:
        val = eval_results.get(f'dir_ang_{q}', None)
        if val is not None:
            ax.axvline(val, color=c, linestyle=ls, linewidth=1.5,
                       label=f'{q}={val:.2f} deg')
    ax.set_xlabel('Direction Angular Error (deg)')
    ax.set_ylabel('Count')
    ax.set_title(f'Direction Angle Distribution (Epoch {epoch})')
    ax.legend(fontsize=8)
    plt.tight_layout()
    path1 = os.path.join(output_dir, f'{prefix}_angle_distribution_epoch{epoch}.png')
    plt.savefig(path1, dpi=150)
    plt.close()

    # --- Plot 2: Midpoint distance distribution ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(eval_results['midpoint_dist'], bins=200, range=(0, 35000),
            histtype='step', linewidth=2, color='teal',
            label=f'All (N={len(eval_results["midpoint_dist"])})')
    if has_cd_split:
        if n_in > 0:
            ax.hist(eval_results['midpoint_dist'][cd_in_mask], bins=200, range=(0, 35000),
                    histtype='step', linewidth=1.5, color='crimson', alpha=0.8,
                    label=f'CD-in (N={n_in})')
        if n_out > 0:
            ax.hist(eval_results['midpoint_dist'][cd_out_mask], bins=200, range=(0, 35000),
                    histtype='step', linewidth=1.5, color='forestgreen', alpha=0.8,
                    label=f'CD-out (N={n_out})')
    for q, c, ls in [('p68', 'orange', '--'), ('p90', 'red', '-.'),
                     ('p99', 'darkred', ':')]:
        val = eval_results.get(f'mid_dist_{q}', None)
        if val is not None:
            ax.axvline(val, color=c, linestyle=ls, linewidth=1.5,
                       label=f'{q}={val:.0f} mm')
    ax.set_xlabel('Midpoint Distance (mm)')
    ax.set_ylabel('Count')
    ax.set_title(f'Midpoint Distance Distribution (Epoch {epoch})')
    ax.legend(fontsize=8)
    plt.tight_layout()
    path2 = os.path.join(output_dir, f'{prefix}_midpoint_distance_distribution_epoch{epoch}.png')
    plt.savefig(path2, dpi=150)
    plt.close()

    # --- Plot 3: Per-endpoint angular error comparison ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(eval_results['ep1_angle'], bins=200, range=(0, 180),
            histtype='step', linewidth=2, alpha=0.8, color='royalblue',
            label=f'EP1 (med={np.median(eval_results["ep1_angle"]):.2f})')
    ax.hist(eval_results['ep2_angle'], bins=200, range=(0, 180),
            histtype='step', linewidth=2, alpha=0.8, color='tomato',
            label=f'EP2 (med={np.median(eval_results["ep2_angle"]):.2f})')
    if has_cd_split:
        if n_in > 0:
            ax.hist(eval_results['mean_ep_angle'][cd_in_mask], bins=200, range=(0, 180),
                    histtype='step', linewidth=1.5, color='crimson', alpha=0.6,
                    label=f'Mean CD-in (med={np.median(eval_results["mean_ep_angle"][cd_in_mask]):.2f})')
        if n_out > 0:
            ax.hist(eval_results['mean_ep_angle'][cd_out_mask], bins=200, range=(0, 180),
                    histtype='step', linewidth=1.5, color='forestgreen', alpha=0.6,
                    label=f'Mean CD-out (med={np.median(eval_results["mean_ep_angle"][cd_out_mask]):.2f})')
    ax.set_xlabel('Endpoint Angular Error (deg)')
    ax.set_ylabel('Count')
    ax.set_title(f'Per-Endpoint Angle Distribution (Epoch {epoch})')
    ax.legend(fontsize=8)
    plt.tight_layout()
    path3 = os.path.join(output_dir, f'{prefix}_endpoint_angle_distribution_epoch{epoch}.png')
    plt.savefig(path3, dpi=150)
    plt.close()

    # --- Plot 4: CD-in vs CD-out separate comparison ---
    if has_cd_split and n_in > 0 and n_out > 0:
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))

        # Direction angle
        ax = axes[0]
        ax.hist(eval_results['dir_angle'][cd_in_mask], bins=200, range=(0, 180),
                histtype='step', linewidth=2, color='crimson', density=True,
                label=f'CD-in (N={n_in}, p68={eval_results.get("cd_in_dir_ang_p68", 0):.2f})')
        ax.hist(eval_results['dir_angle'][cd_out_mask], bins=200, range=(0, 180),
                histtype='step', linewidth=2, color='forestgreen', density=True,
                label=f'CD-out (N={n_out}, p68={eval_results.get("cd_out_dir_ang_p68", 0):.2f})')
        ax.set_xlabel('Direction Angular Error (deg)')
        ax.set_ylabel('Normalized Count')
        ax.set_title('Direction Angle: CD-in vs CD-out')
        ax.legend(fontsize=8)

        # Midpoint distance
        ax = axes[1]
        ax.hist(eval_results['midpoint_dist'][cd_in_mask], bins=200, range=(0, 35000),
                histtype='step', linewidth=2, color='crimson', density=True,
                label=f'CD-in (p68={eval_results.get("cd_in_mid_dist_p68", 0):.0f}mm)')
        ax.hist(eval_results['midpoint_dist'][cd_out_mask], bins=200, range=(0, 35000),
                histtype='step', linewidth=2, color='forestgreen', density=True,
                label=f'CD-out (p68={eval_results.get("cd_out_mid_dist_p68", 0):.0f}mm)')
        ax.set_xlabel('Midpoint Distance (mm)')
        ax.set_ylabel('Normalized Count')
        ax.set_title('Midpoint Distance: CD-in vs CD-out')
        ax.legend(fontsize=8)

        # Mean endpoint angle
        ax = axes[2]
        ax.hist(eval_results['mean_ep_angle'][cd_in_mask], bins=200, range=(0, 180),
                histtype='step', linewidth=2, color='crimson', density=True,
                label=f'CD-in (p68={eval_results.get("cd_in_mean_ep_ang_p68", 0):.2f})')
        ax.hist(eval_results['mean_ep_angle'][cd_out_mask], bins=200, range=(0, 180),
                histtype='step', linewidth=2, color='forestgreen', density=True,
                label=f'CD-out (p68={eval_results.get("cd_out_mean_ep_ang_p68", 0):.2f})')
        ax.set_xlabel('Mean Endpoint Angular Error (deg)')
        ax.set_ylabel('Normalized Count')
        ax.set_title('Mean EP Angle: CD-in vs CD-out')
        ax.legend(fontsize=8)

        plt.suptitle(f'CD-in vs CD-out Comparison (Epoch {epoch})', fontsize=12)
        plt.tight_layout()
        path4 = os.path.join(output_dir, f'{prefix}_cd_in_out_comparison_epoch{epoch}.png')
        plt.savefig(path4, dpi=150)
        plt.close()
        return path1, path2, path3, path4

    return path1, path2, path3


def plot_event_3d(pred_u1: np.ndarray, pred_u2: np.ndarray,
                  gt_u1: np.ndarray, gt_u2: np.ndarray,
                  event_idx: int = 0,
                  sphere_radius: float = 25000.0,
                  output_dir: str = 'plots',
                  wp_unit_vecs: Optional[np.ndarray] = None,
                  wp_charges: Optional[np.ndarray] = None,
                  cd_unit_vecs: Optional[np.ndarray] = None,
                  cd_charges: Optional[np.ndarray] = None):
    """
    Plot 3D sphere visualization for a single event.

    Shows PMT hit heatmap, ground truth endpoints, predicted endpoints, and direction arrows.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    R = sphere_radius

    # Draw wireframe sphere
    u_sphere = np.linspace(0, 2 * np.pi, 30)
    v_sphere = np.linspace(0, np.pi, 20)
    xs = R * np.outer(np.cos(u_sphere), np.sin(v_sphere))
    ys = R * np.outer(np.sin(u_sphere), np.sin(v_sphere))
    zs = R * np.outer(np.ones(np.size(u_sphere)), np.cos(v_sphere))
    ax.plot_wireframe(xs, ys, zs, alpha=0.05, color='gray')

    # Plot WP hits as scatter
    if wp_unit_vecs is not None and wp_charges is not None:
        wp_pos = wp_unit_vecs * R * 1.01  # slightly outside sphere
        sc = ax.scatter(wp_pos[:, 0], wp_pos[:, 1], wp_pos[:, 2],
                       c=wp_charges, cmap='YlOrRd', s=2, alpha=0.3,
                       label='WP hits')

    # Plot CD patches as scatter
    if cd_unit_vecs is not None and cd_charges is not None:
        cd_pos = cd_unit_vecs * R * 0.99  # slightly inside sphere
        ax.scatter(cd_pos[:, 0], cd_pos[:, 1], cd_pos[:, 2],
                  c=cd_charges, cmap='Blues', s=3, alpha=0.3,
                  label='CD patches')

    # Ground truth endpoints
    gt_p1 = gt_u1[event_idx] * R
    gt_p2 = gt_u2[event_idx] * R
    ax.scatter(*gt_p1, color='green', s=200, marker='o', label='GT EP1', zorder=10)
    ax.scatter(*gt_p2, color='blue', s=200, marker='o', label='GT EP2', zorder=10)

    # Predicted endpoints
    pr_p1 = pred_u1[event_idx] * R
    pr_p2 = pred_u2[event_idx] * R
    ax.scatter(*pr_p1, color='lime', s=200, marker='^', label='Pred EP1', zorder=10)
    ax.scatter(*pr_p2, color='red', s=200, marker='^', label='Pred EP2', zorder=10)

    # Direction arrows (GT and Pred tracks)
    arrow_len = R * 0.3
    gt_dir = (gt_p2 - gt_p1)
    gt_dir = gt_dir / (np.linalg.norm(gt_dir) + 1e-10) * arrow_len
    pr_dir = (pr_p2 - pr_p1)
    pr_dir = pr_dir / (np.linalg.norm(pr_dir) + 1e-10) * arrow_len

    ax.quiver(*gt_p1, *gt_dir, color='green', arrow_length_ratio=0.3, linewidth=2)
    ax.quiver(*pr_p1, *pr_dir, color='lime', arrow_length_ratio=0.3, linewidth=2)

    # Connect GT and Pred endpoints
    ax.plot([gt_p1[0], gt_p2[0]], [gt_p1[1], gt_p2[1]], [gt_p1[2], gt_p2[2]],
            'g--', linewidth=1.5, label='GT track')
    ax.plot([pr_p1[0], pr_p2[0]], [pr_p1[1], pr_p2[1]], [pr_p1[2], pr_p2[2]],
            'r--', linewidth=1.5, label='Pred track')

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(f'Event {event_idx}: Endpoint Reconstruction')

    # Set equal aspect
    max_range = R * 1.1
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)

    ax.legend(loc='upper left', fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, f'event_{event_idx}_3d.png')
    plt.savefig(path, dpi=150)
    plt.close()
    return path
