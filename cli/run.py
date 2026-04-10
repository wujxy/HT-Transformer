"""
Unified entry point for HT-Transformer-DS.

Modes:
  --InspectH5 : Run h5 schema inspection
  --TrainModel: Train model
  --Predict   : Run prediction
  --Eval      : Run evaluation + visualization
  --Preprocess: Preprocess H5 data into tokenized .pt files

Usage:
  python -m ht_transformer_ds.cli.run --config configs/default.yaml --TrainModel
  python -m ht_transformer_ds.cli.run --config configs/default.yaml --InspectH5
"""

import os

from loguru import logger
from config.loader import load_config


def main():
    config = load_config()
    mode = config['_mode']

    if mode['inspect_h5']:
        from data.inspect_h5 import inspect_h5
        h5_path = config['data']['h5_path']
        inspect_h5(h5_path)
        return

    if mode.get('preprocess'):
        from data.preprocess import preprocess
        preprocess(config)
        return

    if mode['train']:
        from engine.trainer import Trainer
        trainer = Trainer(config)
        trainer.run()
        return

    if mode['predict']:
        from engine.predictor import Predictor
        checkpoint = config.get('checkpoint_path', None)
        if checkpoint is None:
            # Find latest checkpoint
            ckpt_dir = os.path.join(config['output_path'], config['mission_name'], 'checkpoints')
            if os.path.isdir(ckpt_dir):
                ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith('.pt') or f.endswith('.pth')])
                if ckpts:
                    checkpoint = os.path.join(ckpt_dir, ckpts[-1])
                else:
                    logger.error(f"No checkpoints found in {ckpt_dir}")
                    return
            else:
                logger.error(f"Checkpoint directory not found: {ckpt_dir}")
                return
        predictor = Predictor(config, checkpoint)
        predictor.predict()
        return

    if mode['eval']:
        from engine.predictor import Predictor
        from metrics.endpoint_metrics import evaluate_model, format_metrics, compute_endpoint_metrics
        from visualization.plotting import plot_training_curves, plot_result_distributions, plot_event_3d
        from geometry.detector_geometry import DualPMTPositionLookup
        from data.dataset import create_dataloaders
        from models.ht_transformer import HTTransformer
        import torch
        import numpy as np
        import json

        # Find latest checkpoint
        ckpt_dir = os.path.join(config['output_path'], config['mission_name'], 'checkpoints')
        if not os.path.isdir(ckpt_dir):
            logger.error(f"Checkpoint directory not found: {ckpt_dir}")
            return
        ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith('.pt') or f.endswith('.pth')])
        if not ckpts:
            logger.error(f"No checkpoints found in {ckpt_dir}")
            return
        checkpoint = os.path.join(ckpt_dir, ckpts[-1])

        # Setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        geometry = DualPMTPositionLookup(
            config['data']['geometry_cd'], config['data']['geometry_wp'])

        _, _, test_loader = create_dataloaders(config, geometry)

        # Load model
        model = HTTransformer(config).to(device)
        ckpt = torch.load(checkpoint, map_location=device)
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            model.load_state_dict(ckpt)
        logger.info(f"Loaded checkpoint: {checkpoint}")

        # Evaluate
        metrics = evaluate_model(model, test_loader, device, config)
        metrics_str = format_metrics(metrics)
        logger.info(f"\n{metrics_str}")

        # Output dir
        output_dir = os.path.join(config['output_path'], config['mission_name'], 'eval_results')
        os.makedirs(output_dir, exist_ok=True)

        # Save metrics
        metrics_save = {k: v for k, v in metrics.items() if not k.startswith('_')}
        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics_save, f, indent=2)
        with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
            f.write(metrics_str)

        # Plot training curves
        history_path = os.path.join(config['output_path'], config['mission_name'], 'training_history.json')
        if os.path.exists(history_path):
            plot_dir = os.path.join(config['output_path'], config['mission_name'], 'plots')
            plot_training_curves(history_path, plot_dir)
            logger.info(f"Training curves saved to {plot_dir}")

        # Plot result distributions
        plot_result_distributions(metrics, output_dir)
        logger.info(f"Result distributions saved to {output_dir}")

        # Plot sample events
        pred_u1 = metrics['_pred_u1']
        pred_u2 = metrics['_pred_u2']
        gt_u1 = metrics['_gt_u1']
        gt_u2 = metrics['_gt_u2']
        n_plot = min(5, len(pred_u1))
        for i in range(n_plot):
            plot_event_3d(pred_u1, pred_u2, gt_u1, gt_u2,
                         event_idx=i,
                         sphere_radius=config['data'].get('sphere_radius', 25000.0),
                         output_dir=output_dir)
        logger.info(f"Event visualizations saved to {output_dir}")
        return

    logger.warning("No mode specified. Use --InspectH5, --TrainModel, --Predict, or --Eval")


if __name__ == "__main__":
    main()
