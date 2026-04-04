"""
Prediction module for HT-Transformer Endpoint Reconstruction.

Loads checkpoint, runs inference, saves results.
"""

import os
import numpy as np
import torch
from loguru import logger

from Config import load_config
from Geometry import DualPMTPositionLookup
from HEALPix import HEALPixMapper
from DataLoader import create_dataloaders
from Model import HTTransformer


class Predictor:
    """Prediction orchestration for HT-Transformer."""

    def __init__(self, config: dict, checkpoint_path: str):
        self.cfg = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Geometry + HEALPix
        self.geometry = DualPMTPositionLookup(
            config['data']['geometry_cd'], config['data']['geometry_wp'])

        cd_unit_vecs = self.geometry.cd_position_array.copy()
        norms = np.linalg.norm(cd_unit_vecs, axis=1, keepdims=True)
        valid = (norms.squeeze() > 0)
        cd_unit_vecs[valid] = cd_unit_vecs[valid] / norms[valid]

        self.healpix = HEALPixMapper(nside=config['data']['nside'], cd_unit_vecs=cd_unit_vecs)
        self.healpix.build_knn_adjacency(k=config['model']['cd_knn_k'])

        # Model
        self.model = HTTransformer(config).to(self.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        if 'model_state_dict' in ckpt:
            self.model.load_state_dict(ckpt['model_state_dict'])
        else:
            self.model.load_state_dict(ckpt)
        self.model.eval()
        logger.info(f"Loaded checkpoint: {checkpoint_path}")

        # Sphere radius for recovering coordinates
        self.sphere_radius = config['data'].get('sphere_radius', 25000.0)

    def predict(self, h5_path: str = None, output_dir: str = None) -> dict:
        """
        Run prediction on h5 file.

        Args:
            h5_path: path to h5 file (defaults to config h5_path)
            output_dir: directory to save results

        Returns:
            dict with predictions and metrics
        """
        if h5_path is None:
            h5_path = self.cfg['data']['h5_path']
        if output_dir is None:
            output_dir = os.path.join(self.cfg['output_path'],
                                     self.cfg['mission_name'], 'predict_results')
        os.makedirs(output_dir, exist_ok=True)

        # Create dataloader (use all data as test)
        _, _, test_loader = create_dataloaders(self.cfg, self.geometry, self.healpix)

        # Get kNN adj
        adj = self.healpix.get_knn_adjacency(k=self.cfg['model']['cd_knn_k'])
        adj_tensor = torch.from_numpy(adj).long().to(self.device)

        all_pred_u1 = []
        all_pred_u2 = []
        all_gt_u1 = []
        all_gt_u2 = []
        all_gt_p1 = []
        all_gt_p2 = []

        with torch.no_grad():
            for batch in test_loader:
                batch_gpu = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                             for k, v in batch.items()}
                batch_gpu['cd_knn_adj'] = adj_tensor

                outputs = self.model(batch_gpu)

                all_pred_u1.append(outputs['pred_u1'].cpu().numpy())
                all_pred_u2.append(outputs['pred_u2'].cpu().numpy())
                all_gt_u1.append(batch['u1'].numpy())
                all_gt_u2.append(batch['u2'].numpy())
                if 'p1' in batch:
                    all_gt_p1.append(batch['p1'].numpy())
                    all_gt_p2.append(batch['p2'].numpy())

        pred_u1 = np.concatenate(all_pred_u1)
        pred_u2 = np.concatenate(all_pred_u2)
        gt_u1 = np.concatenate(all_gt_u1)
        gt_u2 = np.concatenate(all_gt_u2)

        # Recover sphere coordinates
        pred_p1 = pred_u1 * self.sphere_radius
        pred_p2 = pred_u2 * self.sphere_radius

        # Save results
        np.savez(os.path.join(output_dir, 'predictions.npz'),
                 pred_u1=pred_u1, pred_u2=pred_u2,
                 pred_p1=pred_p1, pred_p2=pred_p2,
                 gt_u1=gt_u1, gt_u2=gt_u2,
                 gt_p1=np.concatenate(all_gt_p1) if all_gt_p1 else gt_u1,
                 gt_p2=np.concatenate(all_gt_p2) if all_gt_p2 else gt_u2)

        # CSV
        import pandas as pd
        df = pd.DataFrame({
            'pred_u1_x': pred_u1[:, 0], 'pred_u1_y': pred_u1[:, 1], 'pred_u1_z': pred_u1[:, 2],
            'pred_u2_x': pred_u2[:, 0], 'pred_u2_y': pred_u2[:, 1], 'pred_u2_z': pred_u2[:, 2],
            'pred_p1_x': pred_p1[:, 0], 'pred_p1_y': pred_p1[:, 1], 'pred_p1_z': pred_p1[:, 2],
            'pred_p2_x': pred_p2[:, 0], 'pred_p2_y': pred_p2[:, 1], 'pred_p2_z': pred_p2[:, 2],
            'gt_u1_x': gt_u1[:, 0], 'gt_u1_y': gt_u1[:, 1], 'gt_u1_z': gt_u1[:, 2],
            'gt_u2_x': gt_u2[:, 0], 'gt_u2_y': gt_u2[:, 1], 'gt_u2_z': gt_u2[:, 2],
        })
        csv_path = os.path.join(output_dir, 'predictions.csv')
        df.to_csv(csv_path, index=False)

        logger.info(f"Predictions saved to: {output_dir}")
        logger.info(f"  NPZ: predictions.npz")
        logger.info(f"  CSV: predictions.csv ({len(pred_u1)} events)")

        return {
            'pred_u1': pred_u1, 'pred_u2': pred_u2,
            'gt_u1': gt_u1, 'gt_u2': gt_u2,
            'pred_p1': pred_p1, 'pred_p2': pred_p2,
        }
