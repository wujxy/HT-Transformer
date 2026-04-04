"""
Loss Function for Ordered Dual-Endpoint Regression.

Three components:
1. L_ang: Angular loss (1 - cos similarity) per endpoint
2. L_len: Track length constraint (SmoothL1)
3. L_dir: Direction consistency between endpoints
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EndpointLoss(nn.Module):
    """
    Combined loss for ordered dual-endpoint unit vector regression.

    L = lambda_ang * L_ang + lambda_len * L_len + lambda_dir * L_dir
    """

    def __init__(self, lambda_ang: float = 1.0, lambda_len: float = 0.5,
                 lambda_dir: float = 0.25):
        super().__init__()
        self.lambda_ang = lambda_ang
        self.lambda_len = lambda_len
        self.lambda_dir = lambda_dir

    def forward(self, pred_u1: torch.Tensor, pred_u2: torch.Tensor,
                gt_u1: torch.Tensor, gt_u2: torch.Tensor):
        """
        Args:
            pred_u1: (B, 3) predicted first endpoint unit vector
            pred_u2: (B, 3) predicted second endpoint unit vector
            gt_u1: (B, 3) ground truth first endpoint unit vector
            gt_u2: (B, 3) ground truth second endpoint unit vector

        Returns:
            total_loss: scalar
            loss_dict: dict with individual loss components for logging
        """
        eps = 1e-8

        # --- 1. Angular loss (stable 1-cos form) ---
        cos1 = (pred_u1 * gt_u1).sum(dim=-1).clamp(-1.0 + eps, 1.0 - eps)
        cos2 = (pred_u2 * gt_u2).sum(dim=-1).clamp(-1.0 + eps, 1.0 - eps)
        L_ang = 0.5 * ((1.0 - cos1) + (1.0 - cos2))

        # --- 2. Track length constraint ---
        pred_dist = (pred_u2 - pred_u1).norm(dim=-1)
        gt_dist = (gt_u2 - gt_u1).norm(dim=-1)
        L_len = F.smooth_l1_loss(pred_dist, gt_dist)

        # --- 3. Direction consistency ---
        pred_dir = pred_u2 - pred_u1
        gt_dir = gt_u2 - gt_u1
        pred_dir_norm = pred_dir / (pred_dir.norm(dim=-1, keepdim=True) + eps)
        gt_dir_norm = gt_dir / (gt_dir.norm(dim=-1, keepdim=True) + eps)
        dir_cos = (pred_dir_norm * gt_dir_norm).sum(dim=-1).clamp(-1.0 + eps, 1.0 - eps)
        L_dir = 1.0 - dir_cos

        # --- Total ---
        total = (self.lambda_ang * L_ang.mean() +
                 self.lambda_len * L_len +
                 self.lambda_dir * L_dir.mean())

        loss_dict = {
            'loss_total': total.item(),
            'loss_ang': L_ang.mean().item(),
            'loss_len': L_len.item(),
            'loss_dir': L_dir.mean().item(),
        }

        return total, loss_dict
