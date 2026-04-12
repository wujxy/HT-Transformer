"""
Loss Function for Ordered Dual-Endpoint Regression (Composite v2).

Four components:
1. L_ep:  Endpoint cosine loss — main supervision term
2. L_mid: Chord midpoint distance on unit sphere (SmoothL1)
3. L_dir: Track direction consistency (1 - cos, signed, no abs)
4. L_len: Normalized chord length regularizer (SmoothL1, /2.0)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EndpointLoss(nn.Module):
    """
    Composite loss for ordered dual-endpoint unit vector regression.

    L = lambda_ep * L_ep + lambda_mid * L_mid + lambda_dir * L_dir + lambda_len * L_len

    - L_ep:  Direct endpoint supervision (1 - cos), main anchor
    - L_mid: Midpoint position constraint on unit sphere, eliminates lateral drift
    - L_dir: Ordered direction consistency (signed, no abs)
    - L_len: Lightweight chord length regularizer
    """

    def __init__(
        self,
        lambda_ep: float = 1.0,
        lambda_mid: float = 0.5,
        lambda_dir: float = 0.25,
        lambda_len: float = 0.05,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.lambda_ep = lambda_ep
        self.lambda_mid = lambda_mid
        self.lambda_dir = lambda_dir
        self.lambda_len = lambda_len
        self.eps = eps

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
        eps = self.eps

        # --- 1. Endpoint cosine loss (main term) ---
        cos1 = (pred_u1 * gt_u1).sum(dim=-1).clamp(-1.0 + eps, 1.0 - eps)
        cos2 = (pred_u2 * gt_u2).sum(dim=-1).clamp(-1.0 + eps, 1.0 - eps)
        loss_ep = 0.5 * ((1.0 - cos1) + (1.0 - cos2)).mean()

        # --- 2. Midpoint distance on unit sphere ---
        pred_mid = 0.5 * (pred_u1 + pred_u2)
        gt_mid = 0.5 * (gt_u1 + gt_u2)
        mid_dist = (pred_mid - gt_mid).norm(dim=-1)
        loss_mid = F.smooth_l1_loss(mid_dist, torch.zeros_like(mid_dist))

        # --- 3. Directed track direction loss (signed, no abs) ---
        pred_dir = pred_u2 - pred_u1
        gt_dir = gt_u2 - gt_u1
        pred_dir_norm = pred_dir / (pred_dir.norm(dim=-1, keepdim=True) + eps)
        gt_dir_norm = gt_dir / (gt_dir.norm(dim=-1, keepdim=True) + eps)
        dir_cos = (pred_dir_norm * gt_dir_norm).sum(dim=-1).clamp(-1.0 + eps, 1.0 - eps)
        loss_dir = (1.0 - dir_cos).mean()

        # --- 4. Normalized chord length regularizer ---
        pred_len = (pred_u2 - pred_u1).norm(dim=-1)
        gt_len = (gt_u2 - gt_u1).norm(dim=-1)
        len_err = (pred_len - gt_len).abs() / 2.0
        loss_len = F.smooth_l1_loss(len_err, torch.zeros_like(len_err))

        # --- Total ---
        total = (
            self.lambda_ep * loss_ep
            + self.lambda_mid * loss_mid
            + self.lambda_dir * loss_dir
            + self.lambda_len * loss_len
        )

        loss_dict = {
            'loss_total': total.item(),
            'loss_ep': loss_ep.item(),
            'loss_mid': loss_mid.item(),
            'loss_dir': loss_dir.item(),
            'loss_len': loss_len.item(),
        }

        return total, loss_dict
