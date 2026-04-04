"""
Training Loop for HT-Transformer Endpoint Reconstruction.

Follows reference project ModelTrain.py framework style.
"""

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from loguru import logger
from tqdm import tqdm

from Model import HTTransformer
from LossFunction import EndpointLoss
from DataLoader import create_dataloaders
from Geometry import DualPMTPositionLookup
from HEALPix import HEALPixMapper

try:
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs as DDPInit
    HAS_ACCELERATE = True
except ImportError:
    HAS_ACCELERATE = False


def get_cosine_with_warmup_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Cosine annealing with linear warmup."""
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


class Trainer:
    """
    Training orchestration for HT-Transformer.

    Follows reference project pattern:
    - __init__ sets up everything
    - run() executes training
    - Checkpoint save/load
    - Loss logging + plot
    """

    def __init__(self, config: dict):
        self.cfg = config
        self.train_cfg = config['train']
        self.model_cfg = config['model']
        self.data_cfg = config['data']
        self.loss_cfg = config['loss']

        # Setup output dirs
        mission = config.get('mission_name', 'ht_transformer_v1')
        output_base = config.get('output_path', 'output')
        self.output_dir = os.path.join(output_base, mission)
        self.ckpt_dir = os.path.join(self.output_dir, 'checkpoints')
        self.plot_dir = os.path.join(self.output_dir, 'plots')
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)

        # Accelerate vs manual
        self.use_accelerate = self.train_cfg.get('use_accelerate', False)
        if self.use_accelerate and not HAS_ACCELERATE:
            logger.warning("accelerate not installed, falling back to manual mode")
            self.use_accelerate = False

        if self.use_accelerate:
            logger.info("Initializing Accelerator...")
            grad_accum = self.train_cfg.get('gradient_accumulation_steps', 1)
            precision = self.train_cfg.get('precision', 'bf16')
            mp = 'bf16' if precision == 'bf16' else ('fp16' if precision == 'fp16' else 'no')
            self.accelerator = Accelerator(
                gradient_accumulation_steps=grad_accum,
                mixed_precision=mp,
                kwargs_handlers=[DDPInit(find_unused_parameters=True)],
            )
            self.device = self.accelerator.device
            logger.info(f"Accelerator initialized: device={self.device}, "
                        f"num_processes={self.accelerator.num_processes}, "
                        f"process_index={self.accelerator.process_index}")
        else:
            self.accelerator = None
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._is_main = (not self.use_accelerate) or self.accelerator.is_main_process
        logger.info(f"Device: {self.device}")

        # Seed
        seed = self.train_cfg.get('seed', 42)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # For accelerate, set different seed per process for dataloader workers
        if self.use_accelerate:
            import random
            worker_seed = seed + self.accelerator.process_index
            random.seed(worker_seed)

        # Geometry + HEALPix
        logger.info("Loading geometry...")
        self.geometry = DualPMTPositionLookup(
            self.data_cfg['geometry_cd'],
            self.data_cfg['geometry_wp'],
        )

        # Build CD unit vecs for HEALPix lookup
        cd_max = len(self.geometry.cd_position_array)
        cd_unit_vecs = self.geometry.cd_position_array.copy()
        norms = np.linalg.norm(cd_unit_vecs, axis=1, keepdims=True)
        valid = (norms.squeeze() > 0)
        cd_unit_vecs[valid] = cd_unit_vecs[valid] / norms[valid]

        self.healpix = HEALPixMapper(
            nside=self.data_cfg['nside'],
            cd_unit_vecs=cd_unit_vecs,
        )

        # Build kNN adjacency
        self.healpix.build_knn_adjacency(k=self.model_cfg['cd_knn_k'])

        # Data
        logger.info("Creating dataloaders...")
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            config, self.geometry, self.healpix)

        # Model
        logger.info("Building model...")
        self.model = HTTransformer(config).to(self.device)
        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model parameters: {n_params:,}")

        if self._is_main:
            self._log_model_summary()

        # Loss
        self.criterion = EndpointLoss(
            lambda_ang=self.loss_cfg['lambda_ang'],
            lambda_len=self.loss_cfg['lambda_len'],
            lambda_dir=self.loss_cfg['lambda_dir'],
        )

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.train_cfg['lr'],
            weight_decay=self.train_cfg['weight_decay'],
        )

        # Scheduler
        total_steps = self.train_cfg['num_epochs'] * len(self.train_loader)
        warmup = self.train_cfg.get('warmup_steps', 500)
        self.scheduler = get_cosine_with_warmup_scheduler(
            self.optimizer, warmup, total_steps)
        logger.info(f"Total steps: {total_steps}, warmup: {warmup}")

        # AMP (mixed precision) — only used when NOT using accelerate
        if not self.use_accelerate:
            self.use_amp = self.train_cfg.get('precision', 'fp32') in ('bf16', 'fp16')
            self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
            self.amp_dtype = torch.bfloat16 if self.train_cfg.get('precision') == 'bf16' else torch.float16
        else:
            self.use_amp = False
            self.scaler = None
            self.amp_dtype = None

        # Training state
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_ang': [], 'train_len': [], 'train_dir': [],
            'val_ang': [], 'val_len': [], 'val_dir': [],
            'lr': [],
        }
        self.best_val_loss = float('inf')
        self.global_step = 0

        # Add kNN adjacency to config for model access
        self._cd_knn_adj_tensor = None

        # Accelerate prepare (must be after model/optimizer/dataloader creation)
        if self.use_accelerate:
            self.model, self.optimizer, self.train_loader, self.val_loader, self.test_loader = \
                self.accelerator.prepare(
                    self.model, self.optimizer,
                    self.train_loader, self.val_loader, self.test_loader)
            logger.info("accelerator.prepare() done for model, optimizer, dataloaders")

    def _log_model_summary(self):
        """Log model architecture summary with parameter breakdown."""
        m = self.model
        mc = self.model_cfg
        dc = self.data_cfg
        sep = "=" * 70

        lines = [
            sep,
            "Model Architecture Summary",
            sep,
            f"  d_model={m.d_model}, num_layers={m.num_layers}, "
            f"num_heads={mc['num_heads']}, d_ff={mc['d_ff']}, "
            f"head_dim={m.d_model // mc['num_heads']}",
            f"  num_global_tokens={m.num_global}, num_queries={m.num_queries}, "
            f"cd_knn_k={mc['cd_knn_k']}, nside={dc['nside']}",
            "",
            f"  Per-layer attention modules (x{m.num_layers}):",
            f"    WP self-attn      : dense, (B, H, N_wp, N_wp)",
            f"    CD self-attn      : kNN (k={mc['cd_knn_k']}), (B, H, N_cd, N_cd)"
            + (" + RPE" if mc.get('rel_posenc') == 'bucket' else ""),
            f"    WP->CD cross-attn : dense, (B, H, N_wp, N_cd)",
            f"    CD->WP cross-attn : dense, (B, H, N_cd, N_wp)",
            f"    Global->All attn  : dense, (B, H, {m.num_global}, N_wp+N_cd)",
            f"    Query->All attn   : dense, (B, H, {m.num_queries}, N_wp+N_cd+{m.num_global})",
            f"    FFN               : Linear({m.d_model}->{mc['d_ff']}) -> GELU -> Linear({mc['d_ff']}->{m.d_model})",
            "",
            "  Parameter breakdown:",
        ]

        # Compute parameter counts per component
        def count_params(module):
            return sum(p.numel() for p in module.parameters())

        components = [
            ("Token Projectors  ", count_params(m.wp_projector) + count_params(m.cd_projector)),
            ("Type Embedding    ", count_params(m.type_embedding)),
            ("Position Encoding ", count_params(m.abs_pe)),
            ("Encoder Layers    ", count_params(m.encoder_layers)),
            ("  - Attention     ", sum(
                count_params(getattr(layer, name))
                for layer in m.encoder_layers
                for name in ['wp_self_attn', 'cd_self_attn', 'wp_cd_cross',
                             'cd_wp_cross', 'global_attn', 'query_attn']
            )),
            ("  - FFN           ", sum(
                count_params(getattr(layer, name))
                for layer in m.encoder_layers
                for name in ['wp_ffn', 'cd_ffn', 'global_ffn', 'query_ffn']
            )),
            ("  - LayerNorm     ", sum(
                count_params(getattr(layer, name))
                for layer in m.encoder_layers
                for name in ['wp_attn_norm', 'cd_attn_norm', 'wp_cross_norm',
                             'cd_cross_norm', 'global_norm', 'query_norm',
                             'wp_ffn_norm', 'cd_ffn_norm', 'global_ffn_norm',
                             'query_ffn_norm']
            )),
            ("  - RPE           ", sum(
                count_params(layer.rpe) for layer in m.encoder_layers
                if hasattr(layer, 'rpe') and layer.rpe is not None
            )),
            ("Output Heads      ", count_params(m.head1) + count_params(m.head2)),
            ("Learnable Tokens  ", count_params(nn.ParameterList([m.global_tokens, m.query_tokens]))),
        ]

        total = 0
        for name, count in components:
            if not name.startswith("  -"):
                total += count
            lines.append(f"    {name}: {count:>10,}")

        lines.append(f"    {'-' * 43}")
        lines.append(f"    {'Total            '}: {total:>10,}")
        lines.append(sep)

        for line in lines:
            logger.info(line)

    def _get_knn_adj_tensor(self, N_cd: int) -> torch.Tensor:
        """Get kNN adjacency tensor sized for current batch."""
        if self._cd_knn_adj_tensor is None:
            adj = self.healpix.get_knn_adjacency(k=self.model_cfg['cd_knn_k'])
            self._cd_knn_adj_tensor = torch.from_numpy(adj).long()
        return self._cd_knn_adj_tensor[:N_cd].to(self.device)

    def run(self):
        """Execute full training loop."""
        logger.info("=" * 50)
        logger.info("Starting training...")
        logger.info("=" * 50)

        num_epochs = self.train_cfg['num_epochs']
        eval_every = self.train_cfg.get('eval_every', 10)
        save_every = self.train_cfg.get('save_every', 50)

        for epoch in range(num_epochs):
            t0 = time.time()

            # --- Train ---
            train_metrics = self._train_epoch(epoch)
            t_train = time.time() - t0

            # --- Validate ---
            val_metrics = self._val_epoch()
            t_val = time.time() - t0 - t_train

            # --- Log ---
            lr = self.optimizer.param_groups[0]['lr']
            if self._is_main:
                self.history['train_loss'].append(train_metrics['loss_total'])
                self.history['val_loss'].append(val_metrics['loss_total'])
                self.history['train_ang'].append(train_metrics['loss_ang'])
                self.history['train_len'].append(train_metrics['loss_len'])
                self.history['train_dir'].append(train_metrics['loss_dir'])
                self.history['val_ang'].append(val_metrics['loss_ang'])
                self.history['val_len'].append(val_metrics['loss_len'])
                self.history['val_dir'].append(val_metrics['loss_dir'])
                self.history['lr'].append(lr)

                logger.info(
                    f"Epoch [{epoch+1}/{num_epochs}] "
                    f"Train loss: {train_metrics['loss_total']:.4f} "
                    f"(ang={train_metrics['loss_ang']:.4f} len={train_metrics['loss_len']:.4f} "
                    f"dir={train_metrics['loss_dir']:.4f}) "
                    f"Val loss: {val_metrics['loss_total']:.4f} "
                    f"LR: {lr:.6f} "
                    f"Time: {t_train:.1f}s+{t_val:.1f}s"
                )

            # --- Save best ---
            if val_metrics['loss_total'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss_total']
                self._save_checkpoint('best.pth')
                if self._is_main:
                    logger.info(f"  -> New best val loss: {self.best_val_loss:.4f}")

            # --- Periodic save ---
            if (epoch + 1) % save_every == 0:
                self._save_checkpoint(f'epoch_{epoch+1}.pth')

            # --- Periodic eval + plot ---
            if (epoch + 1) % eval_every == 0:
                if self._is_main:
                    self._plot_training_curves()
                self._eval_and_plot(epoch)

        # Final save
        self._save_checkpoint('final.pth')
        if self._is_main:
            self._plot_training_curves()
        self._eval_and_plot(num_epochs - 1)
        if self._is_main:
            self._save_history()
            logger.info("Training complete!")

    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()
        total_loss = 0
        total_ang = 0
        total_len = 0
        total_dir = 0
        n_batches = 0

        total_steps = len(self.train_loader)
        pbar = tqdm(self.train_loader, total=total_steps,
                    desc=f"Epoch {epoch+1}", disable=not self._is_main)

        for batch in pbar:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # Add kNN adj
            N_cd = batch['cd_unit_vecs'].shape[1]
            batch['cd_knn_adj'] = self._get_knn_adj_tensor(N_cd)

            if self.use_accelerate:
                with self.accelerator.accumulate(self.model):
                    outputs = self.model(batch)
                    loss, loss_dict = self.criterion(
                        outputs['pred_u1'], outputs['pred_u2'],
                        batch['u1'], batch['u2']
                    )
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        grad_clip = self.train_cfg.get('grad_clip', 1.0)
                        self.accelerator.clip_grad_norm_(self.model.parameters(), grad_clip)
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()
            else:
                with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=self.amp_dtype):
                    outputs = self.model(batch)
                    loss, loss_dict = self.criterion(
                        outputs['pred_u1'], outputs['pred_u2'],
                        batch['u1'], batch['u2']
                    )

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                grad_clip = self.train_cfg.get('grad_clip', 1.0)
                nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()

            total_loss += loss_dict['loss_total']
            total_ang += loss_dict['loss_ang']
            total_len += loss_dict['loss_len']
            total_dir += loss_dict['loss_dir']
            n_batches += 1
            self.global_step += 1

            if self._is_main:
                pbar.set_postfix(loss=f"{loss_dict['loss_total']:.4f}",
                                 ang=f"{loss_dict['loss_ang']:.4f}")

        return {
            'loss_total': total_loss / max(1, n_batches),
            'loss_ang': total_ang / max(1, n_batches),
            'loss_len': total_len / max(1, n_batches),
            'loss_dir': total_dir / max(1, n_batches),
        }

    @torch.no_grad()
    def _val_epoch(self) -> dict:
        self.model.eval()
        total_loss = 0.0
        total_ang = 0.0
        total_len = 0.0
        total_dir = 0.0
        n_batches = 0

        val_pbar = tqdm(self.val_loader, total=len(self.val_loader),
                        desc="  Val", disable=not self._is_main)

        for batch in val_pbar:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            N_cd = batch['cd_unit_vecs'].shape[1]
            batch['cd_knn_adj'] = self._get_knn_adj_tensor(N_cd)

            if self.use_accelerate:
                outputs = self.model(batch)
            else:
                with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=self.amp_dtype):
                    outputs = self.model(batch)
            loss, loss_dict = self.criterion(
                outputs['pred_u1'], outputs['pred_u2'],
                batch['u1'], batch['u2']
            )

            total_loss += loss_dict['loss_total']
            total_ang += loss_dict['loss_ang']
            total_len += loss_dict['loss_len']
            total_dir += loss_dict['loss_dir']
            n_batches += 1

            if self._is_main:
                val_pbar.set_postfix(loss=f"{loss_dict['loss_total']:.4f}")

        if self.use_accelerate:
            # Gather loss sums across GPUs
            device = self.accelerator.device
            gathered_loss = self.accelerator.gather(torch.tensor(total_loss, device=device))
            gathered_ang = self.accelerator.gather(torch.tensor(total_ang, device=device))
            gathered_len = self.accelerator.gather(torch.tensor(total_len, device=device))
            gathered_dir = self.accelerator.gather(torch.tensor(total_dir, device=device))
            gathered_n = self.accelerator.gather(torch.tensor(n_batches, device=device))
            return {
                'loss_total': gathered_loss.sum().item() / max(1, gathered_n.sum().item()),
                'loss_ang': gathered_ang.sum().item() / max(1, gathered_n.sum().item()),
                'loss_len': gathered_len.sum().item() / max(1, gathered_n.sum().item()),
                'loss_dir': gathered_dir.sum().item() / max(1, gathered_n.sum().item()),
            }
        else:
            return {
                'loss_total': total_loss / max(1, n_batches),
                'loss_ang': total_ang / max(1, n_batches),
                'loss_len': total_len / max(1, n_batches),
                'loss_dir': total_dir / max(1, n_batches),
            }

    @torch.no_grad()
    def _predict_val(self) -> dict:
        """Run full inference on validation set, collecting all predictions."""
        # Non-main processes skip; caller must handle None return
        if self.use_accelerate and not self._is_main:
            return None

        model = self.accelerator.unwrap_model(self.model) if self.use_accelerate else self.model
        model.eval()
        all_pred_u1, all_pred_u2 = [], []
        all_gt_u1, all_gt_u2 = [], []

        for batch in self.val_loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            N_cd = batch['cd_unit_vecs'].shape[1]
            batch['cd_knn_adj'] = self._get_knn_adj_tensor(N_cd)

            outputs = model(batch)

            all_pred_u1.append(outputs['pred_u1'].cpu().numpy())
            all_pred_u2.append(outputs['pred_u2'].cpu().numpy())
            all_gt_u1.append(batch['u1'].cpu().numpy())
            all_gt_u2.append(batch['u2'].cpu().numpy())

        return {
            'pred_u1': np.concatenate(all_pred_u1),
            'pred_u2': np.concatenate(all_pred_u2),
            'gt_u1': np.concatenate(all_gt_u1),
            'gt_u2': np.concatenate(all_gt_u2),
        }

    def _eval_and_plot(self, epoch: int):
        """Run reconstruction metrics computation and plot distributions."""
        from Metrics import compute_training_metrics
        from Plotting import plot_training_eval_distributions

        sphere_radius = self.data_cfg.get('sphere_radius', 25000.0)

        # Collect predictions (returns None on non-main processes)
        preds = self._predict_val()
        if preds is None:
            return

        # Compute metrics
        metrics = compute_training_metrics(
            preds['pred_u1'], preds['pred_u2'],
            preds['gt_u1'], preds['gt_u2'],
            sphere_radius=sphere_radius,
        )

        # Log key metrics
        logger.info(
            f"  Eval metrics @ epoch {epoch+1}: "
            f"dir_ang_p68={metrics['dir_ang_p68']:.2f}deg "
            f"dir_ang_p90={metrics['dir_ang_p90']:.2f}deg "
            f"mid_dist_p68={metrics['mid_dist_p68']:.1f}mm "
            f"mid_dist_p90={metrics['mid_dist_p90']:.1f}mm"
        )

        # Store in history for trend plots
        self.history.setdefault('val_dir_ang_p68', []).append(metrics['dir_ang_p68'])
        self.history.setdefault('val_dir_ang_p90', []).append(metrics['dir_ang_p90'])
        self.history.setdefault('val_mid_dist_p68', []).append(metrics['mid_dist_p68'])
        self.history.setdefault('val_mid_dist_p90', []).append(metrics['mid_dist_p90'])
        self.history.setdefault('eval_epochs', []).append(epoch + 1)

        # Generate distribution plots
        plot_training_eval_distributions(
            metrics, self.plot_dir, epoch=epoch + 1, prefix="val"
        )

    def _save_checkpoint(self, filename: str):
        if not self._is_main:
            return
        path = os.path.join(self.ckpt_dir, filename)
        model_state = self.accelerator.unwrap_model(self.model).state_dict() \
            if self.use_accelerate else self.model.state_dict()
        torch.save({
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'config': self.cfg,
        }, path)
        logger.info(f"Checkpoint saved: {path}")

    def _save_history(self):
        path = os.path.join(self.output_dir, 'training_history.json')
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Training history saved: {path}")

    def _plot_training_curves(self):
        """Save loss curves + reconstruction metric trend plots."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            # Check if we have reconstruction metrics in history
            has_recon = 'val_dir_ang_p68' in self.history and len(self.history['val_dir_ang_p68']) > 0

            if has_recon:
                fig, axes = plt.subplots(3, 2, figsize=(12, 15))
            else:
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            epochs = range(1, len(self.history['train_loss']) + 1)

            # Total loss
            axes[0, 0].plot(epochs, self.history['train_loss'], label='Train')
            axes[0, 0].plot(epochs, self.history['val_loss'], label='Val')
            axes[0, 0].set_title('Total Loss')
            axes[0, 0].legend()
            axes[0, 0].set_xlabel('Epoch')

            # Angle loss
            axes[0, 1].plot(epochs, self.history['train_ang'], label='Train')
            axes[0, 1].plot(epochs, self.history['val_ang'], label='Val')
            axes[0, 1].set_title('Angle Loss')
            axes[0, 1].legend()

            # Length + Dir loss
            axes[1, 0].plot(epochs, self.history['train_len'], label='Len(train)')
            axes[1, 0].plot(epochs, self.history['val_len'], label='Len(val)')
            axes[1, 0].set_title('Length Loss')
            axes[1, 0].legend()

            axes[1, 1].plot(epochs, self.history['train_dir'], label='Dir(train)')
            axes[1, 1].plot(epochs, self.history['val_dir'], label='Dir(val)')
            axes[1, 1].set_title('Direction Loss')
            axes[1, 1].legend()

            # Reconstruction metric trends (row 3)
            if has_recon:
                eval_epochs = self.history['eval_epochs']
                axes[2, 0].plot(eval_epochs, self.history['val_dir_ang_p68'],
                                'o-', label='p68', markersize=3)
                axes[2, 0].plot(eval_epochs, self.history['val_dir_ang_p90'],
                                's-', label='p90', markersize=3)
                axes[2, 0].set_title('Direction Angle Error (deg)')
                axes[2, 0].set_xlabel('Epoch')
                axes[2, 0].legend()

                axes[2, 1].plot(eval_epochs, self.history['val_mid_dist_p68'],
                                'o-', label='p68', markersize=3)
                axes[2, 1].plot(eval_epochs, self.history['val_mid_dist_p90'],
                                's-', label='p90', markersize=3)
                axes[2, 1].set_title('Midpoint Distance (mm)')
                axes[2, 1].set_xlabel('Epoch')
                axes[2, 1].legend()

            plt.tight_layout()
            path = os.path.join(self.plot_dir, 'training_curves.png')
            plt.savefig(path, dpi=150)
            plt.close()
            logger.info(f"Training curves saved: {path}")
        except ImportError:
            logger.warning("matplotlib not available, skipping plot")
