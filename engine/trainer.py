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
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from loguru import logger
from tqdm import tqdm

from models.ht_transformer import HTTransformer
from models.losses.endpoint_loss import EndpointLoss
from data.dataset import create_dataloaders
from geometry.detector_geometry import DualPMTPositionLookup
from geometry.healpix_mapper import HEALPixMapper

try:
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs as DDPInit
    HAS_ACCELERATE = True
except ImportError:
    HAS_ACCELERATE = False


def get_warmup_scheduler(optimizer, warmup_steps: int):
    """Simple linear warmup scheduler (for use before plateau)."""
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        return 1.0
    return LambdaLR(optimizer, lr_lambda)


class WarmupPlateauScheduler:
    """
    Combined warmup + ReduceLROnPlateau scheduler.

    Phase 1: Linear warmup for warmup_epochs
    Phase 2: ReduceLROnPlateau based on validation loss
    """

    def __init__(self, optimizer, warmup_epochs: int, plateau_factor: float = 0.5,
                 plateau_patience: int = 5, plateau_threshold: float = 1e-3,
                 plateau_min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        self.in_warmup = True

        # Warmup scheduler
        self.warmup_scheduler = get_warmup_scheduler(
            optimizer, warmup_epochs * 100)  # Approximate steps

        # Plateau scheduler (created after warmup)
        self.plateau_scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=plateau_factor,
            patience=plateau_patience,
            threshold=plateau_threshold,
            cooldown=1,
            min_lr=plateau_min_lr,
        )

    def step(self, metrics=None, epoch=None):
        """Step the scheduler."""
        if epoch is not None:
            self.current_epoch = epoch

        if self.current_epoch < self.warmup_epochs:
            self.in_warmup = True
            self.warmup_scheduler.step()
        else:
            self.in_warmup = False
            if metrics is not None:
                self.plateau_scheduler.step(metrics)

    def step_batch(self):
        """Step warmup scheduler per batch (during warmup phase only)."""
        if self.in_warmup:
            self.warmup_scheduler.step()

    def get_last_lr(self):
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']

    def state_dict(self):
        return {
            'warmup_scheduler': self.warmup_scheduler.state_dict(),
            'plateau_scheduler': self.plateau_scheduler.state_dict(),
            'current_epoch': self.current_epoch,
            'in_warmup': self.in_warmup,
        }

    def load_state_dict(self, state_dict):
        self.warmup_scheduler.load_state_dict(state_dict['warmup_scheduler'])
        self.plateau_scheduler.load_state_dict(state_dict['plateau_scheduler'])
        self.current_epoch = state_dict['current_epoch']
        self.in_warmup = state_dict['in_warmup']


class EarlyStopping:
    """
    Early stopping based on validation metrics.

    Monitors a metric and stops training if no improvement for patience epochs.
    """

    def __init__(self, patience: int = 8, mode: str = 'min', min_delta: float = 1e-4):
        """
        Args:
            patience: Number of epochs with no improvement before stopping
            mode: 'min' or 'max' - whether lower or higher is better
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

        if mode == 'min':
            self.is_better = lambda score, best: score < best - min_delta
            self.best_score = float('inf')
        else:
            self.is_better = lambda score, best: score > best + min_delta
            self.best_score = float('-inf')

    def __call__(self, score: float, epoch: int) -> bool:
        """
        Check if should stop.

        Returns:
            True if should stop, False otherwise
        """
        if self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
            self.best_epoch = epoch
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False

    def state_dict(self):
        return {
            'counter': self.counter,
            'best_score': self.best_score,
            'early_stop': self.early_stop,
            'best_epoch': self.best_epoch,
        }

    def load_state_dict(self, state_dict):
        self.counter = state_dict['counter']
        self.best_score = state_dict['best_score']
        self.early_stop = state_dict['early_stop']
        self.best_epoch = state_dict['best_epoch']


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
                kwargs_handlers=[DDPInit(find_unused_parameters=False)],
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

        # Scheduler: Warmup + ReduceLROnPlateau
        warmup_epochs = self.train_cfg.get('warmup_epochs', 5)
        self.scheduler = WarmupPlateauScheduler(
            self.optimizer,
            warmup_epochs=warmup_epochs,
            plateau_factor=self.train_cfg.get('plateau_factor', 0.5),
            plateau_patience=self.train_cfg.get('plateau_patience', 5),
            plateau_threshold=self.train_cfg.get('plateau_threshold', 1e-3),
            plateau_min_lr=self.train_cfg.get('plateau_min_lr', 1e-6),
        )
        logger.info(f"Using Warmup+Plateau scheduler: warmup_epochs={warmup_epochs}")

        # Early stopping (V2)
        if self.train_cfg.get('early_stop_patience', 0) > 0:
            self.early_stopping = EarlyStopping(
                patience=self.train_cfg['early_stop_patience'],
                mode=self.train_cfg.get('early_stop_mode', 'min'),
                min_delta=1e-4,
            )
            self.early_stop_monitor = self.train_cfg.get('early_stop_monitor', 'val_dir_ang_p68')
            logger.info(f"Early stopping enabled: monitor={self.early_stop_monitor}, "
                       f"patience={self.train_cfg['early_stop_patience']}")
        else:
            self.early_stopping = None
            self.early_stop_monitor = None

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
        self.current_epoch = 0

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
            "Model Architecture Summary (V2)",
            sep,
            f"  d_model={m.d_model}, num_layers={m.num_layers}, "
            f"num_heads={mc['num_heads']}, d_ff={mc['d_ff']}, "
            f"head_dim={m.d_model // mc['num_heads']}",
            f"  num_global_tokens={m.num_global}, num_queries={m.num_queries}, "
            f"cd_knn_k={mc['cd_knn_k']}, nside={dc['nside']}",
            "",
            f"  Architecture (V2 with DeepSphere):",
            f"    WP projector      : dual-branch [ux,uy,uz]⊕[q,t]",
            f"    CD encoder        : DeepSphere ({mc.get('cd_deepsphere_layers', 4)} layers)",
            f"    CD compression    : {mc.get('cd_compression', 'healpix_pool')} -> {mc.get('cd_fusion_tokens', 128)} tokens",
            f"    WP self-attn      : dense + signed time bias",
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

        # V2.1 architecture components
        wp_layers_params = count_params(m.wp_layers) if hasattr(m, 'wp_layers') else 0
        cd_cond_params = count_params(m.cd_conditioning) if hasattr(m, 'cd_conditioning') else 0
        readout_params = count_params(m.readout) if hasattr(m, 'readout') else 0
        components = [
            ("WP Projector      ", count_params(m.wp_projector)),
            ("CD Projector      ", count_params(m.cd_projector)),
            ("CD Encoder (DeepSphere)", count_params(m.cd_encoder) if hasattr(m, 'cd_encoder') else 0),
            ("CD Compression    ", count_params(m.cd_compression) if hasattr(m, 'cd_compression') else 0),
            ("WP Time Encoding  ", count_params(m.wp_time_encoding) if hasattr(m, 'wp_time_encoding') and m.wp_time_encoding is not None else 0),
            ("Type Embedding    ", count_params(m.type_embedding)),
            ("Position Encoding ", count_params(m.abs_pe)),
            ("WP Layers         ", wp_layers_params),
            ("CD Conditioning   ", cd_cond_params),
            ("CrossModal Readout", readout_params),
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

    def _verify_gradient_flow(self):
        """Verify that all model parameters receive gradients during backward pass."""
        logger.info("Verifying gradient flow...")
        self.model.train()

        # Get a single batch for testing
        test_batch = None
        for batch in self.train_loader:
            test_batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
            break

        if test_batch is None:
            logger.warning("Could not get batch for gradient verification")
            return

        # Forward pass
        outputs = self.model(test_batch)
        loss, _ = self.criterion(outputs['pred_u1'], outputs['pred_u2'],
                                  test_batch['u1'], test_batch['u2'])

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Check gradients
        unused = [name for name, p in self.model.named_parameters()
                 if p.grad is None and p.requires_grad]

        if unused:
            logger.warning(f"⚠ Parameters without gradients ({len(unused)}): {unused[:5]}...")
            logger.warning("find_unused_parameters=True is still needed")
        else:
            logger.info("✓ All parameters have gradients - find_unused_parameters=False is safe")

        # Clean up
        self.optimizer.zero_grad()

    def run(self):
        """Execute full training loop."""
        logger.info("=" * 50)
        logger.info("Starting training...")
        logger.info("=" * 50)

        # Verify gradient flow before training starts (all ranks must run to keep DDP in sync)
        self._verify_gradient_flow()

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
            lr = self.scheduler.get_last_lr()

            # Step scheduler (epoch-based)
            self.scheduler.step(metrics=val_metrics['loss_total'], epoch=epoch)

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
                self.current_epoch = epoch + 1

                # Log scheduler status
                phase = "warmup" if self.scheduler.in_warmup else "plateau"
                scheduler_status = f" [{phase}]"

                logger.info(
                    f"Epoch [{epoch+1}/{num_epochs}] "
                    f"Train loss: {train_metrics['loss_total']:.4f} "
                    f"(ang={train_metrics['loss_ang']:.4f} len={train_metrics['loss_len']:.4f} "
                    f"dir={train_metrics['loss_dir']:.4f}) "
                    f"Val loss: {val_metrics['loss_total']:.4f} "
                    f"LR: {lr:.6f}{scheduler_status} "
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

            # --- Early stopping check (V2) ---
            if self.early_stopping is not None and (epoch + 1) % eval_every == 0:
                # Get monitored metric from history
                metric_key = self.early_stop_monitor
                if metric_key in self.history and len(self.history[metric_key]) > 0:
                    metric_value = self.history[metric_key][-1]
                    should_stop = self.early_stopping(metric_value, epoch + 1)
                    if should_stop and self._is_main:
                        logger.info(
                            f"Early stopping triggered at epoch {epoch+1}. "
                            f"Best {metric_key}={self.early_stopping.best_score:.4f} "
                            f"at epoch {self.early_stopping.best_epoch}"
                        )
                        break

        # Training complete - record final status
        final_epoch = self.current_epoch  # Actual last epoch (may differ from num_epochs if early stopped)

        # Final save
        self._save_checkpoint('final.pth')
        if self._is_main:
            self._plot_training_curves()
        self._eval_and_plot(final_epoch - 1)  # Use actual final epoch
        if self._is_main:
            # Add early stopping info to history
            if self.early_stopping is not None:
                self.history['stopped_early'] = self.early_stopping.early_stop
                self.history['stopped_epoch'] = final_epoch if self.early_stopping.early_stop else None
                self.history['best_epoch'] = self.early_stopping.best_epoch
                self.history['best_monitor_value'] = self.early_stopping.best_score
                self.history['early_stop_monitor'] = self.early_stop_monitor
            else:
                self.history['stopped_early'] = False
                self.history['stopped_epoch'] = None
                self.history['best_epoch'] = final_epoch
                self.history['best_monitor_value'] = None

            self._save_history()

            # Log completion status
            if self.early_stopping is not None and self.early_stopping.early_stop:
                logger.info(
                    f"Training complete! Early stopped at epoch {final_epoch}. "
                    f"Best {self.early_stop_monitor}={self.early_stopping.best_score:.4f} "
                    f"at epoch {self.early_stopping.best_epoch}"
                )
            else:
                logger.info(f"Training complete! Finished all {final_epoch} epochs.")

    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()
        total_loss = 0
        total_ang = 0
        total_len = 0
        total_dir = 0
        n_batches = 0

        # Profiling accumulators (ms)
        t_data_total = 0.0   # dataloader wait + collate (before h2d)
        t_h2d_total = 0.0    # host→device transfer
        t_fwd_total = 0.0    # model forward + loss
        t_bwd_total = 0.0    # backward + optimizer step

        total_steps = len(self.train_loader)
        pbar = tqdm(self.train_loader, total=total_steps,
                    desc=f"Epoch {epoch+1}", disable=not self._is_main)

        for batch in pbar:
            t0 = time.time()

            # --- H2D transfer ---
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            t1 = time.time()

            if self.use_accelerate:
                with self.accelerator.accumulate(self.model):
                    outputs = self.model(batch)
                    loss, loss_dict = self.criterion(
                        outputs['pred_u1'], outputs['pred_u2'],
                        batch['u1'], batch['u2']
                    )
                    t2 = time.time()

                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        grad_clip = self.train_cfg.get('grad_clip', 1.0)
                        self.accelerator.clip_grad_norm_(self.model.parameters(), grad_clip)
                        self.optimizer.step()
                        self.scheduler.step_batch()
                        self.optimizer.zero_grad()
                    t3 = time.time()
            else:
                with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=self.amp_dtype):
                    outputs = self.model(batch)
                    loss, loss_dict = self.criterion(
                        outputs['pred_u1'], outputs['pred_u2'],
                        batch['u1'], batch['u2']
                    )
                    t2 = time.time()

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                grad_clip = self.train_cfg.get('grad_clip', 1.0)
                nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step_batch()
                t3 = time.time()

            total_loss += loss_dict['loss_total']
            total_ang += loss_dict['loss_ang']
            total_len += loss_dict['loss_len']
            total_dir += loss_dict['loss_dir']
            n_batches += 1
            self.global_step += 1

            # Accumulate profiling (t0 = after dataloader yield, t1 = after h2d, etc.)
            dt_h2d = (t1 - t0) * 1000
            dt_fwd = (t2 - t1) * 1000
            dt_bwd = (t3 - t2) * 1000
            t_h2d_total += dt_h2d
            t_fwd_total += dt_fwd
            t_bwd_total += dt_bwd

            if self._is_main:
                if n_batches <= 5 or n_batches % 20 == 0:
                    pbar.write(f"[prof] h2d={dt_h2d:.0f}ms  fwd={dt_fwd:.0f}ms  "
                               f"bwd={dt_bwd:.0f}ms  loss={loss_dict['loss_total']:.4f}")
                pbar.set_postfix(loss=f"{loss_dict['loss_total']:.4f}",
                                 ang=f"{loss_dict['loss_ang']:.4f}")

        if self._is_main and n_batches > 0:
            logger.info(
                f"[prof epoch] avg/step: h2d={t_h2d_total/n_batches:.0f}ms "
                f"fwd={t_fwd_total/n_batches:.0f}ms "
                f"bwd={t_bwd_total/n_batches:.0f}ms "
                f"(total={t_h2d_total+t_fwd_total+t_bwd_total:.0f}ms "
                f"for {n_batches} steps)"
            )

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
        from metrics.endpoint_metrics import compute_training_metrics
        from visualization.plotting import plot_training_eval_distributions

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
