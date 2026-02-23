#!/usr/bin/env python3
"""
Training script for Dual-Encoder Cross-Attention U-Net
Handles multi-GPU training, validation, and checkpointing
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dual_encoder_model import DualEncoderCrossAttentionUNet, combined_loss, soft_dice_loss
from dataset import PairedPETCTDataset


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = os.path.join(args.output_dir, f'fold_{args.fold}_{timestamp}')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save configuration
        with open(os.path.join(self.output_dir, 'config.json'), 'w') as f:
            json.dump(vars(args), f, indent=2)
        
        # Initialize model
        self.model = DualEncoderCrossAttentionUNet(
            base_channels=args.base_channels,
            cross_attention=args.cross_attention,
            disagreement_attention=args.disagreement_attention,
            num_heads=args.num_heads,
        )

        # Print parameter count
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {n_params:,}")
        
        # Multi-GPU support
        if torch.cuda.device_count() > 1 and args.multi_gpu:
            print(f"Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)
        
        self.model = self.model.to(self.device)
        
        # Initialize datasets
        self.train_dataset = PairedPETCTDataset(
            root_dir=args.data_dir,
            split='train',
            fold=args.fold,
            splits_file=args.splits_file,
            patch_size=tuple(args.patch_size),
            samples_per_epoch=args.samples_per_epoch,
            foreground_ratio=args.foreground_ratio,
            augment=True,
            cache_data=args.cache_data
        )
        
        self.val_dataset = PairedPETCTDataset(
            root_dir=args.data_dir,
            split='val',
            fold=args.fold,
            splits_file=args.splits_file,
            patch_size=None,  # Full volume for validation
            augment=False,
            cache_data=args.cache_data
        )
        
        # Data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=1,  # Full volume validation
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        # Optimizer and scheduler
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()
        
        # Mixed precision training
        self.scaler = GradScaler(enabled=args.amp)
        
        # Logging
        self.writer = SummaryWriter(os.path.join(self.output_dir, 'tensorboard'))
        
        # Training state
        self.epoch = 0
        self.best_val_dice = 0.0
        self.patience_counter = 0
        
        # Class weights for loss (only used in dice_ce mode)
        self.class_weights = torch.tensor([0.1, 2.0, 0.5], device=self.device)

        # Loss configuration
        self.loss_mode = args.loss_mode
        self.psma_loss_weight = args.psma_loss_weight
        self.fdg_loss_weight = args.fdg_loss_weight

        # Cross-attention warmup
        self.cross_attn_warmup_epochs = args.cross_attn_warmup
        
    def _get_optimizer(self):
        """Initialize optimizer"""
        if self.args.optimizer == 'adam':
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay
            )
        elif self.args.optimizer == 'sgd':
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.args.learning_rate,
                momentum=0.99,
                nesterov=True,
                weight_decay=self.args.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.args.optimizer}")
    
    def _get_scheduler(self):
        """Initialize learning rate scheduler"""
        if self.args.scheduler == 'poly':
            return torch.optim.lr_scheduler.PolynomialLR(
                self.optimizer,
                total_iters=self.args.epochs,
                power=0.9
            )
        elif self.args.scheduler == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.args.epochs
            )
        else:
            return None
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        epoch_losses = {'total': 0, 'psma': 0, 'fdg': 0, 'dice_psma': 0, 'dice_fdg': 0}
        
        # Calculate cross-attention weight (gradual warmup)
        cross_attn_weight = min(1.0, self.epoch / max(1, self.cross_attn_warmup_epochs))
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch}/{self.args.epochs}')
        
        for batch_idx, batch in enumerate(pbar):
            psma = batch['psma'].to(self.device)
            fdg = batch['fdg'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Mixed precision forward pass
            with autocast(enabled=self.args.amp):
                model_out = self.model(psma, fdg, cross_attn_weight)
                # Handle both return formats (with/without disagree maps)
                if isinstance(model_out, tuple) and len(model_out) == 3:
                    psma_logits, fdg_logits, _ = model_out
                else:
                    psma_logits, fdg_logits = model_out
                losses = combined_loss(
                    psma_logits, fdg_logits, labels, self.class_weights,
                    loss_mode=self.loss_mode,
                    psma_weight=self.psma_loss_weight,
                    fdg_weight=self.fdg_loss_weight,
                )
            
            # Backward pass
            self.optimizer.zero_grad()
            self.scaler.scale(losses['total']).backward()
            
            # Gradient clipping
            if self.args.gradient_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.gradient_clip)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Update metrics
            for key in epoch_losses:
                if key in losses:
                    epoch_losses[key] += losses[key].item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': losses['total'].item(),
                'dice_p': 1 - losses['dice_psma'].item(),
                'dice_f': 1 - losses['dice_fdg'].item(),
                'lr': self.optimizer.param_groups[0]['lr'],
                'attn': f'{cross_attn_weight:.2f}'
            })
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= len(self.train_loader)
        
        return epoch_losses
    
    def validate(self):
        """Validate on full volumes with sliding window"""
        self.model.eval()
        val_losses = {'total': 0, 'psma': 0, 'fdg': 0}
        dice_scores = {'psma_tumor': [], 'psma_normal': [], 'fdg_tumor': [], 'fdg_normal': []}
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                psma = batch['psma'].to(self.device)
                fdg = batch['fdg'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Sliding window inference for large volumes
                psma_logits, fdg_logits = self.sliding_window_inference(
                    psma, fdg,
                    patch_size=self.args.patch_size,
                    overlap=0.5
                )

                # Compute losses
                losses = combined_loss(
                    psma_logits, fdg_logits, labels, self.class_weights,
                    loss_mode=self.loss_mode,
                    psma_weight=self.psma_loss_weight,
                    fdg_weight=self.fdg_loss_weight,
                )
                for key in val_losses:
                    if key in losses:
                        val_losses[key] += losses[key].item()
                
                # Compute dice scores
                psma_pred = psma_logits.argmax(dim=1)
                fdg_pred = fdg_logits.argmax(dim=1)
                
                # Map predictions back to 5-class
                combined_pred = torch.zeros_like(labels)
                combined_pred[psma_pred == 1] = 1  # PSMA tumor
                combined_pred[psma_pred == 2] = 2  # PSMA normal
                combined_pred[fdg_pred == 1] = 3   # FDG tumor
                combined_pred[fdg_pred == 2] = 4   # FDG normal
                
                # Calculate dice for each class
                for cls, name in [(1, 'psma_tumor'), (2, 'psma_normal'), 
                                  (3, 'fdg_tumor'), (4, 'fdg_normal')]:
                    if (labels == cls).any():
                        dice = self.compute_dice(combined_pred == cls, labels == cls)
                        dice_scores[name].append(dice)
        
        # Average losses and dice scores
        for key in val_losses:
            val_losses[key] /= len(self.val_loader)
        
        avg_dice_scores = {}
        for key, scores in dice_scores.items():
            if scores:
                # Convert tensors to CPU and numpy before computing mean
                cpu_scores = [score.cpu().item() if torch.is_tensor(score) else score for score in scores]
                avg_dice_scores[key] = np.mean(cpu_scores)
            else:
                avg_dice_scores[key] = 0.0
        
        # Overall dice
        all_dice = [avg_dice_scores[k] for k in ['psma_tumor', 'fdg_tumor'] if avg_dice_scores[k] > 0]
        avg_dice = np.mean(all_dice) if all_dice else 0.0
        
        return val_losses, avg_dice_scores, avg_dice
    
    def sliding_window_inference(self, psma, fdg, patch_size, overlap=0.5):
        """Sliding window inference for large volumes"""
        B, C, D, H, W = psma.shape
        pd, ph, pw = patch_size
        
        # Calculate strides
        stride_d = int(pd * (1 - overlap))
        stride_h = int(ph * (1 - overlap))
        stride_w = int(pw * (1 - overlap))
        
        # Initialize output
        psma_output = torch.zeros((B, 3, D, H, W), device=self.device)
        fdg_output = torch.zeros((B, 3, D, H, W), device=self.device)
        count_map = torch.zeros((B, 1, D, H, W), device=self.device)
        
        # Gaussian importance weighting
        gaussian = self._get_gaussian_weight(patch_size).to(self.device)
        
        for d in range(0, max(1, D - pd + 1), stride_d):
            for h in range(0, max(1, H - ph + 1), stride_h):
                for w in range(0, max(1, W - pw + 1), stride_w):
                    # Extract patch
                    d_end = min(d + pd, D)
                    h_end = min(h + ph, H)
                    w_end = min(w + pw, W)
                    
                    psma_patch = psma[:, :, d:d_end, h:h_end, w:w_end]
                    fdg_patch = fdg[:, :, d:d_end, h:h_end, w:w_end]
                    
                    # Forward pass
                    with autocast(enabled=self.args.amp):
                        model_out = self.model(psma_patch, fdg_patch)
                        if isinstance(model_out, tuple) and len(model_out) == 3:
                            psma_logits, fdg_logits, _ = model_out
                        else:
                            psma_logits, fdg_logits = model_out
                    
                    # Get actual patch size (may be smaller at boundaries)
                    actual_d = d_end - d
                    actual_h = h_end - h
                    actual_w = w_end - w
                    
                    # Adjust gaussian weight if needed
                    weight = gaussian[:, :, :actual_d, :actual_h, :actual_w]
                    
                    # Accumulate predictions
                    psma_output[:, :, d:d_end, h:h_end, w:w_end] += psma_logits * weight
                    fdg_output[:, :, d:d_end, h:h_end, w:w_end] += fdg_logits * weight
                    count_map[:, :, d:d_end, h:h_end, w:w_end] += weight
        
        # Normalize by count
        psma_output = psma_output / count_map.clamp(min=1e-6)
        fdg_output = fdg_output / count_map.clamp(min=1e-6)
        
        return psma_output, fdg_output
    
    def _get_gaussian_weight(self, patch_size):
        """Create 3D Gaussian importance weight"""
        def gaussian_1d(size):
            sigma = size / 6.0
            x = np.arange(size)
            x = x - size // 2
            g = np.exp(-(x ** 2) / (2 * sigma ** 2))
            return g / g.max()
        
        d_gauss = gaussian_1d(patch_size[0])
        h_gauss = gaussian_1d(patch_size[1])
        w_gauss = gaussian_1d(patch_size[2])
        
        weight = np.outer(np.outer(d_gauss, h_gauss).flatten(), w_gauss).reshape(patch_size)
        weight = torch.from_numpy(weight).float().unsqueeze(0).unsqueeze(0)
        
        return weight
    
    def compute_dice(self, pred, target):
        """Compute Dice coefficient"""
        pred = pred.flatten()
        target = target.flatten()
        intersection = (pred * target).sum()
        return (2. * intersection) / (pred.sum() + target.sum() + 1e-6)
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_dice': self.best_val_dice,
            'args': self.args
        }
        
        # Save latest
        torch.save(checkpoint, os.path.join(self.output_dir, 'checkpoint_latest.pth'))
        
        # Save best
        if is_best:
            torch.save(checkpoint, os.path.join(self.output_dir, 'checkpoint_best.pth'))
        
        # Save periodic
        if self.epoch % self.args.save_every == 0:
            torch.save(checkpoint, os.path.join(self.output_dir, f'checkpoint_epoch_{self.epoch}.pth'))
    
    def train(self):
        """Main training loop"""
        print(f"Starting training on {self.device}")
        print(f"Output directory: {self.output_dir}")
        
        for self.epoch in range(1, self.args.epochs + 1):
            # Training
            train_losses = self.train_epoch()
            
            # Validation
            if self.epoch % self.args.validate_every == 0:
                val_losses, dice_scores, avg_dice = self.validate()
                
                # Logging
                print(f"\nEpoch {self.epoch}/{self.args.epochs}")
                print(f"Train Loss: {train_losses['total']:.4f} "
                      f"(PSMA: {train_losses['psma']:.4f}, FDG: {train_losses['fdg']:.4f})")
                print(f"Val Loss: {val_losses['total']:.4f} "
                      f"(PSMA: {val_losses['psma']:.4f}, FDG: {val_losses['fdg']:.4f})")
                print(f"Dice Scores: PSMA Tumor: {dice_scores.get('psma_tumor', 0):.4f}, "
                      f"FDG Tumor: {dice_scores.get('fdg_tumor', 0):.4f}")
                print(f"Average Dice: {avg_dice:.4f}")
                
                # Tensorboard logging
                self.writer.add_scalars('Loss/train', train_losses, self.epoch)
                self.writer.add_scalars('Loss/val', val_losses, self.epoch)
                self.writer.add_scalars('Dice/val', dice_scores, self.epoch)
                self.writer.add_scalar('Dice/average', avg_dice, self.epoch)
                
                # Save best model
                if avg_dice > self.best_val_dice:
                    self.best_val_dice = avg_dice
                    self.save_checkpoint(is_best=True)
                    self.patience_counter = 0
                    print(f"New best model saved with dice: {avg_dice:.4f}")
                else:
                    self.patience_counter += 1
                
                # Early stopping
                if self.patience_counter >= self.args.patience:
                    print(f"Early stopping triggered after {self.epoch} epochs")
                    break
            
            # Save periodic checkpoint
            self.save_checkpoint()
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Log learning rate
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], self.epoch)
        
        print(f"Training completed. Best validation dice: {self.best_val_dice:.4f}")
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train Dual-Encoder Cross-Attention U-Net')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to paired dataset directory')
    parser.add_argument('--output_dir', type=str, default='experiments',
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--fold', type=int, default=0,
                        help='Fold number for cross-validation')
    parser.add_argument('--splits_file', type=str, default=None,
                        help='Path to splits JSON file')
    
    # Model parameters
    parser.add_argument('--base_channels', type=int, default=32,
                        help='Base number of channels in U-Net')
    parser.add_argument('--cross_attention', action='store_true', default=True,
                        help='Enable cross-attention (default: True)')
    parser.add_argument('--no_cross_attention', dest='cross_attention', action='store_false',
                        help='Disable cross-attention (ablation)')
    parser.add_argument('--disagreement_attention', action='store_true', default=False,
                        help='Use disagreement-aware cross-attention')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads for disagreement attention')
    parser.add_argument('--cross_attn_warmup', type=int, default=0,
                        help='Number of epochs for cross-attention warmup (0=no warmup)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size for training')
    parser.add_argument('--patch_size', type=int, nargs=3, default=[112, 192, 112],
                        help='Patch size for training (default: 112x192x112 to match baselines)')
    parser.add_argument('--samples_per_epoch', type=int, default=250,
                        help='Number of samples per epoch')
    parser.add_argument('--foreground_ratio', type=float, default=0.33,
                        help='Ratio of foreground-focused patches')
    
    # Optimization (defaults matched to nnU-Net baselines)
    parser.add_argument('--optimizer', type=str, default='sgd',
                        choices=['adam', 'sgd'], help='Optimizer (default: sgd to match baselines)')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Initial learning rate (default: 0.01 to match baselines)')
    parser.add_argument('--weight_decay', type=float, default=3e-5,
                        help='Weight decay')
    parser.add_argument('--scheduler', type=str, default='poly',
                        choices=['poly', 'cosine', 'none'], help='LR scheduler')
    parser.add_argument('--gradient_clip', type=float, default=0,
                        help='Gradient clipping value (0=disabled, matching baselines)')

    # Loss configuration
    parser.add_argument('--loss_mode', type=str, default='dice',
                        choices=['dice', 'dice_ce'],
                        help='Loss function: dice (matched baseline) or dice_ce (original DECA)')
    parser.add_argument('--psma_loss_weight', type=float, default=0.5,
                        help='Weight for PSMA loss (default 0.5 = equal)')
    parser.add_argument('--fdg_loss_weight', type=float, default=0.5,
                        help='Weight for FDG loss (default 0.5 = equal)')
    
    # Validation
    parser.add_argument('--validate_every', type=int, default=5,
                        help='Validate every N epochs')
    parser.add_argument('--patience', type=int, default=30,
                        help='Early stopping patience')
    parser.add_argument('--save_every', type=int, default=20,
                        help='Save checkpoint every N epochs')
    
    # System
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--amp', action='store_true', default=False,
                        help='Use automatic mixed precision (off by default to match baselines)')
    parser.add_argument('--multi_gpu', action='store_true', default=False,
                        help='Use multiple GPUs if available')
    parser.add_argument('--cache_data', action='store_true', default=False,
                        help='Cache data in memory')
    
    args = parser.parse_args()
    
    # Create trainer and start training
    trainer = Trainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
