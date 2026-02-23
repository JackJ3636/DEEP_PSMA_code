#!/usr/bin/env python3
"""
Inference script for Dual-Encoder Cross-Attention U-Net
Performs sliding window inference with optional test-time augmentation
"""

import os
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
import nibabel as nib
from tqdm import tqdm
from typing import Tuple, Optional

from dual_encoder_model import DualEncoderCrossAttentionUNet
from dataset import PairedPETCTDataset


class InferenceEngine:
    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.args = checkpoint['args']
        
        # Initialize model with same config used during training
        self.model = DualEncoderCrossAttentionUNet(
            base_channels=self.args.base_channels,
            cross_attention=getattr(self.args, 'cross_attention', True),
            disagreement_attention=getattr(self.args, 'disagreement_attention', False),
            num_heads=getattr(self.args, 'num_heads', 4),
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Loaded model from epoch {checkpoint['epoch']} with best dice: {checkpoint['best_val_dice']:.4f}")
    
    def sliding_window_inference(self, 
                                psma: torch.Tensor, 
                                fdg: torch.Tensor,
                                patch_size: Tuple[int, int, int] = (96, 160, 96),
                                overlap: float = 0.5,
                                tta: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform sliding window inference on full volumes
        
        Args:
            psma: (1, 2, D, H, W) tensor
            fdg: (1, 2, D, H, W) tensor
            patch_size: Size of patches for inference
            overlap: Overlap ratio between patches
            tta: Enable test-time augmentation
        
        Returns:
            Combined predictions (1, D, H, W) with classes 0-4
        """
        
        B, C, D, H, W = psma.shape
        pd, ph, pw = patch_size
        
        # Calculate strides
        stride_d = max(1, int(pd * (1 - overlap)))
        stride_h = max(1, int(ph * (1 - overlap)))
        stride_w = max(1, int(pw * (1 - overlap)))
        
        # Pad if volume is smaller than patch
        pad_d = max(0, pd - D)
        pad_h = max(0, ph - H)
        pad_w = max(0, pw - W)
        
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            pad = (0, pad_w, 0, pad_h, 0, pad_d)
            psma = F.pad(psma, pad, mode='constant', value=0)
            fdg = F.pad(fdg, pad, mode='constant', value=0)
            _, _, D, H, W = psma.shape
        
        # Initialize output tensors
        psma_output = torch.zeros((B, 3, D, H, W), device=self.device)
        fdg_output = torch.zeros((B, 3, D, H, W), device=self.device)
        count_map = torch.zeros((B, 1, D, H, W), device=self.device)
        
        # Gaussian importance weighting
        gaussian = self._get_gaussian_weight(patch_size).to(self.device)
        
        # Test-time augmentation flips
        flips = [(False, False, False)]  # Original
        if tta:
            flips.extend([
                (True, False, False),  # Flip D
                (False, True, False),  # Flip H
                (False, False, True),  # Flip W
            ])
        
        with torch.no_grad():
            for flip_d, flip_h, flip_w in flips:
                # Apply flips
                psma_flip = psma.clone()
                fdg_flip = fdg.clone()
                
                if flip_d:
                    psma_flip = torch.flip(psma_flip, dims=[2])
                    fdg_flip = torch.flip(fdg_flip, dims=[2])
                if flip_h:
                    psma_flip = torch.flip(psma_flip, dims=[3])
                    fdg_flip = torch.flip(fdg_flip, dims=[3])
                if flip_w:
                    psma_flip = torch.flip(psma_flip, dims=[4])
                    fdg_flip = torch.flip(fdg_flip, dims=[4])
                
                # Sliding window
                for d in range(0, D - pd + 1, stride_d):
                    for h in range(0, H - ph + 1, stride_h):
                        for w in range(0, W - pw + 1, stride_w):
                            # Extract patch
                            psma_patch = psma_flip[:, :, d:d+pd, h:h+ph, w:w+pw]
                            fdg_patch = fdg_flip[:, :, d:d+pd, h:h+ph, w:w+pw]
                            
                            # Forward pass
                            with autocast(enabled=True):
                                model_out = self.model(psma_patch, fdg_patch)
                                if isinstance(model_out, tuple) and len(model_out) == 3:
                                    psma_logits, fdg_logits, _ = model_out
                                else:
                                    psma_logits, fdg_logits = model_out
                            
                            # Apply inverse flips to predictions
                            if flip_w:
                                psma_logits = torch.flip(psma_logits, dims=[4])
                                fdg_logits = torch.flip(fdg_logits, dims=[4])
                            if flip_h:
                                psma_logits = torch.flip(psma_logits, dims=[3])
                                fdg_logits = torch.flip(fdg_logits, dims=[3])
                            if flip_d:
                                psma_logits = torch.flip(psma_logits, dims=[2])
                                fdg_logits = torch.flip(fdg_logits, dims=[2])
                            
                            # Accumulate predictions with Gaussian weighting
                            psma_output[:, :, d:d+pd, h:h+ph, w:w+pw] += psma_logits * gaussian
                            fdg_output[:, :, d:d+pd, h:h+ph, w:w+pw] += fdg_logits * gaussian
                            count_map[:, :, d:d+pd, h:h+ph, w:w+pw] += gaussian
        
        # Normalize accumulated predictions
        psma_output = psma_output / count_map.clamp(min=1e-6)
        fdg_output = fdg_output / count_map.clamp(min=1e-6)
        
        # Remove padding
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            psma_output = psma_output[:, :, :D-pad_d, :H-pad_h, :W-pad_w]
            fdg_output = fdg_output[:, :, :D-pad_d, :H-pad_h, :W-pad_w]
        
        # Get predictions
        psma_pred = psma_output.argmax(dim=1)  # (B, D, H, W)
        fdg_pred = fdg_output.argmax(dim=1)
        
        # Combine into 5-class prediction
        combined = torch.zeros_like(psma_pred)
        combined[psma_pred == 1] = 1  # PSMA tumor
        combined[psma_pred == 2] = 2  # PSMA normal
        combined[fdg_pred == 1] = 3   # FDG tumor
        combined[fdg_pred == 2] = 4   # FDG normal
        
        return combined
    
    def _get_gaussian_weight(self, patch_size: Tuple[int, int, int]) -> torch.Tensor:
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
    
    def refine_predictions_with_thresholds(self,
                                          predictions: torch.Tensor,
                                          psma_pet: torch.Tensor,
                                          fdg_pet: torch.Tensor,
                                          psma_threshold: float = 1.0,
                                          fdg_threshold: float = 1.0) -> torch.Tensor:
        """
        Refine predictions using PET thresholds
        Remove predictions where PET signal is below threshold
        
        Args:
            predictions: (B, D, H, W) tensor with classes 0-4
            psma_pet: (B, D, H, W) normalized PET tensor
            fdg_pet: (B, D, H, W) normalized PET tensor
            psma_threshold: Threshold for PSMA (after normalization, typically 1.0)
            fdg_threshold: Threshold for FDG (after normalization, typically 1.0)
        """
        refined = predictions.clone()
        
        # Remove PSMA predictions where PET < threshold
        psma_mask = psma_pet < psma_threshold
        refined[(predictions == 1) & psma_mask] = 0  # PSMA tumor -> background
        refined[(predictions == 2) & psma_mask] = 0  # PSMA normal -> background
        
        # Remove FDG predictions where PET < threshold
        fdg_mask = fdg_pet < fdg_threshold
        refined[(predictions == 3) & fdg_mask] = 0  # FDG tumor -> background
        refined[(predictions == 4) & fdg_mask] = 0  # FDG normal -> background
        
        return refined
    
    def predict_case(self,
                    case_data: dict,
                    patch_size: Tuple[int, int, int] = (96, 160, 96),
                    overlap: float = 0.5,
                    tta: bool = False,
                    refine_with_threshold: bool = True) -> np.ndarray:
        """
        Predict segmentation for a single case
        
        Args:
            case_data: Dictionary with 'psma', 'fdg' tensors
            patch_size: Patch size for sliding window
            overlap: Overlap ratio
            tta: Enable test-time augmentation
            refine_with_threshold: Apply threshold-based refinement
        
        Returns:
            Numpy array with predictions
        """
        psma = case_data['psma'].unsqueeze(0).to(self.device)
        fdg = case_data['fdg'].unsqueeze(0).to(self.device)
        
        # Perform inference
        predictions = self.sliding_window_inference(
            psma, fdg, 
            patch_size=patch_size,
            overlap=overlap,
            tta=tta
        )
        
        # Optional threshold refinement
        if refine_with_threshold:
            psma_pet = psma[:, 0, :, :, :]  # First channel is PET
            fdg_pet = fdg[:, 0, :, :, :]
            predictions = self.refine_predictions_with_thresholds(
                predictions, psma_pet, fdg_pet
            )
        
        return predictions.squeeze(0).cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description='Inference for Dual-Encoder Model')
    
    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to paired dataset directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for predictions')
    
    # Optional arguments
    parser.add_argument('--fold', type=int, default=None,
                        help='Fold number for validation set')
    parser.add_argument('--splits_file', type=str, default=None,
                        help='Path to splits JSON file')
    parser.add_argument('--patch_size', type=int, nargs=3, default=[96, 160, 96],
                        help='Patch size for sliding window')
    parser.add_argument('--overlap', type=float, default=0.5,
                        help='Overlap ratio for sliding window')
    parser.add_argument('--tta', action='store_true',
                        help='Enable test-time augmentation')
    parser.add_argument('--no_threshold_refinement', action='store_true',
                        help='Disable threshold-based refinement')
    parser.add_argument('--save_probabilities', action='store_true',
                        help='Save probability maps in addition to predictions')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize inference engine
    engine = InferenceEngine(args.checkpoint)
    
    # Load dataset
    dataset = PairedPETCTDataset(
        root_dir=args.data_dir,
        split='val' if args.fold is not None else 'test',
        fold=args.fold,
        splits_file=args.splits_file,
        patch_size=None,  # Full volume
        augment=False,
        cache_data=True
    )
    
    print(f"Processing {len(dataset)} cases...")
    
    # Process each case
    for idx in tqdm(range(len(dataset)), desc='Inference'):
        data = dataset[idx]
        case_id = data['case_id']
        
        # Predict
        predictions = engine.predict_case(
            data,
            patch_size=tuple(args.patch_size),
            overlap=args.overlap,
            tta=args.tta,
            refine_with_threshold=not args.no_threshold_refinement
        )
        
        # Load original image for affine matrix
        img_path = os.path.join(args.data_dir, 'imagesTr', f'{case_id}_0000.nii.gz')
        ref_img = nib.load(img_path)
        
        # Save predictions
        pred_img = nib.Nifti1Image(predictions.astype(np.uint8), ref_img.affine)
        nib.save(pred_img, os.path.join(args.output_dir, f'{case_id}.nii.gz'))
    
    print(f"Inference complete. Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()
