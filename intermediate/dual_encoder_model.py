#!/usr/bin/env python3
"""
Dual-Encoder Cross-Attention U-Net for DEEP-PSMA Challenge
Handles both PSMA and FDG PET/CT with intermediate fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ConvBlock(nn.Module):
    """Double convolution block with InstanceNorm and ReLU"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class DownBlock(nn.Module):
    """Downsampling block: MaxPool + ConvBlock"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool3d(2),
            ConvBlock(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.down(x)


class UpBlock(nn.Module):
    """Upsampling block with skip connection"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)
    
    def forward(self, x, skip):
        x = self.up(x)
        # Handle size mismatch
        diff_d = skip.size(2) - x.size(2)
        diff_h = skip.size(3) - x.size(3)
        diff_w = skip.size(4) - x.size(4)
        
        if diff_d != 0 or diff_h != 0 or diff_w != 0:
            x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                         diff_h // 2, diff_h - diff_h // 2,
                         diff_d // 2, diff_d - diff_d // 2])
        
        return self.conv(torch.cat([skip, x], dim=1))


class CrossAttentionBlock(nn.Module):
    """Cross-attention mechanism for feature fusion"""
    def __init__(self, channels: int):
        super().__init__()
        reduced_channels = max(channels // 8, 16)
        self.query = nn.Conv3d(channels, reduced_channels, kernel_size=1)
        self.key = nn.Conv3d(channels, reduced_channels, kernel_size=1)
        self.value = nn.Conv3d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x_q, x_kv):
        B, C, D, H, W = x_q.shape

        # Generate query, key, value
        q = self.query(x_q).view(B, -1, D * H * W)  # (B, C', DHW)
        k = self.key(x_kv).view(B, -1, D * H * W)   # (B, C', DHW)
        v = self.value(x_kv).view(B, -1, D * H * W) # (B, C, DHW)

        # Compute attention
        attn = torch.bmm(q.transpose(1, 2), k)  # (B, DHW, DHW)
        attn = F.softmax(attn / (q.shape[1] ** 0.5), dim=-1)

        # Apply attention
        out = torch.bmm(v, attn.transpose(1, 2))  # (B, C, DHW)
        out = out.view(B, C, D, H, W)

        # Residual connection with learnable weight
        return x_q + self.gamma * out


class DisagreementAwareCrossAttention(nn.Module):
    """Cross-attention modulated by learned tracer disagreement.

    Drop-in replacement for CrossAttentionBlock. Learns a per-voxel
    disagreement map and suppresses attention where tracers disagree.
    """
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        # Q/K/V projections
        self.q_proj = nn.Conv3d(channels, channels, 1)
        self.k_proj = nn.Conv3d(channels, channels, 1)
        self.v_proj = nn.Conv3d(channels, channels, 1)
        self.out_proj = nn.Conv3d(channels, channels, 1)

        # Lightweight disagreement head
        self.disagree_net = nn.Sequential(
            nn.Conv3d(channels * 2, channels // 2, 3, padding=1),
            nn.InstanceNorm3d(channels // 2),
            nn.LeakyReLU(0.01),
            nn.Conv3d(channels // 2, 1, 1),
            nn.Sigmoid()
        )

        # Learnable suppression strength (init 0.5)
        self.alpha = nn.Parameter(torch.tensor(0.5))

        # Residual weight (like gamma in the original block)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, f_query, f_key_value):
        B, C, D, H, W = f_query.shape
        N = D * H * W

        # Compute per-voxel disagreement ∈ [0, 1]
        disagree = self.disagree_net(
            torch.cat([f_query, f_key_value], dim=1)
        )  # (B, 1, D, H, W)

        # Multi-head Q/K/V
        Q = self.q_proj(f_query).reshape(B, self.num_heads, self.head_dim, N)
        K = self.k_proj(f_key_value).reshape(B, self.num_heads, self.head_dim, N)
        V = self.v_proj(f_key_value).reshape(B, self.num_heads, self.head_dim, N)

        # Attention scores
        attn = torch.einsum('bhdn,bhdm->bhnm', Q, K) / (self.head_dim ** 0.5)
        attn = torch.softmax(attn, dim=-1)

        # Suppress attention at disagreement sites
        suppression = (1.0 - self.alpha * disagree).reshape(B, 1, 1, N)
        attn = attn * suppression

        # Weighted values
        out = torch.einsum('bhnm,bhdm->bhdn', attn, V)
        out = out.reshape(B, C, D, H, W)
        out = self.out_proj(out)

        # Residual connection
        return f_query + self.gamma * out, disagree


class DualEncoderCrossAttentionUNet(nn.Module):
    """
    Dual-encoder U-Net with cross-attention for PSMA/FDG segmentation.
    Each modality has its own encoder/decoder with cross-attention fusion.

    Args:
        base_channels: Base number of feature channels.
        cross_attention: Enable cross-attention between encoders.
        disagreement_attention: Use DisagreementAwareCrossAttention instead
            of standard CrossAttentionBlock (requires cross_attention=True).
        num_heads: Number of attention heads for disagreement attention.
    """

    def __init__(self, base_channels: int = 32, cross_attention: bool = True,
                 disagreement_attention: bool = False, num_heads: int = 4):
        super().__init__()
        c = base_channels
        self.use_cross_attention = cross_attention

        # PSMA Encoder (takes 2 channels: PET + CT)
        self.psma_enc1 = ConvBlock(2, c)
        self.psma_enc2 = DownBlock(c, c * 2)
        self.psma_enc3 = DownBlock(c * 2, c * 4)
        self.psma_enc4 = DownBlock(c * 4, c * 8)
        self.psma_bottleneck = ConvBlock(c * 8, c * 16)

        # FDG Encoder (takes 2 channels: PET + CT)
        self.fdg_enc1 = ConvBlock(2, c)
        self.fdg_enc2 = DownBlock(c, c * 2)
        self.fdg_enc3 = DownBlock(c * 2, c * 4)
        self.fdg_enc4 = DownBlock(c * 4, c * 8)
        self.fdg_bottleneck = ConvBlock(c * 8, c * 16)

        # Cross-attention at multiple scales
        if cross_attention:
            if disagreement_attention:
                self.cross_attn_3 = DisagreementAwareCrossAttention(c * 4, num_heads)
                self.cross_attn_4 = DisagreementAwareCrossAttention(c * 8, num_heads)
                self.cross_attn_bottleneck = DisagreementAwareCrossAttention(c * 16, num_heads)
            else:
                self.cross_attn_3 = CrossAttentionBlock(c * 4)
                self.cross_attn_4 = CrossAttentionBlock(c * 8)
                self.cross_attn_bottleneck = CrossAttentionBlock(c * 16)
        self.disagreement_attention = disagreement_attention

        # PSMA Decoder
        self.psma_up4 = UpBlock(c * 16, c * 8)
        self.psma_up3 = UpBlock(c * 8, c * 4)
        self.psma_up2 = UpBlock(c * 4, c * 2)
        self.psma_up1 = UpBlock(c * 2, c)

        # FDG Decoder
        self.fdg_up4 = UpBlock(c * 16, c * 8)
        self.fdg_up3 = UpBlock(c * 8, c * 4)
        self.fdg_up2 = UpBlock(c * 4, c * 2)
        self.fdg_up1 = UpBlock(c * 2, c)

        # Output heads (3 classes each: bg, tumor, normal)
        self.psma_out = nn.Conv3d(c, 3, kernel_size=1)
        self.fdg_out = nn.Conv3d(c, 3, kernel_size=1)

    def _apply_cross_attention(self, attn_module, x_q, x_kv):
        """Apply cross-attention, handling both standard and disagreement variants."""
        result = attn_module(x_q, x_kv)
        if isinstance(result, tuple):
            # DisagreementAwareCrossAttention returns (output, disagree_map)
            return result[0], result[1]
        # Standard CrossAttentionBlock returns just the output
        return result, None

    def forward(self, psma_input: torch.Tensor, fdg_input: torch.Tensor,
                cross_attention_weight: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            psma_input: (B, 2, D, H, W) - PSMA PET and CT
            fdg_input: (B, 2, D, H, W) - FDG PET and CT
            cross_attention_weight: Weight for cross-attention (ignored when
                cross_attention=False; kept for API compatibility)

        Returns:
            psma_logits: (B, 3, D, H, W) - bg, tumor, normal
            fdg_logits: (B, 3, D, H, W) - bg, tumor, normal
        """
        disagree_maps = {}

        # PSMA Encoding
        p1 = self.psma_enc1(psma_input)
        p2 = self.psma_enc2(p1)
        p3 = self.psma_enc3(p2)
        p4 = self.psma_enc4(p3)
        p_bottleneck = self.psma_bottleneck(p4)

        # FDG Encoding
        f1 = self.fdg_enc1(fdg_input)
        f2 = self.fdg_enc2(f1)
        f3 = self.fdg_enc3(f2)
        f4 = self.fdg_enc4(f3)
        f_bottleneck = self.fdg_bottleneck(f4)

        # Cross-attention fusion
        if self.use_cross_attention and cross_attention_weight > 0:
            # Level 3 cross-attention
            p3_fused, d3_p = self._apply_cross_attention(self.cross_attn_3, p3, f3)
            f3_fused, d3_f = self._apply_cross_attention(self.cross_attn_3, f3, p3)
            p3 = p3 + cross_attention_weight * (p3_fused - p3)
            f3 = f3 + cross_attention_weight * (f3_fused - f3)

            # Level 4 cross-attention
            p4_fused, d4_p = self._apply_cross_attention(self.cross_attn_4, p4, f4)
            f4_fused, d4_f = self._apply_cross_attention(self.cross_attn_4, f4, p4)
            p4 = p4 + cross_attention_weight * (p4_fused - p4)
            f4 = f4 + cross_attention_weight * (f4_fused - f4)

            # Bottleneck cross-attention
            p_bot_fused, db_p = self._apply_cross_attention(self.cross_attn_bottleneck, p_bottleneck, f_bottleneck)
            f_bot_fused, db_f = self._apply_cross_attention(self.cross_attn_bottleneck, f_bottleneck, p_bottleneck)
            p_bottleneck = p_bottleneck + cross_attention_weight * (p_bot_fused - p_bottleneck)
            f_bottleneck = f_bottleneck + cross_attention_weight * (f_bot_fused - f_bottleneck)

            if self.disagreement_attention:
                disagree_maps = {
                    'level3_psma': d3_p, 'level3_fdg': d3_f,
                    'level4_psma': d4_p, 'level4_fdg': d4_f,
                    'bottleneck_psma': db_p, 'bottleneck_fdg': db_f,
                }

        # PSMA Decoding
        p_up4 = self.psma_up4(p_bottleneck, p4)
        p_up3 = self.psma_up3(p_up4, p3)
        p_up2 = self.psma_up2(p_up3, p2)
        p_up1 = self.psma_up1(p_up2, p1)
        psma_logits = self.psma_out(p_up1)

        # FDG Decoding
        f_up4 = self.fdg_up4(f_bottleneck, f4)
        f_up3 = self.fdg_up3(f_up4, f3)
        f_up2 = self.fdg_up2(f_up3, f2)
        f_up1 = self.fdg_up1(f_up2, f1)
        fdg_logits = self.fdg_out(f_up1)

        if self.disagreement_attention and disagree_maps:
            return psma_logits, fdg_logits, disagree_maps

        return psma_logits, fdg_logits


def combined_loss(psma_logits: torch.Tensor, fdg_logits: torch.Tensor,
                  labels: torch.Tensor, class_weights: Optional[torch.Tensor] = None,
                  loss_mode: str = 'dice_ce',
                  psma_weight: float = 0.5, fdg_weight: float = 0.5) -> dict:
    """
    Compute combined loss for dual-head model.

    Args:
        psma_logits: (B, 3, D, H, W) - PSMA predictions
        fdg_logits: (B, 3, D, H, W) - FDG predictions
        labels: (B, D, H, W) - Ground truth with classes 0-4
        class_weights: Optional weights for CE loss
        loss_mode: 'dice' for Dice-only (matched baseline) or 'dice_ce'
            for 0.5*Dice + 0.5*CE (original DECA-UNet)
        psma_weight: Weight for PSMA loss in total (default 0.5 = equal)
        fdg_weight: Weight for FDG loss in total (default 0.5 = equal)

    Returns:
        Dictionary with individual and total losses
    """

    # Map labels to PSMA classes (0=bg, 1=tumor, 2=normal)
    psma_labels = labels.clone()
    psma_labels[labels == 3] = 0  # FDG tumor -> bg for PSMA
    psma_labels[labels == 4] = 0  # FDG normal -> bg for PSMA

    # Map labels to FDG classes (0=bg, 1=tumor, 2=normal)
    fdg_labels = labels.clone()
    fdg_labels[labels == 1] = 0  # PSMA tumor -> bg for FDG
    fdg_labels[labels == 2] = 0  # PSMA normal -> bg for FDG
    fdg_labels[labels == 3] = 1  # FDG tumor
    fdg_labels[labels == 4] = 2  # FDG normal

    # Dice losses
    dice_psma = soft_dice_loss(psma_logits, psma_labels)
    dice_fdg = soft_dice_loss(fdg_logits, fdg_labels)

    if loss_mode == 'dice':
        # Dice-only (matched to nnU-Net baselines)
        psma_loss = dice_psma
        fdg_loss = dice_fdg
        result = {
            'total': psma_weight * psma_loss + fdg_weight * fdg_loss,
            'psma': psma_loss,
            'fdg': fdg_loss,
            'dice_psma': dice_psma,
            'dice_fdg': dice_fdg,
        }
    else:
        # 0.5*Dice + 0.5*CE (original DECA-UNet)
        if class_weights is None:
            class_weights = torch.tensor([0.1, 2.0, 0.5], device=psma_logits.device)
        ce_psma = F.cross_entropy(psma_logits, psma_labels, weight=class_weights)
        ce_fdg = F.cross_entropy(fdg_logits, fdg_labels, weight=class_weights)
        psma_loss = 0.5 * ce_psma + 0.5 * dice_psma
        fdg_loss = 0.5 * ce_fdg + 0.5 * dice_fdg
        result = {
            'total': psma_weight * psma_loss + fdg_weight * fdg_loss,
            'psma': psma_loss,
            'fdg': fdg_loss,
            'ce_psma': ce_psma,
            'ce_fdg': ce_fdg,
            'dice_psma': dice_psma,
            'dice_fdg': dice_fdg,
        }

    return result


def soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """
    Soft Dice loss for multi-class segmentation
    """
    num_classes = logits.shape[1]
    probs = F.softmax(logits, dim=1)
    targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
    
    # Compute dice per class
    dims = (0, 2, 3, 4)  # Batch and spatial dims
    intersection = torch.sum(probs * targets_one_hot, dim=dims)
    cardinality = torch.sum(probs + targets_one_hot, dim=dims)
    dice_score = (2. * intersection + epsilon) / (cardinality + epsilon)
    
    # Exclude background from loss
    return 1.0 - dice_score[1:].mean()
