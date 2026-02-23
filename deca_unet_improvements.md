# DECA-UNet Improvements & Disagreement-Aware Attention

## Overview

Two focused changes for the revised paper:

1. **Re-run DECA-UNet with settings matched to the existing baselines** — reuse all previous baseline results.
2. **Add disagreement-aware attention** — a minimal modification to the cross-attention that addresses the paper's core question.

---

## 1. Match DECA-UNet to Baseline Settings

### 1.1 Problem

The reviewers can't tell whether DECA-UNet's PSMA improvement (0.928 → 0.952) comes from fusion or from the different training setup. Fix this by matching DECA-UNet to the baselines, not the other way round.

### 1.2 Settings to Change in DECA-UNet

| Setting | Baseline (keep as-is) | Current DECA-UNet | Revised DECA-UNet |
|---------|----------------------|-------------------|-------------------|
| Optimiser | SGD | AdamW | **SGD** |
| Learning rate | 0.01 | 0.001 | **0.01** |
| LR schedule | Poly decay | Poly decay | Poly decay ✓ |
| Patch size | 112×192×112 | 96×160×96 | **112×192×112 if VRAM allows, else keep and note** |
| Batch size | 2 | 1 | **2 if VRAM allows, else keep and note** |
| Gradient clipping | No | max_norm=12 | **Remove** |
| Mixed precision | No | Yes | **Remove** |
| Attention warmup | N/A | 10 epochs | **Remove** |
| Loss | Dice | 0.5×Dice + 0.5×CE, weighted 5/12 + 7/12 | **Dice only, equal 50/50** |

**On VRAM:** If dual-encoder can't fit batch=2 or patch=112×192×112, state this explicitly in the paper as a legitimate architectural cost of fusion, not a confound.

### 1.3 Models to Train (4 new runs × 5 folds = 20 total)

| # | Model | Purpose |
|---|-------|---------|
| 1 | DECA-UNet (matched settings) | Fair comparison to existing baselines |
| 2 | DECA-UNet (matched, no cross-attn) | Ablation: is attention helping or just the dual-encoder/decoder? |
| 3 | DECA-UNet + Disagreement-Aware Attn (matched) | New contribution |
| 4 | DECA-UNet + Disagree Attn (matched, 5/12+7/12 loss) | Test whether FDG upweighting helps with new attention |

**All previous results reused as-is:** PSMA Baseline, FDG Baseline, Union Vote, OEOD, OETD.

### 1.4 Reporting Additions

- Single summary table: all models (old + new) × all metrics (Dice, Surface Dice, FP/FN Vol, SUVmean Ratio, Vol Ratio), mean ± std, best bolded.
- Parameter counts for every model.
- Paired Wilcoxon signed-rank p-values for: DECA PSMA vs PSMA baseline, DECA FDG vs FDG baseline.

---

## 2. Disagreement-Aware Attention

### 2.1 The Change

Replace each `CrossAttentionBlock` in DECA-UNet with a version that suppresses attention where the two tracers disagree. Everything else (encoders, decoders, skips) stays identical.

**Standard cross-attention:**
```
A = softmax(QK^T / sqrt(d))
output = A @ V
```

**Disagreement-aware:**
```
D = DisagreementNet(concat(f_PSMA, f_FDG))   # per-voxel [0,1], high = disagreement
suppression = 1 - alpha * D                    # alpha learnable, init 0.5
output = (A * suppression) @ V
```

### 2.2 Implementation

Drop-in replacement for your existing cross-attention:

```python
class DisagreementAwareCrossAttention(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        # Keep existing projections
        self.q_proj = nn.Conv3d(channels, channels, 1)
        self.k_proj = nn.Conv3d(channels, channels, 1)
        self.v_proj = nn.Conv3d(channels, channels, 1)
        self.out_proj = nn.Conv3d(channels, channels, 1)

        # NEW: lightweight disagreement head
        self.disagree_net = nn.Sequential(
            nn.Conv3d(channels * 2, channels // 2, 3, padding=1),
            nn.InstanceNorm3d(channels // 2),
            nn.LeakyReLU(0.01),
            nn.Conv3d(channels // 2, 1, 1),
            nn.Sigmoid()
        )
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, f_query, f_key_value):
        B, C, D, H, W = f_query.shape
        N = D * H * W

        disagree = self.disagree_net(
            torch.cat([f_query, f_key_value], dim=1)
        )  # (B, 1, D, H, W)

        Q = self.q_proj(f_query).reshape(B, self.num_heads, self.head_dim, N)
        K = self.k_proj(f_key_value).reshape(B, self.num_heads, self.head_dim, N)
        V = self.v_proj(f_key_value).reshape(B, self.num_heads, self.head_dim, N)

        attn = torch.einsum('bhdn,bhdm->bhnm', Q, K) / (self.head_dim ** 0.5)
        attn = torch.softmax(attn, dim=-1)

        suppression = (1.0 - self.alpha * disagree).reshape(B, 1, 1, N)
        attn = attn * suppression

        out = torch.einsum('bhnm,bhdm->bhdn', attn, V)
        out = out.reshape(B, C, D, H, W)
        out = self.out_proj(out)

        return out, disagree  # keep disagree map for figures
```

### 2.3 What to Show in the Paper

- **Learned alpha**: report final converged value. High alpha = model actively suppresses cross-attention at disagreement sites.
- **Disagreement maps**: overlay on PET MIPs for 2–3 cases showing correspondence with regions where one tracer is avid and the other is cold.
- **FDG Dice recovery**: main hypothesis is FDG Dice improves because features are no longer corrupted at disagreement sites.

---

## 3. Quick Reference

| Component | Status |
|-----------|--------|
| PSMA Baseline Refined | **Reuse** existing results |
| FDG Baseline Refined | **Reuse** existing results |
| Union Vote | **Reuse** existing results |
| OEOD | **Reuse** existing results |
| OETD | **Reuse** existing results |
| DECA-UNet (matched) | **New** run |
| DECA-UNet (no cross-attn) | **New** ablation |
| DECA-UNet + Disagree Attn | **New** main contribution |
| DECA-UNet + Disagree Attn (FDG upweighted) | **New** variant |
| Evaluation code | **Add** Wilcoxon tests + param counts |
