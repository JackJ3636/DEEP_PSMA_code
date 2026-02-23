#!/usr/bin/env python3
"""
Evaluation script for computing metrics on predictions
Calculates Dice scores, sensitivity, and specificity for each class
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import stats
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional


def compute_dice(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """Compute Dice coefficient"""
    intersection = np.sum(pred * target)
    return (2.0 * intersection + smooth) / (np.sum(pred) + np.sum(target) + smooth)


def compute_sensitivity(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """Compute sensitivity (recall)"""
    true_positive = np.sum(pred * target)
    false_negative = np.sum((1 - pred) * target)
    return (true_positive + smooth) / (true_positive + false_negative + smooth)


def compute_specificity(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """Compute specificity"""
    true_negative = np.sum((1 - pred) * (1 - target))
    false_positive = np.sum(pred * (1 - target))
    return (true_negative + smooth) / (true_negative + false_positive + smooth)


def compute_surface_dice(pred: np.ndarray, target: np.ndarray, tau: float = 2.0,
                         spacing: Tuple[float, ...] = (1.0, 1.0, 1.0)) -> float:
    """Compute Surface Dice at tolerance tau (mm)."""
    pred_border = pred.astype(bool) ^ nib.processing.binary_erosion(pred.astype(bool)) if hasattr(nib, 'processing') else _border(pred)
    target_border = target.astype(bool) ^ _border(target)

    # Use distance transforms
    if target_border.sum() == 0 and pred_border.sum() == 0:
        return 1.0
    if target_border.sum() == 0 or pred_border.sum() == 0:
        return 0.0

    dt_target = distance_transform_edt(~target_border, sampling=spacing)
    dt_pred = distance_transform_edt(~pred_border, sampling=spacing)

    pred_on_target = dt_target[pred_border]
    target_on_pred = dt_pred[target_border]

    overlap_pred = np.sum(pred_on_target <= tau)
    overlap_target = np.sum(target_on_pred <= tau)

    return (overlap_pred + overlap_target) / (pred_border.sum() + target_border.sum() + 1e-8)


def _border(mask: np.ndarray) -> np.ndarray:
    """Extract border voxels via erosion."""
    from scipy.ndimage import binary_erosion as _erode
    eroded = _erode(mask.astype(bool))
    return mask.astype(bool) ^ eroded


def compute_hausdorff_95(pred: np.ndarray, target: np.ndarray,
                         spacing: Tuple[float, ...] = (1.0, 1.0, 1.0)) -> float:
    """Compute 95th percentile Hausdorff distance (mm)."""
    if pred.sum() == 0 and target.sum() == 0:
        return 0.0
    if pred.sum() == 0 or target.sum() == 0:
        return float('inf')

    pred_border = _border(pred)
    target_border = _border(target)

    dt_target = distance_transform_edt(~target_border, sampling=spacing)
    dt_pred = distance_transform_edt(~pred_border, sampling=spacing)

    d_pred_to_target = dt_target[pred_border]
    d_target_to_pred = dt_pred[target_border]

    return max(np.percentile(d_pred_to_target, 95), np.percentile(d_target_to_pred, 95))


def compute_fp_fn_volume(pred: np.ndarray, target: np.ndarray,
                         voxel_volume_ml: float = 0.001) -> Tuple[float, float]:
    """Compute false positive and false negative volumes in mL."""
    fp = np.sum(pred.astype(bool) & ~target.astype(bool)) * voxel_volume_ml
    fn = np.sum(~pred.astype(bool) & target.astype(bool)) * voxel_volume_ml
    return fp, fn


def wilcoxon_test(scores_a: List[float], scores_b: List[float]) -> Tuple[float, float]:
    """Paired Wilcoxon signed-rank test. Returns (statistic, p-value)."""
    a = np.array(scores_a)
    b = np.array(scores_b)
    diff = a - b
    # Remove zero differences (Wilcoxon cannot handle them)
    nonzero = diff != 0
    if nonzero.sum() < 2:
        return 0.0, 1.0
    stat, p = stats.wilcoxon(a[nonzero], b[nonzero])
    return float(stat), float(p)


def count_parameters(checkpoint_path: str) -> int:
    """Count trainable parameters from a checkpoint."""
    import torch
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state = ckpt.get('model_state_dict', ckpt)
    return sum(v.numel() for v in state.values())


def compute_metrics_for_class(pred: np.ndarray, target: np.ndarray, class_id: int) -> Dict[str, float]:
    """Compute all metrics for a specific class"""
    pred_binary = (pred == class_id).astype(np.float32)
    target_binary = (target == class_id).astype(np.float32)
    
    # Skip if class not present in target
    if target_binary.sum() == 0:
        return None
    
    fp_vol, fn_vol = compute_fp_fn_volume(pred_binary, target_binary)

    metrics = {
        'dice': compute_dice(pred_binary, target_binary),
        'surface_dice': compute_surface_dice(pred_binary, target_binary, tau=2.0),
        'hausdorff_95': compute_hausdorff_95(pred_binary, target_binary),
        'sensitivity': compute_sensitivity(pred_binary, target_binary),
        'specificity': compute_specificity(pred_binary, target_binary),
        'fp_volume_ml': fp_vol,
        'fn_volume_ml': fn_vol,
        'volume_pred': pred_binary.sum(),
        'volume_true': target_binary.sum(),
        'volume_diff': pred_binary.sum() - target_binary.sum()
    }

    return metrics


def evaluate_case(pred_path: str, label_path: str) -> Dict[str, Dict[str, float]]:
    """Evaluate a single case"""
    # Load volumes
    pred = nib.load(pred_path).get_fdata().astype(np.int32)
    label = nib.load(label_path).get_fdata().astype(np.int32)
    
    # Ensure same shape
    assert pred.shape == label.shape, f"Shape mismatch: {pred.shape} vs {label.shape}"
    
    # Class names
    class_names = {
        1: 'psma_tumor',
        2: 'psma_normal',
        3: 'fdg_tumor',
        4: 'fdg_normal'
    }
    
    # Compute metrics for each class
    results = {}
    for class_id, class_name in class_names.items():
        metrics = compute_metrics_for_class(pred, label, class_id)
        if metrics is not None:
            results[class_name] = metrics
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate segmentation predictions')
    parser.add_argument('--predictions_dir', type=str, required=True,
                        help='Directory containing prediction files')
    parser.add_argument('--labels_dir', type=str, required=True,
                        help='Directory containing ground truth labels')
    parser.add_argument('--output_file', type=str, default='metrics.csv',
                        help='Output CSV file for metrics')
    parser.add_argument('--json_output', type=str, default='metrics.json',
                        help='Output JSON file for detailed metrics')
    parser.add_argument('--compare_json', type=str, default=None,
                        help='Path to baseline metrics.json for Wilcoxon comparison')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (for reporting parameter count)')

    args = parser.parse_args()
    
    # Get list of prediction files
    pred_files = [f for f in os.listdir(args.predictions_dir) if f.endswith('.nii.gz')]
    pred_files.sort()
    
    print(f"Found {len(pred_files)} prediction files")
    
    # Evaluate each case
    all_results = {}
    metric_keys = ['dice', 'surface_dice', 'hausdorff_95', 'sensitivity',
                    'specificity', 'fp_volume_ml', 'fn_volume_ml']
    class_metrics = {
        cn: {m: [] for m in metric_keys}
        for cn in ['psma_tumor', 'psma_normal', 'fdg_tumor', 'fdg_normal']
    }
    
    for pred_file in tqdm(pred_files, desc='Evaluating'):
        case_id = pred_file.replace('.nii.gz', '')
        
        # Find corresponding label file
        label_file = f"{case_id}.nii.gz"
        label_path = os.path.join(args.labels_dir, label_file)
        
        if not os.path.exists(label_path):
            print(f"Warning: Label not found for {case_id}")
            continue
        
        pred_path = os.path.join(args.predictions_dir, pred_file)
        
        # Evaluate
        try:
            case_results = evaluate_case(pred_path, label_path)
            all_results[case_id] = case_results
            
            # Accumulate metrics
            for class_name in class_metrics.keys():
                if class_name in case_results:
                    for metric in metric_keys:
                        if metric in case_results[class_name]:
                            class_metrics[class_name][metric].append(
                                case_results[class_name][metric]
                            )
        except Exception as e:
            print(f"Error evaluating {case_id}: {str(e)}")
            continue
    
    # Compute summary statistics
    summary = {}
    rows = []

    for class_name, metrics in class_metrics.items():
        if len(metrics['dice']) > 0:
            s = {}
            for m in metric_keys:
                if metrics[m]:
                    s[f'{m}_mean'] = float(np.mean(metrics[m]))
                    s[f'{m}_std'] = float(np.std(metrics[m]))
            s['n_cases'] = len(metrics['dice'])
            summary[class_name] = s

            rows.append({
                'Class': class_name,
                'Dice': f"{s['dice_mean']:.4f}±{s['dice_std']:.4f}",
                'Surface Dice': f"{s.get('surface_dice_mean', 0):.4f}±{s.get('surface_dice_std', 0):.4f}",
                'HD95': f"{s.get('hausdorff_95_mean', 0):.2f}±{s.get('hausdorff_95_std', 0):.2f}",
                'FP Vol (mL)': f"{s.get('fp_volume_ml_mean', 0):.2f}±{s.get('fp_volume_ml_std', 0):.2f}",
                'FN Vol (mL)': f"{s.get('fn_volume_ml_mean', 0):.2f}±{s.get('fn_volume_ml_std', 0):.2f}",
                'N': s['n_cases'],
            })
    
    # Save results
    df = pd.DataFrame(rows)
    df.to_csv(args.output_file, index=False)
    print(f"\nMetrics saved to {args.output_file}")

    # Save detailed JSON
    detailed_results = {
        'per_case': all_results,
        'summary': summary,
        'per_case_dice': {  # Flat lists for easy Wilcoxon loading
            cn: class_metrics[cn]['dice']
            for cn in class_metrics if class_metrics[cn]['dice']
        },
    }

    # Report parameter count if checkpoint provided
    if args.checkpoint and os.path.exists(args.checkpoint):
        n_params = count_parameters(args.checkpoint)
        detailed_results['trainable_parameters'] = n_params
        print(f"\nTrainable parameters: {n_params:,}")

    with open(args.json_output, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    print(f"Detailed metrics saved to {args.json_output}")

    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)

    # Print key metrics
    if 'psma_tumor' in summary:
        print(f"\nPSMA Tumor Dice: {summary['psma_tumor']['dice_mean']:.4f} ± {summary['psma_tumor']['dice_std']:.4f}")
    if 'fdg_tumor' in summary:
        print(f"FDG Tumor Dice:  {summary['fdg_tumor']['dice_mean']:.4f} ± {summary['fdg_tumor']['dice_std']:.4f}")

    # Wilcoxon signed-rank test against baseline
    if args.compare_json and os.path.exists(args.compare_json):
        with open(args.compare_json, 'r') as f:
            baseline = json.load(f)
        baseline_dice = baseline.get('per_case_dice', {})

        print("\n" + "=" * 80)
        print("WILCOXON SIGNED-RANK TEST vs BASELINE")
        print("=" * 80)
        for cn in ['psma_tumor', 'fdg_tumor', 'psma_normal', 'fdg_normal']:
            current = class_metrics.get(cn, {}).get('dice', [])
            base = baseline_dice.get(cn, [])
            if current and base and len(current) == len(base):
                stat, p = wilcoxon_test(current, base)
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
                print(f"  {cn:15s}  W={stat:.1f}  p={p:.6f}  {sig}")
            elif current and base:
                print(f"  {cn:15s}  SKIPPED (length mismatch: {len(current)} vs {len(base)})")
        print("=" * 80)
        print("Significance: *** p<0.001, ** p<0.01, * p<0.05, n.s. not significant")
        print("Note: Apply Bonferroni correction for multiple comparisons (divide alpha by # tests).")


if __name__ == '__main__':
    main()
