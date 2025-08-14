#!/usr/bin/env python3
"""
Validation metrics computation script for CrackSegmenter.

This script computes comprehensive validation metrics for predicted masks against
ground truth masks, supporting both existing and new metrics (IoU, Dice, XOR, HM).
"""

import os
import sys
import glob
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.training.metrics import calculate_iou, calculate_dice, calculate_xor_metric, calculate_hm_metric


def setup_logging(log_level: str = 'INFO'):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def load_mask(mask_path: str, threshold: float = 0.5) -> np.ndarray:
    """Load and binarize mask from file."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not load mask from {mask_path}")
    
    # Normalize to 0-1 range and binarize
    mask = mask.astype(np.float32) / 255.0
    mask = (mask > threshold).astype(np.uint8)
    return mask


def get_matching_files(pred_dir: str, gt_dir: str, pred_ext: str = '*.png', gt_ext: str = '*.png') -> List[Tuple[str, str]]:
    """Get matching predicted and ground truth mask files."""
    pred_files = sorted(glob.glob(os.path.join(pred_dir, pred_ext)))
    gt_files = sorted(glob.glob(os.path.join(gt_dir, gt_ext)))
    
    # Create dictionaries for quick lookup
    pred_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in pred_files}
    gt_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in gt_files}
    
    # Find matching files
    matching_pairs = []
    for basename in pred_dict.keys():
        if basename in gt_dict:
            matching_pairs.append((pred_dict[basename], gt_dict[basename]))
        else:
            logging.warning(f"No ground truth found for prediction: {basename}")
    
    return matching_pairs


def evaluate_model_dataset(model_name: str, dataset_name: str, 
                          pred_base_path: str, gt_base_path: str,
                          pred_ext: str = '*.png', gt_ext: str = '*.png') -> Optional[Tuple[Dict, List]]:
    """Evaluate a single model-dataset combination."""
    
    # Construct paths
    pred_dir = os.path.join(pred_base_path, model_name, dataset_name)
    gt_dir = os.path.join(gt_base_path, dataset_name, 'val', 'masks')
    
    # Check if directories exist
    if not os.path.exists(pred_dir):
        logging.warning(f"Prediction directory not found: {pred_dir}")
        return None
    
    if not os.path.exists(gt_dir):
        logging.warning(f"Ground truth directory not found: {gt_dir}")
        return None
    
    # Get matching files
    matching_pairs = get_matching_files(pred_dir, gt_dir, pred_ext, gt_ext)
    
    if not matching_pairs:
        logging.warning(f"No matching files found for {model_name} on {dataset_name}")
        return None
    
    # Compute metrics for each pair
    all_metrics = []
    failed_count = 0
    
    for pred_file, gt_file in tqdm(matching_pairs, desc=f"{model_name}-{dataset_name}"):
        try:
            # Load masks
            pred_mask = load_mask(pred_file)
            gt_mask = load_mask(gt_file)
            
            # Ensure same shape
            if pred_mask.shape != gt_mask.shape:
                logging.warning(f"Shape mismatch for {os.path.basename(pred_file)}: "
                              f"pred {pred_mask.shape} vs gt {gt_mask.shape}")
                # Resize prediction to match ground truth
                pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]), 
                                     interpolation=cv2.INTER_NEAREST)
                pred_mask = (pred_mask > 0.5).astype(np.uint8)
            
            # Convert to tensors for metric computation
            import torch
            pred_tensor = torch.from_numpy(pred_mask).unsqueeze(0).unsqueeze(0).float()
            gt_tensor = torch.from_numpy(gt_mask).unsqueeze(0).unsqueeze(0).float()
            
            # Compute metrics
            metrics = {
                'image_name': os.path.basename(pred_file),
                'IoU': calculate_iou(pred_tensor, gt_tensor),
                'Dice': calculate_dice(pred_tensor, gt_tensor),
                'XOR': calculate_xor_metric(pred_tensor, gt_tensor),
                'HM': calculate_hm_metric(pred_tensor, gt_tensor)
            }
            
            all_metrics.append(metrics)
            
        except Exception as e:
            logging.error(f"Error processing {pred_file}: {e}")
            failed_count += 1
            continue
    
    if not all_metrics:
        logging.error(f"No successful metric computations for {model_name} on {dataset_name}")
        return None
    
    # Compute average metrics
    avg_metrics = {
        'Model': model_name,
        'Dataset': dataset_name,
        'IoU': np.mean([m['IoU'] for m in all_metrics]),
        'Dice': np.mean([m['Dice'] for m in all_metrics]),
        'XOR': np.mean([m['XOR'] for m in all_metrics]),
        'HM': np.mean([m['HM'] for m in all_metrics]),
        'Total_Images': len(matching_pairs),
        'Successful_Images': len(all_metrics),
        'Failed_Images': failed_count
    }
    
    logging.info(f"Completed {model_name} on {dataset_name}: "
                f"IoU={avg_metrics['IoU']:.4f}, "
                f"Dice={avg_metrics['Dice']:.4f}, "
                f"XOR={avg_metrics['XOR']:.4f}, "
                f"HM={avg_metrics['HM']:.4f}")
    
    return avg_metrics, all_metrics


def compute_validation_metrics(models_to_evaluate: List[str], 
                              datasets_to_evaluate: List[str],
                              pred_base_path: str,
                              gt_base_path: str,
                              save_detailed: bool = True, 
                              output_dir: str = './validation_results') -> Tuple[List, List]:
    """
    Compute validation metrics for all model-dataset combinations.
    
    Args:
        models_to_evaluate: List of models to evaluate
        datasets_to_evaluate: List of datasets to evaluate
        pred_base_path: Base path for predicted masks
        gt_base_path: Base path for ground truth masks
        save_detailed: Whether to save detailed per-image metrics
        output_dir: Directory to save results
        
    Returns:
        Tuple of (summary_results, detailed_results)
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Storage for results
    summary_results = []
    detailed_results = []
    
    total_combinations = len(models_to_evaluate) * len(datasets_to_evaluate)
    current_combination = 0
    
    for model_name in models_to_evaluate:
        for dataset_name in datasets_to_evaluate:
            current_combination += 1
            logging.info(f"Evaluating combination {current_combination}/{total_combinations}: "
                        f"{model_name} on {dataset_name}")
            
            result = evaluate_model_dataset(model_name, dataset_name, pred_base_path, gt_base_path)
            
            if result is not None:
                avg_metrics, individual_metrics = result
                summary_results.append(avg_metrics)
                
                if save_detailed:
                    for metrics in individual_metrics:
                        metrics['Model'] = model_name
                        metrics['Dataset'] = dataset_name
                        detailed_results.append(metrics)
            else:
                # Add failed entry to summary
                failed_entry = {
                    'Model': model_name,
                    'Dataset': dataset_name,
                    'IoU': np.nan,
                    'Dice': np.nan,
                    'XOR': np.nan,
                    'HM': np.nan,
                    'Total_Images': 0,
                    'Successful_Images': 0,
                    'Failed_Images': 0
                }
                summary_results.append(failed_entry)
    
    # Save summary results
    if summary_results:
        summary_df = pd.DataFrame(summary_results)
        summary_csv_path = os.path.join(output_dir, 'validation_metrics_summary.csv')
        summary_df.to_csv(summary_csv_path, index=False)
        logging.info(f"Summary results saved to: {summary_csv_path}")
        
        # Print summary statistics
        print("\n" + "="*80)
        print("VALIDATION METRICS SUMMARY")
        print("="*80)
        
        # Overall statistics
        valid_results = summary_df.dropna()
        if not valid_results.empty:
            print(f"Successfully evaluated combinations: {len(valid_results)}")
            print(f"Failed combinations: {len(summary_df) - len(valid_results)}")
            print("\nOverall Average Metrics:")
            print(f"IoU: {valid_results['IoU'].mean():.4f} ± {valid_results['IoU'].std():.4f}")
            print(f"Dice: {valid_results['Dice'].mean():.4f} ± {valid_results['Dice'].std():.4f}")
            print(f"XOR:  {valid_results['XOR'].mean():.4f} ± {valid_results['XOR'].std():.4f}")
            print(f"HM:   {valid_results['HM'].mean():.4f} ± {valid_results['HM'].std():.4f}")
            
            # Best performing models
            print(f"\nTop 5 Models by IoU:")
            top_iou = valid_results.nlargest(5, 'IoU')[['Model', 'Dataset', 'IoU', 'Dice']]
            print(top_iou.to_string(index=False))
            
            print(f"\nTop 5 Models by Dice Score:")
            top_dice = valid_results.nlargest(5, 'Dice')[['Model', 'Dataset', 'IoU', 'Dice']]
            print(top_dice.to_string(index=False))
    
    # Save detailed results if requested
    if save_detailed and detailed_results:
        detailed_df = pd.DataFrame(detailed_results)
        detailed_csv_path = os.path.join(output_dir, 'validation_metrics_detailed.csv')
        detailed_df.to_csv(detailed_csv_path, index=False)
        logging.info(f"Detailed results saved to: {detailed_csv_path}")
    
    return summary_results, detailed_results


def generate_model_comparison_report(summary_results: List[Dict], output_dir: str = './validation_results'):
    """Generate additional analysis reports."""
    
    if not summary_results:
        return
    
    df = pd.DataFrame(summary_results)
    valid_df = df.dropna()
    
    if valid_df.empty:
        return
    
    # Model performance summary
    model_summary = valid_df.groupby('Model').agg({
        'IoU': ['mean', 'std', 'count'],
        'Dice': ['mean', 'std'],
        'XOR': ['mean', 'std'],
        'HM': ['mean', 'std']
    }).round(4)
    
    model_summary.columns = ['_'.join(col).strip() for col in model_summary.columns]
    model_summary_path = os.path.join(output_dir, 'model_performance_summary.csv')
    model_summary.to_csv(model_summary_path)
    
    # Dataset difficulty analysis
    dataset_summary = valid_df.groupby('Dataset').agg({
        'IoU': ['mean', 'std', 'count'],
        'Dice': ['mean', 'std'],
        'XOR': ['mean', 'std'],
        'HM': ['mean', 'std']
    }).round(4)
    
    dataset_summary.columns = ['_'.join(col).strip() for col in dataset_summary.columns]
    dataset_summary_path = os.path.join(output_dir, 'dataset_difficulty_analysis.csv')
    dataset_summary.to_csv(dataset_summary_path)
    
    logging.info(f"Additional reports saved to: {output_dir}")


def main():
    """Main function to run the validation metrics computation."""
    parser = argparse.ArgumentParser(description='Compute validation metrics for CrackSegmenter models')
    parser.add_argument('--pred_base_path', type=str, required=True,
                       help='Base path for predicted masks')
    parser.add_argument('--gt_base_path', type=str, required=True,
                       help='Base path for ground truth masks')
    parser.add_argument('--models', nargs='+', default=None,
                       help='Specific models to evaluate')
    parser.add_argument('--datasets', nargs='+', default=None,
                       help='Specific datasets to evaluate')
    parser.add_argument('--output_dir', type=str, default='./validation_results',
                       help='Output directory for results')
    parser.add_argument('--save_detailed', action='store_true',
                       help='Save detailed per-image metrics')
    parser.add_argument('--log_level', type=str, default='INFO',
                       help='Logging level (DEBUG, INFO, WARNING, ERROR)')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Default models and datasets if not specified
    if args.models is None:
        args.models = ['Crack-Segmenter-v2', 'Crack-Segmenter-v1', 'Crack-Segmenter-v0']
    
    if args.datasets is None:
        args.datasets = ['cracktree200', 'cfd', 'forest', 'gaps_384']
    
    logging.info(f"Evaluating models: {args.models}")
    logging.info(f"Evaluating datasets: {args.datasets}")
    logging.info(f"Prediction base path: {args.pred_base_path}")
    logging.info(f"Ground truth base path: {args.gt_base_path}")
    
    # Compute validation metrics
    summary_results, detailed_results = compute_validation_metrics(
        models_to_evaluate=args.models,
        datasets_to_evaluate=args.datasets,
        pred_base_path=args.pred_base_path,
        gt_base_path=args.gt_base_path,
        save_detailed=args.save_detailed,
        output_dir=args.output_dir
    )
    
    # Generate additional reports
    generate_model_comparison_report(summary_results, args.output_dir)
    
    print("\nValidation metrics computation completed!")
    print(f"Results saved in {args.output_dir}/")


if __name__ == "__main__":
    main()
