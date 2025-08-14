#!/usr/bin/env python3
"""
Comprehensive evaluation script for CrackSegmenter.

This script evaluates all trained models on validation datasets, computes metrics,
and optionally saves predicted masks for analysis.
"""

import os
import sys
import glob
import math
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import logging
from typing import Dict, List, Tuple, Optional

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.models import create_model
from src.data.loaders import create_dataloaders
from src.training.metrics import evaluate_metrics
from src.utils.config import load_config


def setup_logging(log_dir: str = None):
    """Setup logging configuration."""
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, 'evaluation.log')),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )


class ModelEvaluator:
    """Comprehensive model evaluator for crack segmentation."""
    
    def __init__(self, config_path: str = None, device: str = None):
        """Initialize the evaluator."""
        self.device = torch.device(device if device else 
                                 ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Load configuration
        if config_path and os.path.exists(config_path):
            self.config = load_config(config_path)
        else:
            self.config = self._get_default_config()
        
        # Setup logging
        setup_logging(self.config.get('utils', {}).get('save_dir'))
        
        logging.info(f"Using device: {self.device}")
        logging.info(f"Configuration: {self.config}")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration for evaluation."""
        return {
            'model': {
                'num_classes': 1,
                'nChannel': 100,
                'backbone': 'resnet18',
                'pretrained': True,
                'use_cbam': True,
                'use_transformer': True,
                'architecture': 'Crack-Segmenter-v2',
                'use_dice': True,
                'use_bce': True,
                'baseline': False
            },
            'data': {
                'image_size': 448,
                'num_workers': 4,
                'mask_ext': '.png'
            },
            'evaluation': {
                'save_masks': True,
                'visualize': False,
                'mask_format': '.png',
                'compute_metrics': True
            },
            'utils': {
                'save_dir': './evaluation_results',
                'checkpoint_root': './trained_models',
                'dataset_root': './datasets'
            }
        }
    
    def get_available_models(self) -> List[str]:
        """Get list of available trained models."""
        checkpoint_root = self.config['utils']['checkpoint_root']
        available_models = []
        
        if os.path.exists(checkpoint_root):
            for item in os.listdir(checkpoint_root):
                item_path = os.path.join(checkpoint_root, item)
                if os.path.isdir(item_path):
                    # Check if it contains model checkpoints
                    checkpoints = glob.glob(os.path.join(item_path, "**/*.pth"), recursive=True)
                    if checkpoints:
                        available_models.append(item)
        
        return available_models
    
    def get_available_datasets(self) -> List[str]:
        """Get list of available datasets."""
        dataset_root = self.config['utils']['dataset_root']
        available_datasets = []
        
        if os.path.exists(dataset_root):
            for item in os.listdir(dataset_root):
                item_path = os.path.join(dataset_root, item)
                if os.path.isdir(item_path):
                    # Check if it has the expected structure
                    val_images = os.path.join(item_path, "val", "images")
                    val_masks = os.path.join(item_path, "val", "masks")
                    if os.path.exists(val_images) and os.path.exists(val_masks):
                        available_datasets.append(item)
        
        return available_datasets
    
    def load_model(self, architecture: str, checkpoint_path: str) -> torch.nn.Module:
        """Load a trained model from checkpoint."""
        try:
            # Create model
            if self.config['model']['baseline']:
                model = create_model(
                    architecture=architecture,
                    encoder_name=self.config['model']['backbone'],
                    in_channels=3,
                    num_classes=self.config['model']['num_classes'],
                    encoder_weights='imagenet' if self.config['model']['pretrained'] else None
                )
            else:
                model = create_model(
                    architecture=architecture,
                    input_dim=3,
                    embed_size=self.config['model']['nChannel']
                )
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model = model.to(self.device)
            model.eval()
            
            logging.info(f"Model loaded successfully: {architecture}")
            return model
            
        except Exception as e:
            logging.error(f"Failed to load model {architecture}: {e}")
            raise
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess image for model input."""
        # Read image
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Resize if needed
        target_size = self.config['data']['image_size']
        if img_bgr.shape[0] != target_size or img_bgr.shape[1] != target_size:
            img_bgr = cv2.resize(img_bgr, (target_size, target_size))
        
        # Convert to tensor
        img_tensor = torch.from_numpy(
            img_bgr.transpose(2, 0, 1).astype("float32") / 255.0
        ).unsqueeze(0).to(self.device)
        
        return img_tensor
    
    def load_ground_truth(self, mask_path: str) -> Tuple[np.ndarray, torch.Tensor]:
        """Load ground truth mask."""
        # Read mask
        mask_np = cv2.imread(mask_path, 0)
        if mask_np is None:
            raise ValueError(f"Could not read mask: {mask_path}")
        
        # Resize if needed
        target_size = self.config['data']['image_size']
        if mask_np.shape[0] != target_size or mask_np.shape[1] != target_size:
            mask_np = cv2.resize(mask_np, (target_size, target_size))
        
        # Convert to binary (0/1)
        mask_np = np.where(mask_np == 0, 0, 1).astype(np.uint8)
        
        # Convert to tensor
        mask_tensor = torch.from_numpy(mask_np).to(self.device)
        
        return mask_np, mask_tensor
    
    def create_binary_mask(self, prediction: torch.Tensor, 
                          ground_truth: torch.Tensor) -> torch.Tensor:
        """Create binary mask from model prediction."""
        if prediction.dim() == 4:  # (B, C, H, W)
            if prediction.size(1) == 1:  # Single channel
                pred_mask = torch.sigmoid(prediction.squeeze(1)) > 0.5
            else:  # Multiple channels
                pred_mask = prediction.argmax(1)
        else:  # (B, H, W) or (H, W)
            pred_mask = prediction
        
        # Ensure binary output
        pred_mask = (pred_mask > 0).float()
        
        return pred_mask
    
    def compute_metrics(self, prediction: torch.Tensor, 
                       ground_truth: torch.Tensor) -> Dict[str, float]:
        """Compute evaluation metrics."""
        try:
            metrics = evaluate_metrics(prediction, ground_truth, num_classes=1)
            return metrics
        except Exception as e:
            logging.warning(f"Failed to compute metrics: {e}")
            return {"IoU": float('nan'), "Dice": float('nan'), 
                   "Precision": float('nan'), "Recall": float('nan'), 
                   "F1 Score": float('nan')}
    
    def save_predicted_mask(self, mask: np.ndarray, output_path: str):
        """Save predicted mask to disk."""
        try:
            # Ensure mask is in correct format
            if mask.dtype != np.uint8:
                mask = (mask * 255).astype(np.uint8)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save mask
            cv2.imwrite(output_path, mask)
            
        except Exception as e:
            logging.error(f"Failed to save mask {output_path}: {e}")
    
    def visualize_results(self, image: np.ndarray, prediction: np.ndarray, 
                         ground_truth: np.ndarray, title: str = ""):
        """Visualize prediction results."""
        if not self.config['evaluation']['visualize']:
            return
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Input Image")
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(prediction, cmap='gray')
        plt.title("Prediction")
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(ground_truth, cmap='gray')
        plt.title("Ground Truth")
        plt.axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)
        plt.close()
    
    def evaluate_model_on_dataset(self, model: torch.nn.Module, 
                                architecture: str, dataset: str) -> Dict:
        """Evaluate a single model on a single dataset."""
        logging.info(f"Evaluating {architecture} on {dataset}")
        
        # Setup paths
        dataset_root = self.config['utils']['dataset_root']
        val_images_dir = os.path.join(dataset_root, dataset, "val", "images")
        val_masks_dir = os.path.join(dataset_root, dataset, "val", "masks")
        
        # Check if dataset exists
        if not os.path.exists(val_images_dir) or not os.path.exists(val_masks_dir):
            logging.warning(f"Dataset {dataset} not found or incomplete")
            return {
                "Model": architecture,
                "Dataset": dataset,
                "IoU": float('nan'),
                "Dice": float('nan'),
                "Precision": float('nan'),
                "Recall": float('nan'),
                "F1 Score": float('nan'),
                "Error": "Dataset not found"
            }
        
        # Get file lists
        image_files = sorted(glob.glob(os.path.join(val_images_dir, "*")))
        mask_files = sorted(glob.glob(os.path.join(val_masks_dir, "*")))
        
        if len(image_files) == 0:
            logging.warning(f"No images found in {val_images_dir}")
            return {
                "Model": architecture,
                "Dataset": dataset,
                "IoU": float('nan'),
                "Dice": float('nan'),
                "Precision": float('nan'),
                "Recall": float('nan'),
                "F1 Score": float('nan'),
                "Error": "No images found"
            }
        
        # Setup output directory for masks
        if self.config['evaluation']['save_masks']:
            masks_output_dir = os.path.join(
                self.config['utils']['save_dir'], 
                "predicted_masks", dataset, architecture
            )
            os.makedirs(masks_output_dir, exist_ok=True)
        
        # Initialize metrics
        all_metrics = {
            "IoU": [], "Dice": [], "Precision": [], 
            "Recall": [], "F1 Score": []
        }
        
        # Evaluation loop
        with torch.no_grad():
            for img_path, mask_path in tqdm(
                zip(image_files, mask_files), 
                total=len(image_files),
                desc=f"{architecture} on {dataset}"
            ):
                try:
                    # Load and preprocess
                    img_tensor = self.preprocess_image(img_path)
                    gt_np, gt_tensor = self.load_ground_truth(mask_path)
                    
                    # Model prediction
                    prediction = model(img_tensor)
                    if isinstance(prediction, tuple):
                        prediction = prediction[0]
                    
                    # Create binary mask
                    pred_mask = self.create_binary_mask(prediction, gt_tensor)
                    
                    # Compute metrics
                    if self.config['evaluation']['compute_metrics']:
                        metrics = self.compute_metrics(pred_mask, gt_tensor)
                        for key in all_metrics:
                            if key in metrics:
                                all_metrics[key].append(metrics[key])
                    
                    # Save predicted mask
                    if self.config['evaluation']['save_masks']:
                        mask_filename = os.path.splitext(os.path.basename(img_path))[0]
                        mask_ext = self.config['evaluation']['mask_format']
                        mask_output_path = os.path.join(
                            masks_output_dir, f"{mask_filename}{mask_ext}"
                        )
                        self.save_predicted_mask(
                            pred_mask.cpu().numpy(), mask_output_path
                        )
                    
                    # Visualize if requested
                    if self.config['evaluation']['visualize']:
                        self.visualize_results(
                            cv2.imread(img_path),
                            pred_mask.cpu().numpy(),
                            gt_np,
                            f"{architecture} on {dataset}"
                        )
                
                except Exception as e:
                    logging.error(f"Error processing {img_path}: {e}")
                    continue
        
        # Compute average metrics
        avg_metrics = {}
        for key, values in all_metrics.items():
            if values:
                avg_metrics[key] = float(np.nanmean(values))
            else:
                avg_metrics[key] = float('nan')
        
        # Prepare result
        result = {
            "Model": architecture,
            "Dataset": dataset,
            **avg_metrics
        }
        
        logging.info(f"Evaluation completed: {architecture} on {dataset}")
        return result
    
    def run_evaluation(self, models: List[str] = None, 
                      datasets: List[str] = None) -> pd.DataFrame:
        """Run comprehensive evaluation."""
        # Get available models and datasets if not specified
        if models is None:
            models = self.get_available_models()
        if datasets is None:
            datasets = self.get_available_datasets()
        
        if not models:
            logging.error("No models available for evaluation")
            return pd.DataFrame()
        
        if not datasets:
            logging.error("No datasets available for evaluation")
            return pd.DataFrame()
        
        logging.info(f"Starting evaluation of {len(models)} models on {len(datasets)} datasets")
        
        results = []
        
        # Evaluate each model on each dataset
        for architecture in models:
            for dataset in datasets:
                try:
                    # Find checkpoint
                    checkpoint_root = self.config['utils']['checkpoint_root']
                    checkpoint_path = os.path.join(
                        checkpoint_root, dataset, architecture, "best_model.pth"
                    )
                    
                    if not os.path.exists(checkpoint_path):
                        logging.warning(f"Checkpoint not found: {checkpoint_path}")
                        continue
                    
                    # Load model and evaluate
                    model = self.load_model(architecture, checkpoint_path)
                    result = self.evaluate_model_on_dataset(model, architecture, dataset)
                    results.append(result)
                    
                except Exception as e:
                    logging.error(f"Failed to evaluate {architecture} on {dataset}: {e}")
                    continue
        
        # Create results DataFrame
        if results:
            results_df = pd.DataFrame(results)
            
            # Save results
            output_dir = self.config['utils']['save_dir']
            os.makedirs(output_dir, exist_ok=True)
            
            csv_path = os.path.join(output_dir, "evaluation_results.csv")
            results_df.to_csv(csv_path, index=False)
            
            logging.info(f"Evaluation results saved to: {csv_path}")
            
            return results_df
        else:
            logging.error("No evaluation results generated")
            return pd.DataFrame()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Evaluate CrackSegmenter models')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--models', nargs='+', default=None,
                       help='Specific models to evaluate')
    parser.add_argument('--datasets', nargs='+', default=None,
                       help='Specific datasets to evaluate on')
    parser.add_argument('--save_masks', action='store_true',
                       help='Save predicted masks')
    parser.add_argument('--visualize', action='store_true',
                       help='Show visualization during evaluation')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = ModelEvaluator(config_path=args.config)
    
    # Update configuration based on arguments
    if args.save_masks:
        evaluator.config['evaluation']['save_masks'] = True
    if args.visualize:
        evaluator.config['evaluation']['visualize'] = True
    if args.output_dir:
        evaluator.config['utils']['save_dir'] = args.output_dir
    
    # Run evaluation
    results = evaluator.run_evaluation(
        models=args.models,
        datasets=args.datasets
    )
    
    if not results.empty:
        print("\n=== Evaluation Results ===")
        print(results.to_string(index=False))
        
        # Print summary statistics
        print("\n=== Summary Statistics ===")
        for metric in ['IoU', 'Dice', 'Precision', 'Recall', 'F1 Score']:
            if metric in results.columns:
                values = results[metric].dropna()
                if len(values) > 0:
                    print(f"{metric}: {values.mean():.4f} ± {values.std():.4f}")
    else:
        print("No evaluation results generated")


if __name__ == '__main__':
    main()
