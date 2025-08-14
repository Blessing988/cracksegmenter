#!/usr/bin/env python3
"""
Simple evaluation script for single model evaluation.

This script evaluates a single trained model on validation data and computes metrics.
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.models import create_model
from src.data.loaders import create_dataloaders
from src.training.losses import get_loss_function
from src.training.metrics import evaluate_metrics, print_metrics
from src.utils.config import load_config


def setup_logging():
    """Setup basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def evaluate_model(config_path: str, model_path: str, dataset_name: str = None):
    """Evaluate a single model."""
    # Load configuration
    config = load_config(config_path)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Create model
    if config['model']['baseline']:
        model = create_model(
            architecture=config['model']['architecture'],
            encoder_name=config['model']['backbone'],
            in_channels=3,
            num_classes=config['model']['num_classes'],
            encoder_weights='imagenet' if config['model']['pretrained'] else None
        )
    else:
        model = create_model(
            architecture=config['model']['architecture'],
            input_dim=3,
            embed_size=config['model']['nChannel']
        )
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    logging.info(f"Model loaded from: {model_path}")
    
    # Create data loader
    if dataset_name:
        config['data']['dataset_name'] = dataset_name
    
    dataloaders = create_dataloaders(
        dataset_name=config['data']['dataset_name'],
        root_dir=config['data']['root_dir'],
        batch_size=1,  # Use batch size 1 for evaluation
        image_size=config['data']['image_size'],
        num_workers=config['data']['num_workers']
    )
    
    val_loader = dataloaders['val_loader']
    logging.info(f"Validation samples: {len(val_loader.dataset)}")
    
    # Loss function
    if config['model']['use_dice'] and config['model']['use_bce']:
        criterion = get_loss_function('combined')
    elif config['model']['use_dice']:
        criterion = get_loss_function('dice')
    else:
        criterion = get_loss_function('bce')
    
    # Evaluation
    model.eval()
    val_loss = 0
    all_metrics = {'IoU': [], 'Dice': [], 'Precision': [], 'Recall': [], 'F1 Score': []}
    
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            if isinstance(outputs, tuple):
                main_output = outputs[0]
            else:
                main_output = outputs
            
            # Compute loss
            if main_output.dim() == 4 and main_output.size(1) == 1:
                main_output = main_output.squeeze(1)
            
            loss = criterion(main_output, masks.float())
            val_loss += loss.item()
            
            # Compute metrics
            preds = torch.sigmoid(main_output) > 0.5
            batch_metrics = evaluate_metrics(preds.cpu(), masks.cpu(), config['model']['num_classes'])
            
            for key in all_metrics:
                if key in batch_metrics:
                    all_metrics[key].append(batch_metrics[key])
    
    # Average metrics
    val_loss /= len(val_loader)
    avg_metrics = {key: np.nanmean(values) for key, values in all_metrics.items()}
    
    # Print results
    print(f"\n=== Evaluation Results ===")
    print(f"Validation Loss: {val_loss:.4f}")
    print("\nMetrics:")
    print_metrics(avg_metrics, "Validation")
    
    return val_loss, avg_metrics


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Evaluate CrackSegmenter model')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Dataset name (overrides config)')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Check if files exist
    if not os.path.exists(args.config):
        logging.error(f"Configuration file not found: {args.config}")
        return
    
    if not os.path.exists(args.model_path):
        logging.error(f"Model file not found: {args.model_path}")
        return
    
    # Run evaluation
    try:
        val_loss, metrics = evaluate_model(args.config, args.model_path, args.dataset)
        logging.info("Evaluation completed successfully")
    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        raise


if __name__ == '__main__':
    main()
