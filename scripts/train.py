#!/usr/bin/env python3
"""
Main training script for CrackSegmenter.

This script handles training of both baseline models and CrackSegmenter variants
based on the configuration in config.yaml.
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import csv
import numpy as np
from tqdm import tqdm
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.models import create_model
from src.data.loaders import create_dataloaders
from src.training.losses import get_loss_function
from src.training.metrics import evaluate_metrics, print_metrics
from src.utils.config import load_config, save_config


def setup_logging(log_dir: str):
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )


def train_one_epoch_baseline(model, dataloader, criterion, optimizer, device, 
                            num_classes=1, config=None):
    """Train one epoch for baseline models."""
    model.train()
    epoch_loss = 0
    metrics = {'IoU': [], 'Dice': [], 'Precision': [], 'Recall': [], 'F1 Score': []}
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for images, masks in progress_bar:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        
        # Forward pass
        if config and config['model']['architecture'] == 'FCN':
            outputs = model(images)['out']
        else:
            outputs = model(images)
            
        outputs = outputs.squeeze(1)
        loss = criterion(outputs, masks.float())
        
        # Backward pass
        loss.backward()
        optimizer.step()

        # Calculate metrics
        preds = torch.sigmoid(outputs) > 0.5
        batch_metrics = evaluate_metrics(preds.cpu(), masks.cpu(), num_classes)
        
        for key in metrics:
            metrics[key].append(batch_metrics[key])

        epoch_loss += loss.item() * images.size(0)
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'IoU': f'{batch_metrics["IoU"]:.4f}'
        })

    epoch_loss /= len(dataloader.dataset)
    avg_metrics = {key: np.nanmean(metrics[key]) for key in metrics}

    return epoch_loss, avg_metrics


def validate_baseline(model, dataloader, criterion, device, num_classes=1, config=None):
    """Validate baseline models."""
    model.eval()
    epoch_loss = 0
    metrics = {'IoU': [], 'Dice': [], 'Precision': [], 'Recall': [], 'F1 Score': []}

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation")
        
        for images, masks in progress_bar:
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            if config and config['model']['architecture'] == 'FCN':
                outputs = model(images)['out']
            else:
                outputs = model(images)
                
            outputs = outputs.squeeze(1)
            loss = criterion(outputs, masks.float())
            
            # Calculate metrics
            preds = torch.sigmoid(outputs) > 0.5
            batch_metrics = evaluate_metrics(preds.cpu(), masks.cpu(), num_classes)
            
            for key in metrics:
                metrics[key].append(batch_metrics[key])

            epoch_loss += loss.item() * images.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'IoU': f'{batch_metrics["IoU"]:.4f}'
            })

    epoch_loss /= len(dataloader.dataset)
    avg_metrics = {key: np.nanmean(metrics[key]) for key in metrics}

    return epoch_loss, avg_metrics


def train_one_epoch_cracksegmenter(model, dataloader, criterion, optimizer, device,
                            num_classes=1):
    """Train one epoch for CrackSegmenter models."""
    model.train()
    epoch_loss = 0
    metrics = {'IoU': [], 'Dice': [], 'Precision': [], 'Recall': [], 'F1 Score': []}
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for images, masks in progress_bar:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        
        # Forward pass (CrackSegmenter returns multiple outputs)
        outputs = model(images)
        if isinstance(outputs, tuple):
            main_output = outputs[0]  # Main prediction
        else:
            main_output = outputs
            
        main_output = main_output.squeeze(1)
        loss = criterion(main_output, masks.float())
        
        # Backward pass
        loss.backward()
        optimizer.step()

        # Calculate metrics
        preds = torch.sigmoid(main_output) > 0.5
        batch_metrics = evaluate_metrics(preds.cpu(), masks.cpu(), num_classes)
        
        for key in metrics:
            metrics[key].append(batch_metrics[key])

        epoch_loss += loss.item() * images.size(0)
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'IoU': f'{batch_metrics["IoU"]:.4f}'
        })

    epoch_loss /= len(dataloader.dataset)
    avg_metrics = {key: np.nanmean(metrics[key]) for key in metrics}

    return epoch_loss, avg_metrics


def validate_cracksegmenter(model, dataloader, criterion, device, num_classes=1):
    """Validate CrackSegmenter models."""
    model.eval()
    epoch_loss = 0
    metrics = {'IoU': [], 'Dice': [], 'Precision': [], 'Recall': [], 'F1 Score': []}

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation")
        
        for images, masks in progress_bar:
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            outputs = model(images)
            if isinstance(outputs, tuple):
                main_output = outputs[0]
            else:
                main_output = outputs
                
            main_output = main_output.squeeze(1)
            loss = criterion(main_output, masks.float())
            
            # Calculate metrics
            preds = torch.sigmoid(main_output) > 0.5
            batch_metrics = evaluate_metrics(preds.cpu(), masks.cpu(), num_classes)
            
            for key in metrics:
                metrics[key].append(batch_metrics[key])

            epoch_loss += loss.item() * images.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'IoU': f'{batch_metrics["IoU"]:.4f}'
            })

    epoch_loss /= len(dataloader.dataset)
    avg_metrics = {key: np.nanmean(metrics[key]) for key in metrics}

    return epoch_loss, avg_metrics


def save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, 
                   save_dir, filename):
    """Save model checkpoint."""
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_val_loss': best_val_loss,
    }
    
    checkpoint_path = os.path.join(save_dir, filename)
    torch.save(checkpoint, checkpoint_path)
    logging.info(f"Checkpoint saved: {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description='Train CrackSegmenter')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    save_dir = config['utils']['save_dir']
    setup_logging(save_dir)
    
    logging.info("Starting CrackSegmenter training")
    logging.info(f"Configuration: {config}")
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Create model
    if config['model']['baseline']:
        logging.info("Creating baseline model")
        model = create_model(
            architecture=config['model']['architecture'],
            encoder_name=config['model']['backbone'],
            in_channels=3,
            num_classes=config['model']['num_classes'],
            encoder_weights='imagenet' if config['model']['pretrained'] else None
        )
    else:
        logging.info("Creating CrackSegmenter model")
        model = create_model(
            architecture=config['model']['architecture'],
            input_dim=3,
            embed_size=config['model']['nChannel']
        )
    
    model = model.to(device)
    
    # Create data loaders
    logging.info("Creating data loaders")
    dataloaders = create_dataloaders(
        dataset_name=config['data']['dataset_name'],
        root_dir=config['data']['root_dir'],
        batch_size=config['training']['batch_size'],
        image_size=config['data']['image_size'],
        num_workers=config['data']['num_workers']
    )
    
    train_loader = dataloaders['train_loader']
    val_loader = dataloaders['val_loader']
    
    logging.info(f"Train samples: {len(train_loader.dataset)}")
    logging.info(f"Val samples: {len(val_loader.dataset)}")
    
    # Loss function
    if config['model']['use_dice'] and config['model']['use_bce']:
        criterion = get_loss_function('combined')
    elif config['model']['use_dice']:
        criterion = get_loss_function('dice')
    else:
        criterion = get_loss_function('bce')
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', patience=20, factor=0.5, verbose=True
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        logging.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        logging.info(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    logging.info("Starting training loop")
    
    for epoch in range(start_epoch, config['training']['num_epochs']):
        logging.info(f"\nEpoch {epoch + 1}/{config['training']['num_epochs']}")
        
        # Training
        if config['model']['baseline']:
            train_loss, train_metrics = train_one_epoch_baseline(
                model, train_loader, criterion, optimizer, device,
                config['model']['num_classes'], config
            )
        else:
            train_loss, train_metrics = train_one_epoch_cracksegmenter(
                model, train_loader, criterion, optimizer, device,
                config['model']['num_classes']
            )
        
        # Validation
        if config['model']['baseline']:
            val_loss, val_metrics = validate_baseline(
                model, val_loader, criterion, device,
                config['model']['num_classes'], config
            )
        else:
            val_loss, val_metrics = validate_cracksegmenter(
                model, val_loader, criterion, device,
                config['model']['num_classes']
            )
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Log metrics
        logging.info(f"Train Loss: {train_loss:.4f}")
        logging.info(f"Val Loss: {val_loss:.4f}")
        logging.info("Train Metrics:")
        print_metrics(train_metrics, "Train Metrics")
        logging.info("Val Metrics:")
        print_metrics(val_metrics, "Validation Metrics")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_val_loss,
                save_dir, 'best_model.pth'
            )
            logging.info("New best model saved!")
        
        # Save regular checkpoint
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_val_loss,
                save_dir, f'checkpoint_epoch_{epoch + 1}.pth'
            )
        
        # Early stopping
        if epoch > 50 and val_loss > best_val_loss * 1.1:
            logging.info("Early stopping triggered")
            break
    
    logging.info("Training completed!")
    logging.info(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()
