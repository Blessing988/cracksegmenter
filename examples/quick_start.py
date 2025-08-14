#!/usr/bin/env python3
"""
Quick start example for CrackSegmenter.

This script demonstrates how to:
1. Load a configuration
2. Create a model
3. Set up data loaders
4. Run training
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from utils.config import load_config, get_default_config, create_config_file
from models import create_model
from data.loaders import create_dataloaders
from training.losses import get_loss_function
from training.metrics import evaluate_metrics, print_metrics
import torch


def main():
    """Main function demonstrating CrackSegmenter usage."""
    
    print("🚀 CrackSegmenter Quick Start Example")
    print("=" * 50)
    
    # 1. Configuration
    print("\n1. Setting up configuration...")
    
    # Check if config exists, otherwise create default
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        print("Creating default configuration file...")
        create_config_file(config_path)
    
    # Load configuration
    config = load_config(config_path)
    print("✅ Configuration loaded successfully!")
    
    # 2. Model Creation
    print("\n2. Creating model...")
    
    # Example: Create CrackSegmenter model
    model = create_model(
        architecture=config['model']['architecture'],
        input_dim=3,
        embed_size=config['model']['nChannel']
    )
    
    print(f"✅ Model created: {config['model']['architecture']}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 3. Data Loaders
    print("\n3. Setting up data loaders...")
    
    try:
        # Note: This will fail if dataset doesn't exist, but shows the API
        dataloaders = create_dataloaders(
            dataset_name=config['data']['dataset_name'],
            root_dir=config['data']['root_dir'],
            batch_size=config['training']['batch_size'],
            image_size=config['data']['image_size'],
            num_workers=config['data']['num_workers']
        )
        
        print("✅ Data loaders created successfully!")
        print(f"   Train samples: {len(dataloaders['train_loader'].dataset)}")
        print(f"   Val samples: {len(dataloaders['val_loader'].dataset)}")
        
    except Exception as e:
        print(f"⚠️  Data loader creation failed (expected if dataset not available): {e}")
        print("   This is normal if you haven't set up your dataset yet.")
    
    # 4. Loss Function
    print("\n4. Setting up loss function...")
    
    criterion = get_loss_function('combined')
    print("✅ Loss function created: Combined Loss (BCE + Dice)")
    
    # 5. Training Setup
    print("\n5. Training setup...")
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ Device: {device}")
    
    # Move model to device
    model = model.to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    print("✅ Optimizer: Adam")
    
    # 6. Example Forward Pass
    print("\n6. Testing model forward pass...")
    
    try:
        # Create dummy input
        batch_size = 2
        input_tensor = torch.randn(
            batch_size, 3, config['data']['image_size'], config['data']['image_size']
        ).to(device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(input_tensor)
        
        if isinstance(outputs, tuple):
            main_output = outputs[0]
        else:
            main_output = outputs
        
        print(f"✅ Forward pass successful!")
        print(f"   Input shape: {input_tensor.shape}")
        print(f"   Output shape: {main_output.shape}")
        
        # Test loss computation
        dummy_target = torch.randint(0, 2, main_output.shape).float().to(device)
        loss = criterion(main_output, dummy_target)
        print(f"   Loss value: {loss.item():.4f}")
        
    except Exception as e:
        print(f"⚠️  Forward pass failed: {e}")
    
    # 7. Summary
    print("\n" + "=" * 50)
    print("🎉 CrackSegmenter setup completed successfully!")
    print("\nNext steps:")
    print("1. Prepare your dataset in the specified directory")
    print("2. Update the configuration file with your dataset path")
    print("3. Run training: python scripts/train.py")
    print("4. Run inference: python scripts/inference.py --model_path <path> --image_path <path>")
    
    print("\nFor more information, see the README.md file.")
    print("Happy training! 🚀")


if __name__ == "__main__":
    main()
