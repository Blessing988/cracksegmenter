"""
Basic tests for CrackSegmenter package.

These tests verify that the package structure is correct and basic functionality works.
"""

import sys
import os
import unittest
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))


class TestPackageStructure(unittest.TestCase):
    """Test that the package structure is correct."""
    
    def test_imports(self):
        """Test that all modules can be imported."""
        try:
            from models import create_model
            from data.loaders import create_dataloaders
            from training.losses import get_loss_function
            from training.metrics import evaluate_metrics
            from utils.config import load_config
            print("✅ All modules imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import modules: {e}")
    
    def test_model_creation(self):
        """Test that models can be created."""
        try:
            from models import create_model
            import torch
            
            # Test CrackSegmenter model creation
            model = create_model('Crack-Segmenter-v2', input_dim=3, embed_size=100)
            self.assertIsNotNone(model)
            
            # Test baseline model creation
            baseline_model = create_model('unet', in_channels=3, num_classes=1)
            self.assertIsNotNone(baseline_model)
            
            print("✅ Model creation successful")
            
        except Exception as e:
            self.fail(f"Failed to create models: {e}")
    
    def test_loss_functions(self):
        """Test that loss functions can be created."""
        try:
            from training.losses import get_loss_function
            
            # Test different loss functions
            bce_loss = get_loss_function('bce')
            dice_loss = get_loss_function('dice')
            combined_loss = get_loss_function('combined')
            
            self.assertIsNotNone(bce_loss)
            self.assertIsNotNone(dice_loss)
            self.assertIsNotNone(combined_loss)
            
            print("✅ Loss function creation successful")
            
        except Exception as e:
            self.fail(f"Failed to create loss functions: {e}")
    
    def test_config_utilities(self):
        """Test configuration utilities."""
        try:
            from utils.config import get_default_config, validate_config
            
            # Test default config
            config = get_default_config()
            self.assertIn('model', config)
            self.assertIn('training', config)
            self.assertIn('data', config)
            self.assertIn('utils', config)
            
            # Test validation
            validate_config(config)
            
            print("✅ Configuration utilities working")
            
        except Exception as e:
            self.fail(f"Configuration utilities failed: {e}")
    
    def test_transforms(self):
        """Test data transforms."""
        try:
            from data.transforms import get_transforms, get_training_transforms
            
            # Test transform creation
            train_transforms = get_training_transforms(448)
            self.assertIsNotNone(train_transforms)
            
            print("✅ Data transforms working")
            
        except Exception as e:
            self.fail(f"Data transforms failed: {e}")


class TestModelArchitectures(unittest.TestCase):
    """Test specific model architectures."""
    
    def test_cracksegmenter_architectures(self):
        """Test CrackSegmenter architecture variants."""
        try:
            from models.cracksegmenter import (
                MSFormer_SAE_AGF, MSFormer_SAE, MSFormer_AGF,
                MSFormer_v1, MSFormer_v2, MSFormer_v3
            )
            import torch
            
            # Test each architecture
            architectures = [
                MSFormer_SAE_AGF, MSFormer_SAE, MSFormer_AGF,
                MSFormer_v1, MSFormer_v2, MSFormer_v3
            ]
            
            for arch_class in architectures:
                model = arch_class(input_dim=3, embed_size=100)
                self.assertIsNotNone(model)
                
                # Test forward pass
                x = torch.randn(1, 3, 448, 448)
                with torch.no_grad():
                    output = model(x)
                
                self.assertIsNotNone(output)
            
            print("✅ All CrackSegmenter architectures working")
            
        except Exception as e:
            self.fail(f"CrackSegmenter architectures failed: {e}")
    
    def test_baseline_models(self):
        """Test baseline model architectures."""
        try:
            from models.baselines import get_baseline_model
            import torch
            
            # Test different baseline models
            baseline_names = ['unet', 'fcn', 'deeplabv3', 'deeplabv3plus']
            
            for name in baseline_names:
                model = get_baseline_model(name, in_channels=3, num_classes=1)
                self.assertIsNotNone(model)
                
                # Test forward pass
                x = torch.randn(1, 3, 448, 448)
                with torch.no_grad():
                    output = model(x)
                
                self.assertIsNotNone(output)
            
            print("✅ All baseline models working")
            
        except Exception as e:
            self.fail(f"Baseline models failed: {e}")


if __name__ == '__main__':
    print("🧪 Running CrackSegmenter tests...")
    print("=" * 50)
    
    # Run tests
    unittest.main(verbosity=2)
