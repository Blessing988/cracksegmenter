#!/usr/bin/env python3
"""
Tests for validation metrics functionality.
"""

import sys
import os
import unittest
from pathlib import Path
import tempfile
import shutil
import numpy as np
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from training.metrics import calculate_xor_metric, calculate_hm_metric


class TestValidationMetrics(unittest.TestCase):
    """Test validation metrics functionality."""
    
    def test_xor_metric_basic(self):
        """Test basic XOR metric calculation."""
        # Create simple test tensors
        pred = torch.tensor([[[[1, 0], [0, 1]]]], dtype=torch.float32)
        target = torch.tensor([[[[1, 1], [0, 0]]]], dtype=torch.float32)
        
        xor_score = calculate_xor_metric(pred, target)
        self.assertIsInstance(xor_score, float)
        self.assertGreaterEqual(xor_score, 0.0)
        self.assertLessEqual(xor_score, 1.0)
    
    def test_xor_metric_empty_masks(self):
        """Test XOR metric with empty masks."""
        # Both empty
        pred = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
        target = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
        
        xor_score = calculate_xor_metric(pred, target)
        self.assertEqual(xor_score, 0.0)
        
        # Only prediction empty
        pred = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
        target = torch.ones((1, 1, 2, 2), dtype=torch.float32)
        
        xor_score = calculate_xor_metric(pred, target)
        self.assertEqual(xor_score, 1.0)
    
    def test_hm_metric_basic(self):
        """Test basic HM metric calculation."""
        # Create simple test tensors
        pred = torch.tensor([[[[1, 0], [0, 1]]]], dtype=torch.float32)
        target = torch.tensor([[[[1, 1], [0, 0]]]], dtype=torch.float32)
        
        hm_score = calculate_hm_metric(pred, target)
        self.assertIsInstance(hm_score, float)
        self.assertGreaterEqual(hm_score, 0.0)
        self.assertLessEqual(hm_score, 1.0)
    
    def test_hm_metric_empty_masks(self):
        """Test HM metric with empty masks."""
        # Both empty
        pred = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
        target = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
        
        hm_score = calculate_hm_metric(pred, target)
        self.assertEqual(hm_score, 0.0)
    
    def test_metrics_consistency(self):
        """Test that metrics are consistent across different tensor shapes."""
        # Test with different shapes
        shapes = [(1, 1, 4, 4), (1, 1, 8, 8), (1, 1, 16, 16)]
        
        for shape in shapes:
            pred = torch.rand(shape) > 0.5
            target = torch.rand(shape) > 0.5
            
            xor_score = calculate_xor_metric(pred.float(), target.float())
            hm_score = calculate_hm_metric(pred.float(), target.float())
            
            self.assertIsInstance(xor_score, float)
            self.assertIsInstance(hm_score, float)
            self.assertGreaterEqual(xor_score, 0.0)
            self.assertLessEqual(xor_score, 1.0)
            self.assertGreaterEqual(hm_score, 0.0)
            self.assertLessEqual(hm_score, 1.0)


class TestValidationMetricsScripts(unittest.TestCase):
    """Test that validation metrics scripts can be imported."""
    
    def test_validate_metrics_script_exists(self):
        """Test that validate_metrics.py script exists."""
        try:
            script_path = Path(__file__).parent.parent / 'scripts' / 'validate_metrics.py'
            self.assertTrue(script_path.exists(), "validate_metrics.py script should exist")
        except Exception as e:
            self.fail(f"Failed to check validate_metrics.py: {e}")
    
    def test_validation_config_exists(self):
        """Test that validation metrics config file exists."""
        config_path = Path(__file__).parent.parent / 'configs' / 'validation_metrics_config.yaml'
        self.assertTrue(config_path.exists(), "validation_metrics_config.yaml should exist")
    
    def test_validation_example_exists(self):
        """Test that validation metrics example file exists."""
        example_path = Path(__file__).parent.parent / 'examples' / 'validation_metrics_example.py'
        self.assertTrue(example_path.exists(), "validation_metrics_example.py should exist")


if __name__ == '__main__':
    print("🧪 Running validation metrics tests...")
    print("=" * 50)
    
    # Run tests
    unittest.main(verbosity=2)
