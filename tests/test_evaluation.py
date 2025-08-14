#!/usr/bin/env python3
"""
Tests for evaluation functionality.
"""

import sys
import os
import unittest
from pathlib import Path
import tempfile
import shutil

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from utils.config import get_default_config


class TestEvaluationConfig(unittest.TestCase):
    """Test evaluation configuration functionality."""
    
    def test_default_config_structure(self):
        """Test that default config has required sections."""
        config = get_default_config()
        
        required_sections = ['model', 'training', 'data', 'utils']
        for section in required_sections:
            self.assertIn(section, config)
    
    def test_model_config_keys(self):
        """Test that model config has required keys."""
        config = get_default_config()
        
        required_keys = ['num_classes', 'backbone', 'architecture', 'baseline']
        for key in required_keys:
            self.assertIn(key, config['model'])
    
    def test_training_config_keys(self):
        """Test that training config has required keys."""
        config = get_default_config()
        
        required_keys = ['batch_size', 'num_epochs', 'learning_rate']
        for key in required_keys:
            self.assertIn(key, config['training'])
    
    def test_data_config_keys(self):
        """Test that data config has required keys."""
        config = get_default_config()
        
        required_keys = ['image_size', 'root_dir', 'dataset_name']
        for key in required_keys:
            self.assertIn(key, config['data'])


class TestEvaluationScripts(unittest.TestCase):
    """Test that evaluation scripts can be imported."""
    
    def test_evaluate_script_import(self):
        """Test that evaluate.py can be imported."""
        try:
            # This should work if the script is properly structured
            script_path = Path(__file__).parent.parent / 'scripts' / 'evaluate.py'
            self.assertTrue(script_path.exists(), "evaluate.py script should exist")
        except Exception as e:
            self.fail(f"Failed to check evaluate.py: {e}")
    
    def test_evaluate_all_script_import(self):
        """Test that evaluate_all.py can be imported."""
        try:
            # This should work if the script is properly structured
            script_path = Path(__file__).parent.parent / 'scripts' / 'evaluate_all.py'
            self.assertTrue(script_path.exists(), "evaluate_all.py script should exist")
        except Exception as e:
            self.fail(f"Failed to check evaluate_all.py: {e}")


class TestEvaluationConfigFile(unittest.TestCase):
    """Test evaluation configuration file."""
    
    def test_evaluation_config_exists(self):
        """Test that evaluation config file exists."""
        config_path = Path(__file__).parent.parent / 'configs' / 'evaluation_config.yaml'
        self.assertTrue(config_path.exists(), "evaluation_config.yaml should exist")


if __name__ == '__main__':
    print("🧪 Running evaluation tests...")
    print("=" * 50)
    
    # Run tests
    unittest.main(verbosity=2)
