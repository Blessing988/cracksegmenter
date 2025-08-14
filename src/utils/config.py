"""
Configuration management utilities for CrackSegmenter.

This module contains:
- Loading and saving YAML configuration files
- Configuration validation
- Default configuration generation
"""

import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        Dict[str, Any]: Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate configuration
        validate_config(config)
        
        return config
    
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML in configuration file: {e}")
    except Exception as e:
        raise Exception(f"Error loading configuration: {e}")


def save_config(config: Dict[str, Any], config_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        config_path (str): Path to save configuration file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def validate_config(config: Dict[str, Any]):
    """
    Validate configuration dictionary.
    
    Args:
        config (Dict[str, Any]): Configuration to validate
        
    Raises:
        ValueError: If configuration is invalid
    """
    required_sections = ['model', 'training', 'data', 'utils']
    
    # Check required sections
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Validate model section
    model_config = config['model']
    required_model_keys = ['num_classes', 'backbone', 'architecture', 'baseline']
    for key in required_model_keys:
        if key not in model_config:
            raise ValueError(f"Missing required model configuration: {key}")
    
    # Validate training section
    training_config = config['training']
    required_training_keys = ['batch_size', 'num_epochs', 'learning_rate']
    for key in required_training_keys:
        if key not in training_config:
            raise ValueError(f"Missing required training configuration: {key}")
    
    # Validate data section
    data_config = config['data']
    required_data_keys = ['image_size', 'root_dir', 'dataset_name']
    for key in required_data_keys:
        if key not in data_config:
            raise ValueError(f"Missing required data configuration: {key}")
    
    # Validate utils section
    utils_config = config['utils']
    if 'save_dir' not in utils_config:
        raise ValueError("Missing required utils configuration: save_dir")


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration.
    
    Returns:
        Dict[str, Any]: Default configuration dictionary
    """
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
        'training': {
            'batch_size': 4,
            'num_epochs': 500,
            'learning_rate': 0.0001,
            'weight_decay': 0.00001,
            'early_stopping_patience': 50
        },
        'data': {
            'image_size': 448,
            'root_dir': '/path/to/datasets',
            'dataset_name': 'cracktree200',
            'num_workers': 4,
            'mask_ext': '.png'
        },
        'utils': {
            'save_dir': './trained_models'
        }
    }


def create_config_file(config_path: str, config: Optional[Dict[str, Any]] = None):
    """
    Create a new configuration file.
    
    Args:
        config_path (str): Path to create configuration file
        config (Dict[str, Any], optional): Configuration to use, defaults to default config
    """
    if config is None:
        config = get_default_config()
    
    save_config(config, config_path)
    print(f"Configuration file created: {config_path}")


def update_config(config: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update configuration with new values.
    
    Args:
        config (Dict[str, Any]): Original configuration
        updates (Dict[str, Any]): Updates to apply
        
    Returns:
        Dict[str, Any]: Updated configuration
    """
    updated_config = config.copy()
    
    def deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = deep_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d
    
    return deep_update(updated_config, updates)


def merge_configs(configs: list) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries.
    
    Args:
        configs (list): List of configuration dictionaries
        
    Returns:
        Dict[str, Any]: Merged configuration
    """
    if not configs:
        return {}
    
    merged = configs[0].copy()
    
    for config in configs[1:]:
        merged = update_config(merged, config)
    
    return merged


def print_config(config: Dict[str, Any], title: str = "Configuration"):
    """
    Print configuration in a formatted way.
    
    Args:
        config (Dict[str, Any]): Configuration to print
        title (str): Title for the output
    """
    print(f"\n{title}")
    print("=" * len(title))
    
    def print_section(data, indent=0):
        for key, value in data.items():
            if isinstance(value, dict):
                print(" " * indent + f"{key}:")
                print_section(value, indent + 2)
            else:
                print(" " * indent + f"{key}: {value}")
    
    print_section(config)
    print()


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get configuration value using dot notation.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        key_path (str): Dot-separated key path (e.g., 'model.architecture')
        default (Any): Default value if key not found
        
    Returns:
        Any: Configuration value or default
    """
    keys = key_path.split('.')
    value = config
    
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default


def set_config_value(config: Dict[str, Any], key_path: str, value: Any):
    """
    Set configuration value using dot notation.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        key_path (str): Dot-separated key path (e.g., 'model.architecture')
        value (Any): Value to set
    """
    keys = key_path.split('.')
    current = config
    
    # Navigate to the parent of the target key
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    # Set the value
    current[keys[-1]] = value
