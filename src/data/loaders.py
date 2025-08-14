"""
Data loader utilities for crack segmentation.

This module contains:
- Functions to create DataLoaders
- Data splitting utilities
- Batch processing helpers
"""

import torch
from torch.utils.data import DataLoader, random_split
from typing import Tuple, Optional
from .datasets import get_dataset
from .transforms import get_transforms


def get_dataloader(dataset_name: str, root_dir: str, batch_size: int = 4,
                   image_size: int = 448, num_workers: int = 4,
                   train_split: float = 0.8, val_split: float = 0.1,
                   test_split: float = 0.1, shuffle: bool = True,
                   pin_memory: bool = True) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create DataLoaders for training, validation, and optionally testing.
    
    Args:
        dataset_name (str): Name of the dataset
        root_dir (str): Root directory path
        batch_size (int): Batch size for training
        image_size (int): Target image size
        num_workers (int): Number of worker processes
        train_split (float): Fraction of data for training
        val_split (float): Fraction of data for validation
        test_split (float): Fraction of data for testing
        shuffle (bool): Whether to shuffle training data
        pin_memory (bool): Whether to pin memory for GPU training
        
    Returns:
        Tuple[DataLoader, DataLoader, Optional[DataLoader]]: (train_loader, val_loader, test_loader)
    """
    # Validate splits
    total_split = train_split + val_split + test_split
    if abs(total_split - 1.0) > 1e-6:
        raise ValueError(f"Split fractions must sum to 1.0, got {total_split}")
    
    # Get transforms
    train_transforms = get_transforms(image_size, is_training=True)
    val_transforms = get_transforms(image_size, is_training=False)
    test_transforms = get_transforms(image_size, is_training=False)
    
    # Create full dataset
    full_dataset = get_dataset(dataset_name, root_dir, 'train', train_transforms)
    
    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    test_loader = None
    if test_size > 0:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )
    
    return train_loader, val_loader, test_loader


def create_dataloaders(dataset_name: str, root_dir: str, batch_size: int = 4,
                       image_size: int = 448, num_workers: int = 4,
                       train_split: float = 0.8, val_split: float = 0.1,
                       test_split: float = 0.1, **kwargs) -> dict:
    """
    Create all DataLoaders and return as a dictionary.
    
    Args:
        dataset_name (str): Name of the dataset
        root_dir (str): Root directory path
        batch_size (int): Batch size for training
        image_size (int): Target image size
        num_workers (int): Number of worker processes
        train_split (float): Fraction of data for training
        val_split (float): Fraction of data for validation
        test_split (float): Fraction of data for testing
        **kwargs: Additional arguments for get_dataloader
        
    Returns:
        dict: Dictionary containing all DataLoaders and dataset info
    """
    train_loader, val_loader, test_loader = get_dataloader(
        dataset_name=dataset_name,
        root_dir=root_dir,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=num_workers,
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        **kwargs
    )
    
    # Get dataset info
    dataset_info = {
        'name': dataset_name,
        'train_size': len(train_loader.dataset),
        'val_size': len(val_loader.dataset),
        'test_size': len(test_loader.dataset) if test_loader else 0,
        'num_classes': 1,  # Binary segmentation
        'image_size': image_size,
        'batch_size': batch_size
    }
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'dataset_info': dataset_info
    }


def get_simple_dataloader(dataset_name: str, root_dir: str, split: str = 'train',
                          batch_size: int = 4, image_size: int = 448,
                          num_workers: int = 4, is_training: bool = True,
                          **kwargs) -> DataLoader:
    """
    Create a simple DataLoader for a specific split.
    
    Args:
        dataset_name (str): Name of the dataset
        root_dir (str): Root directory path
        split (str): Dataset split ('train', 'val', 'test')
        batch_size (int): Batch size
        image_size (int): Target image size
        num_workers (int): Number of worker processes
        is_training (bool): Whether this is for training
        **kwargs: Additional arguments for DataLoader
        
    Returns:
        DataLoader: Configured DataLoader
    """
    # Get transforms
    transforms = get_transforms(image_size, is_training=is_training)
    
    # Create dataset
    dataset = get_dataset(dataset_name, root_dir, split, transforms)
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        num_workers=num_workers,
        drop_last=is_training,
        **kwargs
    )
    
    return dataloader


def get_batch_info(dataloader: DataLoader) -> dict:
    """
    Get information about a DataLoader batch.
    
    Args:
        dataloader (DataLoader): The DataLoader to inspect
        
    Returns:
        dict: Batch information
    """
    # Get a sample batch
    sample_batch = next(iter(dataloader))
    images, masks = sample_batch
    
    batch_info = {
        'batch_size': images.size(0),
        'image_channels': images.size(1),
        'image_height': images.size(2),
        'image_width': images.size(3),
        'mask_channels': masks.size(1),
        'mask_height': masks.size(2),
        'mask_width': masks.size(3),
        'image_dtype': images.dtype,
        'mask_dtype': masks.dtype,
        'dataset_size': len(dataloader.dataset),
        'num_batches': len(dataloader)
    }
    
    return batch_info
