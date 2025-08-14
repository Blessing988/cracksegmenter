"""
Data augmentation and transformation functions for crack segmentation.

This module contains:
- Training transforms with augmentation
- Validation transforms (minimal)
- Custom transform utilities
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from typing import Tuple


def get_transforms(image_size: int = 448, is_training: bool = True):
    """
    Get data transforms for training or validation.
    
    Args:
        image_size (int): Target image size
        is_training (bool): Whether to apply training augmentations
        
    Returns:
        A.Compose: Albumentations transform pipeline
    """
    if is_training:
        return get_training_transforms(image_size)
    else:
        return get_validation_transforms(image_size)


def get_training_transforms(image_size: int = 448):
    """
    Get training transforms with data augmentation.
    
    Args:
        image_size (int): Target image size
        
    Returns:
        A.Compose: Training transform pipeline
    """
    train_transforms = A.Compose([
        # Spatial-level transforms (applied to both image and mask)
        A.Resize(height=image_size, width=image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        
        # Image-only transforms
        A.GaussNoise(p=0.2),
        A.ColorJitter(
            brightness=0.2, 
            contrast=0.2, 
            saturation=0.2, 
            hue=0.1, 
            p=0.5
        ),
        
        # Normalize the image using ResNet's mean and std
        A.Normalize(
            mean=(0.485, 0.456, 0.406), 
            std=(0.229, 0.224, 0.225)
        ),
        
        # Convert image and mask to PyTorch tensors
        ToTensorV2()
    ], additional_targets={"mask": "mask"})
    
    return train_transforms


def get_validation_transforms(image_size: int = 448):
    """
    Get validation transforms (minimal augmentation).
    
    Args:
        image_size (int): Target image size
        
    Returns:
        A.Compose: Validation transform pipeline
    """
    val_transforms = A.Compose([
        # Resize images and masks
        A.Resize(height=image_size, width=image_size),
        
        # Normalize the image using ResNet's mean and std
        A.Normalize(
            mean=(0.485, 0.456, 0.406), 
            std=(0.229, 0.224, 0.225)
        ),
        
        # Convert image and mask to PyTorch tensors
        ToTensorV2()
    ], additional_targets={"mask": "mask"})
    
    return val_transforms


def get_augmentation(image_size: int = 448, augmentation_level: str = 'medium'):
    """
    Get augmentation transforms with different intensity levels.
    
    Args:
        image_size (int): Target image size
        augmentation_level (str): Augmentation intensity ('light', 'medium', 'heavy')
        
    Returns:
        A.Compose: Augmentation transform pipeline
    """
    if augmentation_level == 'light':
        return get_light_augmentation(image_size)
    elif augmentation_level == 'medium':
        return get_medium_augmentation(image_size)
    elif augmentation_level == 'heavy':
        return get_heavy_augmentation(image_size)
    else:
        raise ValueError(f"Unsupported augmentation level: {augmentation_level}")


def get_light_augmentation(image_size: int = 448):
    """Light augmentation for sensitive data."""
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.HorizontalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        A.GaussNoise(p=0.1),
        A.Normalize(
            mean=(0.485, 0.456, 0.406), 
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2()
    ], additional_targets={"mask": "mask"})


def get_medium_augmentation(image_size: int = 448):
    """Medium augmentation (default training transforms)."""
    return get_training_transforms(image_size)


def get_heavy_augmentation(image_size: int = 448):
    """Heavy augmentation for robust training."""
    heavy_transforms = A.Compose([
        # Spatial transforms
        A.Resize(height=image_size, width=image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=45, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        A.RandomScale(scale_limit=0.2, p=0.5),
        A.RandomTranslate(percent=0.2, p=0.5),
        
        # Elastic transforms
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
        
        # Noise and blur
        A.GaussNoise(p=0.3),
        A.MotionBlur(blur_limit=7, p=0.3),
        A.MedianBlur(blur_limit=5, p=0.2),
        
        # Color transforms
        A.ColorJitter(
            brightness=0.3, 
            contrast=0.3, 
            saturation=0.3, 
            hue=0.2, 
            p=0.5
        ),
        A.RandomBrightnessContrast(p=0.5),
        A.RandomGamma(p=0.3),
        
        # Normalization
        A.Normalize(
            mean=(0.485, 0.456, 0.406), 
            std=(0.229, 0.224, 0.225)
        ),
        
        ToTensorV2()
    ], additional_targets={"mask": "mask"})
    
    return heavy_transforms


def get_test_transforms(image_size: int = 448):
    """
    Get test transforms (no augmentation, just normalization).
    
    Args:
        image_size (int): Target image size
        
    Returns:
        A.Compose: Test transform pipeline
    """
    test_transforms = A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(
            mean=(0.485, 0.456, 0.406), 
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2()
    ])
    
    return test_transforms


def get_inference_transforms(image_size: int = 448):
    """
    Get transforms for inference (same as test transforms).
    
    Args:
        image_size (int): Target image size
        
    Returns:
        A.Compose: Inference transform pipeline
    """
    return get_test_transforms(image_size)
