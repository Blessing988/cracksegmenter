"""
Dataset classes for crack segmentation.

This module contains:
- Base CrackDataset class
- Specific dataset implementations (CrackTree200, CFD, Forest, GAPS)
- Dataset utilities and helpers
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
from typing import Tuple, Optional, List


class CrackDataset(Dataset):
    """Base dataset class for crack segmentation."""
    
    def __init__(self, root_dir: str, split: str = 'train', 
                 transform=None, mask_ext: str = '.png', dataset_subdir=None):
        """
        Initialize the dataset.
        
        Args:
            root_dir (str): Root directory containing images and masks
            split (str): Dataset split ('train', 'val', 'test')
            transform: Data augmentation transforms
            mask_ext (str): Extension for mask files
            dataset_subdir (str): Subdirectory name for the dataset (e.g., 'CRACK500')
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.mask_ext = mask_ext
        self.dataset_subdir = dataset_subdir
        
        # Setup paths - handle both flat and subdirectory structures
        if dataset_subdir:
            # Subdirectory structure: root_dir/dataset_name/split/images
            self.images_dir = os.path.join(root_dir, dataset_subdir, split, 'images')
            self.masks_dir = os.path.join(root_dir, dataset_subdir, split, 'masks')
        else:
            # Flat structure: root_dir/split/images
            self.images_dir = os.path.join(root_dir, split, 'images')
            self.masks_dir = os.path.join(root_dir, split, 'masks')
        
        # Get file list
        self.image_files = self._get_image_files()
        
        if len(self.image_files) == 0:
            raise RuntimeError(f"No images found in {self.images_dir}")
    
    def _get_image_files(self) -> List[str]:
        """Get list of image files in the split directory."""
        if not os.path.exists(self.images_dir):
            return []
        
        # Get all image files
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_files.extend([
                f for f in os.listdir(self.images_dir) 
                if f.lower().endswith(ext)
            ])
        
        return sorted(image_files)
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx (int): Sample index
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (image, mask) pair
        """
        # Load image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        
        # Load mask (same name with different extension)
        mask_name = os.path.splitext(img_name)[0] + self.mask_ext
        mask_path = os.path.join(self.masks_dir, mask_name)
        
        # Load image and mask
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        # Convert to numpy arrays
        image = np.array(image)
        mask = np.array(mask)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        else:
            # Convert to tensors if no transforms
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0).float() / 255.0
        
        return image, mask


class CrackTree200Dataset(CrackDataset):
    """CrackTree200 dataset for pavement crack detection."""
    
    def __init__(self, root_dir: str, split: str = 'train', transform=None):
        super().__init__(root_dir, split, transform, mask_ext='.png')
        
        # CrackTree200 specific setup
        self.dataset_name = 'cracktree200'
        
        # Validate dataset structure
        self._validate_dataset()
    
    def _validate_dataset(self):
        """Validate CrackTree200 dataset structure."""
        expected_dirs = [
            os.path.join(self.root_dir, self.split, 'images'),
            os.path.join(self.root_dir, self.split, 'masks')
        ]
        
        for dir_path in expected_dirs:
            if not os.path.exists(dir_path):
                raise RuntimeError(f"Directory not found: {dir_path}")


class CFDDataset(CrackDataset):
    """CFD (Concrete Fracture Detection) dataset."""
    
    def __init__(self, root_dir: str, split: str = 'train', transform=None):
        super().__init__(root_dir, split, transform, mask_ext='.png', dataset_subdir='CFD')
        
        self.dataset_name = 'cfd'
        
        # CFD specific validation
        self._validate_dataset()
    
    def _validate_dataset(self):
        """Validate CFD dataset structure."""
        # CFD might have different structure
        if not os.path.exists(self.images_dir):
            raise RuntimeError(f"Images directory not found: {self.images_dir}")


class ForestDataset(CrackDataset):
    """Forest crack dataset."""
    
    def __init__(self, root_dir: str, split: str = 'train', transform=None):
        super().__init__(root_dir, split, transform, mask_ext='.png')
        
        self.dataset_name = 'forest'
        
        # Forest specific validation
        self._validate_dataset()
    
    def _validate_dataset(self):
        """Validate Forest dataset structure."""
        if not os.path.exists(self.images_dir):
            raise RuntimeError(f"Images directory not found: {self.images_dir}")


class GAPSDataset(CrackDataset):
    """GAPS dataset with 384x384 resolution."""
    
    def __init__(self, root_dir: str, split: str = 'train', transform=None):
        super().__init__(root_dir, split, transform, mask_ext='.png', dataset_subdir='GAPS384')
        
        self.dataset_name = 'gaps_384'
        self.target_size = (384, 384)
        
        # GAPS specific validation
        self._validate_dataset()
    
    def _validate_dataset(self):
        """Validate GAPS dataset structure."""
        if not os.path.exists(self.images_dir):
            raise RuntimeError(f"Images directory not found: {self.images_dir}")


class CRACK500Dataset(CrackDataset):
    """CRACK500 dataset."""
    
    def __init__(self, root_dir: str, split: str = 'train', transform=None):
        super().__init__(root_dir, split, transform, mask_ext='.png', dataset_subdir='CRACK500')
        
        self.dataset_name = 'crack500'
        
        # CRACK500 specific validation
        self._validate_dataset()
    
    def _validate_dataset(self):
        """Validate CRACK500 dataset structure."""
        if not os.path.exists(self.images_dir):
            raise RuntimeError(f"Images directory not found: {self.images_dir}")


class VolkerDataset(CrackDataset):
    """Volker dataset."""
    
    def __init__(self, root_dir: str, split: str = 'train', transform=None):
        super().__init__(root_dir, split, transform, mask_ext='.png', dataset_subdir='Volker')
        
        self.dataset_name = 'volker'
        
        # Volker specific validation
        self._validate_dataset()
    
    def _validate_dataset(self):
        """Validate Volker dataset structure."""
        if not os.path.exists(self.images_dir):
            raise RuntimeError(f"Images directory not found: {self.images_dir}")


class SylvieDataset(CrackDataset):
    """Sylvie dataset."""
    
    def __init__(self, root_dir: str, split: str = 'train', transform=None):
        super().__init__(root_dir, split, transform, mask_ext='.png', dataset_subdir='Sylvie')
        
        self.dataset_name = 'sylvie'
        
        # Sylvie specific validation
        self._validate_dataset()
    
    def _validate_dataset(self):
        """Validate Sylvie dataset structure."""
        if not os.path.exists(self.images_dir):
            raise RuntimeError(f"Images directory not found: {self.images_dir}")


class RissbilderDataset(CrackDataset):
    """Rissbilder dataset."""
    
    def __init__(self, root_dir: str, split: str = 'train', transform=None):
        super().__init__(root_dir, split, transform, mask_ext='.png', dataset_subdir='Rissbilder')
        
        self.dataset_name = 'rissbilder'
        
        # Rissbilder specific validation
        self._validate_dataset()
    
    def _validate_dataset(self):
        """Validate Rissbilder dataset structure."""
        if not os.path.exists(self.images_dir):
            raise RuntimeError(f"Images directory not found: {self.images_dir}")


class EugenMillerDataset(CrackDataset):
    """Eugen Miller dataset."""
    
    def __init__(self, root_dir: str, split: str = 'train', transform=None):
        super().__init__(root_dir, split, transform, mask_ext='.png', dataset_subdir='Eugen_Miller')
        
        self.dataset_name = 'eugen_miller'
        
        # Eugen Miller specific validation
        self._validate_dataset()
    
    def _validate_dataset(self):
        """Validate Eugen Miller dataset structure."""
        if not os.path.exists(self.images_dir):
            raise RuntimeError(f"Images directory not found: {self.images_dir}")


class DeepCrackDataset(CrackDataset):
    """DeepCrack dataset."""
    
    def __init__(self, root_dir: str, split: str = 'train', transform=None):
        super().__init__(root_dir, split, transform, mask_ext='.png', dataset_subdir='DeepCrack')
        
        self.dataset_name = 'deepcrack'
        
        # DeepCrack specific validation
        self._validate_dataset()
    
    def _validate_dataset(self):
        """Validate DeepCrack dataset structure."""
        if not os.path.exists(self.images_dir):
            raise RuntimeError(f"Images directory not found: {self.images_dir}")


def get_dataset(dataset_name: str, root_dir: str, split: str = 'train', 
                transform=None) -> CrackDataset:
    """
    Factory function to get dataset by name.
    
    Args:
        dataset_name (str): Name of the dataset
        root_dir (str): Root directory path
        split (str): Dataset split
        transform: Data augmentation transforms
        
    Returns:
        CrackDataset: Configured dataset
        
    Raises:
        ValueError: If dataset name is not supported
    """
    dataset_map = {
        'cracktree200': CrackTree200Dataset,
        'cfd': CFDDataset,
        'forest': ForestDataset,
        'gaps_384': GAPSDataset,
        'crack500': CRACK500Dataset,
        'volker': VolkerDataset,
        'sylvie': SylvieDataset,
        'rissbilder': RissbilderDataset,
        'eugen_miller': EugenMillerDataset,
        'deepcrack': DeepCrackDataset,
    }
    
    dataset_name = dataset_name.lower()
    
    if dataset_name not in dataset_map:
        raise ValueError(f"Unsupported dataset: {dataset_name}. "
                       f"Supported: {list(dataset_map.keys())}")
    
    return dataset_map[dataset_name](root_dir, split, transform)
