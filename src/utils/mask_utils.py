"""
Mask utility functions for crack segmentation.
"""

import torch
import numpy as np


def create_mask(prediction, ground_truth, restrict_to_gt=True):
    """
    Create a mask from prediction and ground truth.
    
    Args:
        prediction: Model prediction tensor
        ground_truth: Ground truth mask tensor
        restrict_to_gt: Whether to restrict prediction to ground truth area
        
    Returns:
        Binary mask tensor
    """
    if restrict_to_gt:
        # Only consider predictions within the ground truth area
        mask = (ground_truth > 0).float()
        prediction = prediction * mask
    
    # Convert to binary mask
    if prediction.dim() == 0:
        prediction = prediction.unsqueeze(0)
    
    # Handle different prediction formats
    if prediction.max() > 1:
        # Multi-class prediction, take argmax
        prediction = (prediction == prediction.max()).float()
    else:
        # Binary prediction, threshold at 0.5
        prediction = (prediction > 0.5).float()
    
    return prediction
