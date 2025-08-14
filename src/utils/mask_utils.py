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
        # Multi-class prediction, convert to binary
        # Find the most frequent class within the ground truth area
        if ground_truth.sum() > 0:
            # Get the most common class in the ground truth region
            gt_region = ground_truth > 0
            if gt_region.sum() > 0:
                pred_in_gt = prediction[gt_region]
                if pred_in_gt.numel() > 0:
                    most_common_class = torch.mode(pred_in_gt).values.item()
                    # Create binary mask: 1 for most common class, 0 otherwise
                    prediction = (prediction == most_common_class).float()
                else:
                    prediction = (prediction == prediction.max()).float()
            else:
                prediction = (prediction == prediction.max()).float()
        else:
            # If no ground truth, use the most common class overall
            prediction = (prediction == prediction.max()).float()
    else:
        # Binary prediction, threshold at 0.5
        prediction = (prediction > 0.5).float()
    
    return prediction
