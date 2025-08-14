"""
Training logic for crack segmentation models.

This module contains:
- Loss functions
- Evaluation metrics
- Training utilities
"""

from .losses import *
from .metrics import *

__all__ = [
    # Losses
    "BinaryDiceLoss",
    "CombinedLoss",
    "loss_ce",
    "loss_inter",
    "loss_intra",
    
    # Metrics
    "evaluate_metrics",
    "calculate_iou",
    "calculate_dice",
    "calculate_precision_recall",
    "calculate_xor_metric",
    "calculate_hm_metric",
]
