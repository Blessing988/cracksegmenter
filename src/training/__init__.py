"""
Training logic for crack segmentation models.

This module contains:
- Main training loop and trainer class
- Loss functions
- Evaluation metrics
- Training utilities
"""

from .trainer import *
from .losses import *
from .metrics import *

__all__ = [
    # Trainer
    "CrackSegmenterTrainer",
    "train_one_epoch",
    "validate",
    
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
]
