"""
Loss functions for crack segmentation training.

This module contains:
- Binary Dice Loss
- Combined Loss (BCE + Dice)
- Custom loss functions for MSFormer
- Loss utilities and helpers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class BinaryDiceLoss(nn.Module):
    """Binary Dice Loss for crack segmentation."""
    
    def __init__(self, smooth: float = 1e-6, square_denominator: bool = False):
        """
        Initialize Binary Dice Loss.
        
        Args:
            smooth (float): Smoothing factor to avoid division by zero
            square_denominator (bool): Whether to square the denominator
        """
        super().__init__()
        self.smooth = smooth
        self.square_denominator = square_denominator
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Binary Dice Loss.
        
        Args:
            input (torch.Tensor): Predicted logits [B, 1, H, W]
            target (torch.Tensor): Ground truth masks [B, 1, H, W]
            
        Returns:
            torch.Tensor: Dice loss value
        """
        # Apply sigmoid to get probabilities
        input = torch.sigmoid(input)
        
        # Flatten tensors
        input = input.view(-1)
        target = target.view(-1)
        
        # Calculate intersection and union
        intersection = (input * target).sum()
        
        if self.square_denominator:
            denominator = (input * input).sum() + (target * target).sum()
        else:
            denominator = input.sum() + target.sum()
        
        # Calculate Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        
        # Return loss (1 - Dice)
        return 1.0 - dice


class CombinedLoss(nn.Module):
    """Combined loss combining BCE and Dice loss."""
    
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5,
                 smooth: float = 1e-6):
        """
        Initialize Combined Loss.
        
        Args:
            bce_weight (float): Weight for BCE loss
            dice_weight (float): Weight for Dice loss
            smooth (float): Smoothing factor for Dice loss
        """
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = BinaryDiceLoss(smooth=smooth)
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            input (torch.Tensor): Predicted logits [B, 1, H, W]
            target (torch.Tensor): Ground truth masks [B, 1, H, W]
            
        Returns:
            torch.Tensor: Combined loss value
        """
        bce_loss = self.bce_loss(input, target)
        dice_loss = self.dice_loss(input, target)
        
        combined_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        
        return combined_loss


def loss_ce(output, target):
    """Cross-entropy loss for multi-class segmentation."""
    return F.cross_entropy(output, target)


def loss_inter(context_f, context_s, weight):
    """Inter-scale loss between fine and small contexts."""
    return F.mse_loss(context_f, context_s) * weight


def loss_intra(att_score, identity):
    """Intra-scale loss for attention scores."""
    return F.mse_loss(att_score, identity)





class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, 
                 reduction: str = 'mean'):
        """
        Initialize Focal Loss.
        
        Args:
            alpha (float): Weighting factor for rare class
            gamma (float): Focusing parameter
            reduction (str): Reduction method ('none', 'mean', 'sum')
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Focal Loss.
        
        Args:
            input (torch.Tensor): Predicted logits
            target (torch.Tensor): Ground truth masks
            
        Returns:
            torch.Tensor: Focal loss value
        """
        # Apply sigmoid
        input = torch.sigmoid(input)
        
        # Calculate BCE loss
        bce_loss = F.binary_cross_entropy(input, target, reduction='none')
        
        # Calculate focal weight
        pt = input * target + (1 - input) * (1 - target)
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha weighting
        alpha_weight = self.alpha * target + (1 - self.alpha) * (1 - target)
        
        # Calculate focal loss
        focal_loss = alpha_weight * focal_weight * bce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class IoULoss(nn.Module):
    """IoU-based loss for segmentation."""
    
    def __init__(self, smooth: float = 1e-6):
        """
        Initialize IoU Loss.
        
        Args:
            smooth (float): Smoothing factor
        """
        super().__init__()
        self.smooth = smooth
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute IoU Loss.
        
        Args:
            input (torch.Tensor): Predicted logits
            target (torch.Tensor): Ground truth masks
            
        Returns:
            torch.Tensor: IoU loss value
        """
        # Apply sigmoid
        input = torch.sigmoid(input)
        
        # Flatten tensors
        input = input.view(-1)
        target = target.view(-1)
        
        # Calculate intersection and union
        intersection = (input * target).sum()
        union = input.sum() + target.sum() - intersection
        
        # Calculate IoU
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        # Return loss (1 - IoU)
        return 1.0 - iou


def get_loss_function(loss_name: str = 'combined', **kwargs):
    """
    Factory function to get loss function by name.
    
    Args:
        loss_name (str): Name of the loss function
        **kwargs: Additional arguments for the loss function
        
    Returns:
        nn.Module: Configured loss function
        
    Raises:
        ValueError: If loss name is not supported
    """
    loss_map = {
        'bce': nn.BCEWithLogitsLoss,
        'dice': BinaryDiceLoss,
        'combined': CombinedLoss,
        'focal': FocalLoss,
        'iou': IoULoss,
    }
    
    loss_name = loss_name.lower()
    
    if loss_name not in loss_map:
        raise ValueError(f"Unsupported loss function: {loss_name}. "
                       f"Supported: {list(loss_map.keys())}")
    
    return loss_map[loss_name](**kwargs)
