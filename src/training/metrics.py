"""
Evaluation metrics for crack segmentation.

This module contains:
- IoU (Intersection over Union)
- Dice coefficient
- Precision, Recall, F1-Score
- Additional segmentation metrics
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Union
from sklearn.metrics import precision_score, recall_score, f1_score
import logging


def calculate_iou(pred: torch.Tensor, target: torch.Tensor, 
                 threshold: float = 0.5) -> float:
    """
    Calculate Intersection over Union (IoU).
    
    Args:
        pred (torch.Tensor): Predicted probabilities [B, 1, H, W]
        target (torch.Tensor): Ground truth masks [B, 1, H, W]
        threshold (float): Threshold for binary prediction
        
    Returns:
        float: IoU value
    """
    # Convert to binary predictions
    pred_binary = (pred > threshold).float()
    
    # Calculate intersection and union
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum() - intersection
    
    # Avoid division by zero
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    iou = intersection / union
    return iou.item()


def calculate_dice(pred: torch.Tensor, target: torch.Tensor, 
                  threshold: float = 0.5, smooth: float = 1e-6) -> float:
    """
    Calculate Dice coefficient.
    
    Args:
        pred (torch.Tensor): Predicted probabilities [B, 1, H, W]
        target (torch.Tensor): Ground truth masks [B, 1, H, W]
        threshold (float): Threshold for binary prediction
        smooth (float): Smoothing factor
        
    Returns:
        float: Dice coefficient value
    """
    # Convert to binary predictions
    pred_binary = (pred > threshold).float()
    
    # Calculate intersection and sum
    intersection = (pred_binary * target).sum()
    total = pred_binary.sum() + target.sum()
    
    # Avoid division by zero
    if total == 0:
        return 1.0 if intersection == 0 else 0.0
    
    dice = (2.0 * intersection + smooth) / (total + smooth)
    return dice.item()


def calculate_precision_recall(pred: torch.Tensor, target: torch.Tensor,
                              threshold: float = 0.5) -> Tuple[float, float]:
    """
    Calculate Precision and Recall.
    
    Args:
        pred (torch.Tensor): Predicted probabilities [B, 1, H, W]
        target (torch.Tensor): Ground truth masks [B, 1, H, W]
        threshold (float): Threshold for binary prediction
        
    Returns:
        Tuple[float, float]: (precision, recall) values
    """
    # Convert to binary predictions
    pred_binary = (pred > threshold).float()
    
    # Flatten tensors
    pred_flat = pred_binary.view(-1).cpu().numpy()
    target_flat = target.view(-1).cpu().numpy()
    
    # Ensure binary classification
    if len(np.unique(pred_flat)) > 2 or len(np.unique(target_flat)) > 2:
        # Convert to binary: any non-zero value becomes 1
        pred_flat = (pred_flat > 0).astype(int)
        target_flat = (target_flat > 0).astype(int)
    
    # Calculate precision and recall
    precision = precision_score(target_flat, pred_flat, zero_division=0, average='binary')
    recall = recall_score(target_flat, pred_flat, zero_division=0, average='binary')
    
    return precision, recall


def calculate_hm_score(pred: torch.Tensor, target: torch.Tensor,
                      threshold: float = 0.5) -> float:
    """
    Calculate HM (Hamming-like) score for segmentation masks.
    HM score = (union - intersection) / union
    
    Args:
        pred (torch.Tensor): Predicted probabilities [B, 1, H, W]
        target (torch.Tensor): Ground truth masks [B, 1, H, W]
        threshold (float): Threshold for binary prediction
        
    Returns:
        float: HM score value
    """
    # Convert to binary predictions
    pred_binary = (pred > threshold).float()
    
    # Convert to numpy for calculation
    pred_flat = pred_binary.view(-1).cpu().numpy()
    target_flat = target.view(-1).cpu().numpy()
    
    # Calculate intersection and union
    intersection = np.sum(pred_flat * target_flat)
    union = np.sum(np.logical_or(pred_flat, target_flat))
    
    # Avoid division by zero
    if union == 0:
        return 0.0
    
    hm_score = (union - intersection) / union
    return hm_score


def calculate_xor_score(pred: torch.Tensor, target: torch.Tensor,
                       threshold: float = 0.5) -> float:
    """
    Calculate XOR score for segmentation masks.
    XOR score = (union - intersection) / sum(ground_truth)
    
    Args:
        pred (torch.Tensor): Predicted probabilities [B, 1, H, W]
        target (torch.Tensor): Ground truth masks [B, 1, H, W]
        threshold (float): Threshold for binary prediction
        
    Returns:
        float: XOR score value
    """
    # Convert to binary predictions
    pred_binary = (pred > threshold).float()
    
    # Convert to numpy for calculation
    pred_flat = pred_binary.view(-1).cpu().numpy()
    target_flat = target.view(-1).cpu().numpy()
    
    # Calculate intersection, union, and ground truth sum
    intersection = np.sum(pred_flat * target_flat)
    union = np.sum(np.logical_or(pred_flat, target_flat))
    gt_sum = np.sum(target_flat)
    
    # Avoid division by zero
    if gt_sum == 0:
        return 0.0
    
    xor_score = (union - intersection) / gt_sum
    return xor_score


def dice_metric(A: np.ndarray, B: np.ndarray) -> float:
    """
    Calculate Dice coefficient between two numpy arrays.
    
    Args:
        A (np.ndarray): First binary array
        B (np.ndarray): Second binary array
        
    Returns:
        float: Dice coefficient
    """
    intersect = np.sum(A * B)
    fsum = np.sum(A)
    ssum = np.sum(B)
    
    if fsum + ssum == 0:
        return 1.0 if intersect == 0 else 0.0
    
    dice = (2 * intersect) / (fsum + ssum)
    return dice


def hm_metric(A: np.ndarray, B: np.ndarray) -> float:
    """
    Calculate HM (Hamming-like) metric between two numpy arrays.
    
    Args:
        A (np.ndarray): First binary array
        B (np.ndarray): Second binary array
        
    Returns:
        float: HM score
    """
    intersection = A * B
    union = np.logical_or(A, B)
    
    if np.sum(union) == 0:
        return 0.0
    
    hm_score = (np.sum(union) - np.sum(intersection)) / np.sum(union)
    return hm_score


def xor_metric(A: np.ndarray, GT: np.ndarray) -> float:
    """
    Calculate XOR metric between prediction and ground truth.
    
    Args:
        A (np.ndarray): Prediction binary array
        GT (np.ndarray): Ground truth binary array
        
    Returns:
        float: XOR score
    """
    intersection = A * GT
    union = np.logical_or(A, GT)
    
    if np.sum(GT) == 0:
        return 0.0
    
    xor_score = (np.sum(union) - np.sum(intersection)) / np.sum(GT)
    return xor_score


def calculate_f1_score(pred: torch.Tensor, target: torch.Tensor,
                      threshold: float = 0.5) -> float:
    """
    Calculate F1-Score.
    
    Args:
        pred (torch.Tensor): Predicted probabilities [B, 1, H, W]
        target (torch.Tensor): Ground truth masks [B, 1, H, W]
        threshold (float): Threshold for binary prediction
        
    Returns:
        float: F1-Score value
    """
    # Convert to binary predictions
    pred_binary = (pred > threshold).float()
    
    # Flatten tensors
    pred_flat = pred_binary.view(-1).cpu().numpy()
    target_flat = target.view(-1).cpu().numpy()
    
    # Ensure binary classification
    if len(np.unique(pred_flat)) > 2 or len(np.unique(target_flat)) > 2:
        # Convert to binary: any non-zero value becomes 1
        pred_flat = (pred_flat > 0).astype(int)
        target_flat = (target_flat > 0).astype(int)
    
    # Calculate F1-Score
    f1 = f1_score(target_flat, pred_flat, zero_division=0, average='binary')
    
    return f1


def calculate_accuracy(pred: torch.Tensor, target: torch.Tensor,
                      threshold: float = 0.5) -> float:
    """
    Calculate accuracy.
    
    Args:
        pred (torch.Tensor): Predicted probabilities [B, 1, H, W]
        target (torch.Tensor): Ground truth masks [B, 1, H, W]
        threshold (float): Threshold for binary prediction
        
    Returns:
        float: Accuracy value
    """
    # Convert to binary predictions
    pred_binary = (pred > threshold).float()
    
    # Calculate accuracy
    correct = (pred_binary == target).float().sum()
    total = target.numel()
    
    accuracy = correct / total
    return accuracy.item()


def calculate_hausdorff_distance(pred: torch.Tensor, target: torch.Tensor,
                                threshold: float = 0.5) -> float:
    """
    Calculate Hausdorff distance between prediction and target.
    
    Args:
        pred (torch.Tensor): Predicted probabilities [B, 1, H, W]
        target (torch.Tensor): Ground truth masks [B, 1, H, W]
        threshold (float): Threshold for binary prediction
        
    Returns:
        float: Hausdorff distance value
    """
    # Convert to binary predictions
    pred_binary = (pred > threshold).float()
    
    # Convert to numpy for scipy operations
    pred_np = pred_binary.squeeze().cpu().numpy()
    target_np = target.squeeze().cpu().numpy()
    
    # Ensure binary masks
    pred_binary_np = (pred_np > 0.5).astype(np.uint8)
    target_binary_np = (target_np > 0.5).astype(np.uint8)
    
    try:
        from scipy.spatial.distance import directed_hausdorff
        
        # Find coordinates of non-zero pixels
        pred_coords = np.column_stack(np.where(pred_binary_np > 0))
        target_coords = np.column_stack(np.where(target_binary_np > 0))
        
        if len(pred_coords) == 0 or len(target_coords) == 0:
            return 0.0
        
        # Calculate directed Hausdorff distances
        d1, _, _ = directed_hausdorff(pred_coords, target_coords)
        d2, _, _ = directed_hausdorff(target_coords, pred_coords)
        
        # Return the maximum of the two directed distances
        hausdorff = max(d1, d2)
        return float(hausdorff)
        
    except ImportError:
        # Fallback to simple distance calculation
        logging.warning("scipy not available, using fallback Hausdorff calculation")
        
        # Simple approximation using center of mass
        if np.sum(pred_binary_np) == 0 or np.sum(target_binary_np) == 0:
            return 0.0
        
        pred_center = np.mean(pred_coords, axis=0)
        target_center = np.mean(target_coords, axis=0)
        
        distance = np.linalg.norm(pred_center - target_center)
        return float(distance)


def calculate_xor_metric(pred: torch.Tensor, target: torch.Tensor,
                         threshold: float = 0.5) -> float:
    """
    Calculate XOR metric between prediction and target.
    
    Args:
        pred (torch.Tensor): Predicted probabilities [B, 1, H, W]
        target (torch.Tensor): Ground truth masks [B, 1, H, W]
        threshold (float): Threshold for binary prediction
        
    Returns:
        float: XOR metric value
    """
    # Convert to binary predictions
    pred_binary = (pred > threshold).float()
    
    # Convert to numpy
    pred_np = pred_binary.squeeze().cpu().numpy()
    target_np = target.squeeze().cpu().numpy()
    
    # Ensure binary masks
    pred_binary_np = (pred_np > 0.5).astype(np.uint8)
    target_binary_np = (target_np > 0.5).astype(np.uint8)
    
    # Calculate XOR metric
    intersection = pred_binary_np * target_binary_np
    union = np.logical_or(pred_binary_np, target_binary_np)
    
    if np.sum(target_binary_np) == 0:
        return 0.0 if np.sum(pred_binary_np) == 0 else 1.0
    
    xor_score = (np.sum(union) - np.sum(intersection)) / np.sum(target_binary_np)
    return float(xor_score)


def calculate_hm_metric(pred: torch.Tensor, target: torch.Tensor,
                        threshold: float = 0.5) -> float:
    """
    Calculate HM metric between prediction and target.
    
    Args:
        pred (torch.Tensor): Predicted probabilities [B, 1, H, W]
        target (torch.Tensor): Ground truth masks [B, 1, H, W]
        threshold (float): Threshold for binary prediction
        
    Returns:
        float: HM metric value
    """
    # Convert to binary predictions
    pred_binary = (pred > threshold).float()
    
    # Convert to numpy
    pred_np = pred_binary.squeeze().cpu().numpy()
    target_np = target.squeeze().cpu().numpy()
    
    # Ensure binary masks
    pred_binary_np = (pred_np > 0.5).astype(np.uint8)
    target_binary_np = (target_np > 0.5).astype(np.uint8)
    
    # Calculate HM metric
    intersection = pred_binary_np * target_binary_np
    union = np.logical_or(pred_binary_np, target_binary_np)
    
    if np.sum(union) == 0:
        return 0.0
    
    hm_score = (np.sum(union) - np.sum(intersection)) / np.sum(union)
    return float(hm_score)


def evaluate_metrics(pred: torch.Tensor, target: torch.Tensor,
                    num_classes: int = 1, threshold: float = 0.5) -> Dict[str, float]:
    """
    Evaluate all metrics for a batch of predictions.
    
    Args:
        pred (torch.Tensor): Predicted probabilities [B, 1, H, W]
        target (torch.Tensor): Ground truth masks [B, 1, H, W]
        num_classes (int): Number of classes (1 for binary)
        threshold (float): Threshold for binary prediction
        
    Returns:
        Dict[str, float]: Dictionary of metric values
    """
    metrics = {}
    
    # Basic metrics
    metrics['IoU'] = calculate_iou(pred, target, threshold)
    metrics['Dice'] = calculate_dice(pred, target, threshold)
    metrics['Accuracy'] = calculate_accuracy(pred, target, threshold)
    
    # Precision, Recall, F1
    precision, recall = calculate_precision_recall(pred, target, threshold)
    metrics['Precision'] = precision
    metrics['Recall'] = recall
    metrics['F1 Score'] = calculate_f1_score(pred, target, threshold)
    
    # Additional metrics
    try:
        metrics['Hausdorff'] = calculate_hausdorff_distance(pred, target, threshold)
    except:
        metrics['Hausdorff'] = 0.0
    
    # New metrics
    try:
        metrics['XOR Score'] = calculate_xor_score(pred, target, threshold)
        metrics['HM Score'] = calculate_hm_score(pred, target, threshold)
    except:
        metrics['XOR Score'] = 0.0
        metrics['HM Score'] = 0.0
    
    return metrics


def evaluate_batch_metrics(pred: torch.Tensor, target: torch.Tensor,
                          threshold: float = 0.5) -> Dict[str, torch.Tensor]:
    """
    Evaluate metrics for each sample in a batch.
    
    Args:
        pred (torch.Tensor): Predicted probabilities [B, 1, H, W]
        target (torch.Tensor): Ground truth masks [B, 1, H, W]
        threshold (float): Threshold for binary prediction
        
    Returns:
        Dict[str, torch.Tensor]: Dictionary of per-sample metric values
    """
    batch_size = pred.size(0)
    metrics = {
        'IoU': torch.zeros(batch_size),
        'Dice': torch.zeros(batch_size),
        'Precision': torch.zeros(batch_size),
        'Recall': torch.zeros(batch_size),
        'F1 Score': torch.zeros(batch_size)
    }
    
    for i in range(batch_size):
        pred_sample = pred[i:i+1]
        target_sample = target[i:i+1]
        
        sample_metrics = evaluate_metrics(pred_sample, target_sample, threshold=threshold)
        
        for metric_name, value in sample_metrics.items():
            if metric_name in metrics:
                metrics[metric_name][i] = value
    
    return metrics


def save_metrics(epoch, train_loss, train_metrics, val_loss, val_metrics, output_csv):
    """
    Save training and validation metrics to CSV file.
    
    Args:
        epoch (int): Current epoch number
        train_loss (float): Training loss
        train_metrics (list): List of training metrics [IoU, Dice, Precision, Recall, F1, HM, XOR]
        val_loss (float): Validation loss
        val_metrics (list): List of validation metrics [IoU, Dice, Precision, Recall, F1, HM, XOR]
        output_csv (str): Path to output CSV file
    """
    import csv
    import os
    
    # Create the directory if it doesn't exist
    output_dir = os.path.dirname(output_csv)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Convert tensors to scalar values using .item()
    train_metrics = [m.item() if isinstance(m, torch.Tensor) else m for m in train_metrics]
    val_metrics = [m.item() if isinstance(m, torch.Tensor) else m for m in val_metrics]

    # Check if the CSV file exists, if not, write the header
    file_exists = os.path.isfile(output_csv)
    with open(output_csv, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            # Write the header row - updated to include HM and XOR scores
            header = [
                'Epoch',
                'Train Loss', 'Train IoU', 'Train Dice Score', 'Train Precision', 'Train Recall', 'Train F1 score', 'Train HM Score', 'Train XOR Score',
                'Val Loss', 'Val IoU', 'Val Dice Score', 'Val Precision', 'Val Recall', 'Val F1 score', 'Val HM Score', 'Val XOR Score'
            ]
            writer.writerow(header)
        # Write the metrics row
        writer.writerow([epoch] + [train_loss] + train_metrics + [val_loss] + val_metrics)


def save_metrics_dict(metrics: Dict[str, float], filepath: str):
    """
    Save metrics dictionary to a JSON file.
    
    Args:
        metrics (Dict[str, float]): Dictionary of metrics
        filepath (str): Path to save the metrics
    """
    import json
    
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)


def load_metrics(filepath: str) -> Dict[str, float]:
    """
    Load metrics from a file.
    
    Args:
        filepath (str): Path to the metrics file
        
    Returns:
        Dict[str, float]: Loaded metrics
    """
    import json
    
    with open(filepath, 'r') as f:
        metrics = json.load(f)
    
    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "Evaluation Metrics"):
    """
    Print metrics in a formatted way.
    
    Args:
        metrics (Dict[str, float]): Dictionary of metrics
        title (str): Title for the metrics output
    """
    print(f"\n{title}")
    print("=" * len(title))
    
    for metric_name, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric_name:15}: {value:.4f}")
        else:
            print(f"{metric_name:15}: {value}")
    
    print()


def compare_metrics(metrics_list: List[Dict[str, float]], 
                   model_names: List[str]) -> Dict[str, List[float]]:
    """
    Compare metrics across multiple models.
    
    Args:
        metrics_list (List[Dict[str, float]]): List of metric dictionaries
        model_names (List[str]): List of model names
        
    Returns:
        Dict[str, List[float]]: Comparison table
    """
    if len(metrics_list) != len(model_names):
        raise ValueError("Number of metrics and model names must match")
    
    # Get all metric names
    all_metrics = set()
    for metrics in metrics_list:
        all_metrics.update(metrics.keys())
    
    # Create comparison table
    comparison = {}
    for metric in all_metrics:
        comparison[metric] = []
        for metrics in metrics_list:
            comparison[metric].append(metrics.get(metric, 0.0))
    
    return comparison


def print_comparison(comparison: Dict[str, List[float]], 
                    model_names: List[str]):
    """
    Print comparison table.
    
    Args:
        comparison (Dict[str, List[float]]): Comparison data
        model_names (List[str]): Model names
    """
    print("\nModel Comparison")
    print("=" * 50)
    
    # Print header
    print(f"{'Metric':<15}", end="")
    for name in model_names:
        print(f"{name:<12}", end="")
    print()
    
    print("-" * 50)
    
    # Print metrics
    for metric, values in comparison.items():
        print(f"{metric:<15}", end="")
        for value in values:
            print(f"{value:<12.4f}", end="")
        print()
    
    print()
