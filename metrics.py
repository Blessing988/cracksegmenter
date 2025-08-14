# import torch
# import numpy as np
# import os
# import csv


# def calculate_iou(pred_mask, true_mask, num_classes):
#     ious = []
#     pred_mask = pred_mask.view(-1)
#     true_mask = true_mask.view(-1)
#     if num_classes == 1:
#         pred_inds = (pred_mask == 1)
#         true_inds = (true_mask == 1)
#         intersection = (pred_inds & true_inds).sum().float()
#         union = (pred_inds | true_inds).sum().float()
#         if union == 0:
#             ious.append(float('nan'))  # Ignore if there is no ground truth
#         else:
#             ious.append((intersection / union).item())
            
#     else:
#         for cls in range(num_classes):
#             pred_inds = (pred_mask == cls)
#             true_inds = (true_mask == cls)
#             intersection = (pred_inds & true_inds).sum().float()
#             union = (pred_inds | true_inds).sum().float()
#             if union == 0:
#                 ious.append(float('nan'))  # Ignore if there is no ground truth
#             else:
#                 ious.append((intersection / union).item())
                
#     return ious


# def calculate_dice(pred_mask, true_mask, num_classes):
#     dices = []
#     pred_mask = pred_mask.view(-1)
#     true_mask = true_mask.view(-1)
#     if num_classes == 1:
#         pred_inds = (pred_mask == 1)
#         true_inds = (true_mask == 1)
#         intersection = (pred_inds & true_inds).sum().float()
#         total = pred_inds.sum() + true_inds.sum()
#         if total == 0:
#             dices.append(float('nan'))
#         else:
#             dices.append((2 * intersection / total).item())
    
#     else:
#         for cls in range(num_classes):
#             pred_inds = (pred_mask == cls)
#             true_inds = (true_mask == cls)
#             intersection = (pred_inds & true_inds).sum().float()
#             total = pred_inds.sum() + true_inds.sum()
#             if total == 0:
#                 dices.append(float('nan'))
#             else:
#                 dices.append((2 * intersection / total).item())
#     return dices


# def calculate_precision_recall(pred_mask, true_mask, num_classes):
#     precisions = []
#     recalls = []
#     pred_mask = pred_mask.view(-1)
#     true_mask = true_mask.view(-1)
    
#     if num_classes == 1 :
#         pred_inds = (pred_mask == 1)
#         true_inds = (true_mask == 1)
#         true_positive = (pred_inds & true_inds).sum().float()
#         predicted_positive = pred_inds.sum().float()
#         actual_positive = true_inds.sum().float()
#         if predicted_positive == 0:
#             precisions.append(float('nan'))
#         else:
#             precisions.append((true_positive / predicted_positive).item())
#         if actual_positive == 0:
#             recalls.append(float('nan'))
#         else:
#             recalls.append((true_positive / actual_positive).item())
            
#     else:
#         for cls in range(num_classes):
#             pred_inds = (pred_mask == cls)
#             true_inds = (true_mask == cls)
#             true_positive = (pred_inds & true_inds).sum().float()
#             predicted_positive = pred_inds.sum().float()
#             actual_positive = true_inds.sum().float()
#             if predicted_positive == 0:
#                 precisions.append(float('nan'))
#             else:
#                 precisions.append((true_positive / predicted_positive).item())
#             if actual_positive == 0:
#                 recalls.append(float('nan'))
#             else:
#                 recalls.append((true_positive / actual_positive).item())
                
#     return precisions, recalls


# def calculate_f1_score(precisions, recalls):
#     f1_scores = []
#     for p, r in zip(precisions, recalls):
#         if p == 0 or r == 0 or p != p or r != r:  # Check for zero or NaN
#             f1_scores.append(float('nan'))
#         else:
#             f1_scores.append(2 * p * r / (p + r))
#     return f1_scores


# def evaluate_metrics(pred_mask, true_mask, num_classes):
#     ious = calculate_iou(pred_mask, true_mask, num_classes)
#     dices = calculate_dice(pred_mask, true_mask, num_classes)
#     precisions, recalls = calculate_precision_recall(pred_mask, true_mask, num_classes)
#     f1_scores = calculate_f1_score(precisions, recalls)

#     metrics = {
#         'IoU': np.nanmean(ious) if len(ious) > 0 else np.nan,
#         'Dice': np.nanmean(dices) if len(dices) > 0 else np.nan,
#         'Precision': np.nanmean(precisions) if len(precisions) > 0 else np.nan,
#         'Recall': np.nanmean(recalls) if len(recalls) > 0 else np.nan,
#         'F1 Score': np.nanmean(f1_scores) if len(f1_scores) > 0 else np.nan
#     }

#     return metrics


# def save_metrics(epoch, train_loss, train_metrics, val_loss, val_metrics, output_csv):
#     # Create the directory if it doesn't exist
#     output_dir = os.path.dirname(output_csv)
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     # Convert tensors to scalar values using .item()
#     train_metrics = [m.item() if isinstance(m, torch.Tensor) else m for m in train_metrics]
#     val_metrics = [m.item() if isinstance(m, torch.Tensor) else m for m in val_metrics]

#     # Check if the CSV file exists, if not, write the header
#     file_exists = os.path.isfile(output_csv)
#     with open(output_csv, 'a', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         if not file_exists:
#             # Write the header row
#             header = [
#                 'Epoch',
#                 'Train Loss', 'Train IoU', 'Train Dice Score', 'Train Precision', 'Train Recall', 'Train F1 score',
#                 'Val Loss', 'Val IoU', 'Val Dice Score', 'Val Precision', 'Val Recall', 'Val F1 score'
#             ]
#             writer.writerow(header)
#         # Write the metrics row
#         writer.writerow([epoch] + [train_loss] +  train_metrics + [val_loss] + val_metrics)


import torch
import numpy as np
import os
import csv


def calculate_iou(pred_mask, true_mask, num_classes):
    ious = []
    pred_mask = pred_mask.view(-1)
    true_mask = true_mask.view(-1)
    if num_classes == 1:
        pred_inds = (pred_mask == 1)
        true_inds = (true_mask == 1)
        intersection = (pred_inds & true_inds).sum().float()
        union = (pred_inds | true_inds).sum().float()
        if union == 0:
            ious.append(float('nan'))  # Ignore if there is no ground truth
        else:
            ious.append((intersection / union).item())
            
    else:
        for cls in range(num_classes):
            pred_inds = (pred_mask == cls)
            true_inds = (true_mask == cls)
            intersection = (pred_inds & true_inds).sum().float()
            union = (pred_inds | true_inds).sum().float()
            if union == 0:
                ious.append(float('nan'))  # Ignore if there is no ground truth
            else:
                ious.append((intersection / union).item())
                
    return ious


def calculate_dice(pred_mask, true_mask, num_classes):
    dices = []
    pred_mask = pred_mask.view(-1)
    true_mask = true_mask.view(-1)
    if num_classes == 1:
        pred_inds = (pred_mask == 1)
        true_inds = (true_mask == 1)
        intersection = (pred_inds & true_inds).sum().float()
        total = pred_inds.sum() + true_inds.sum()
        if total == 0:
            dices.append(float('nan'))
        else:
            dices.append((2 * intersection / total).item())
    
    else:
        for cls in range(num_classes):
            pred_inds = (pred_mask == cls)
            true_inds = (true_mask == cls)
            intersection = (pred_inds & true_inds).sum().float()
            total = pred_inds.sum() + true_inds.sum()
            if total == 0:
                dices.append(float('nan'))
            else:
                dices.append((2 * intersection / total).item())
    return dices


def calculate_precision_recall(pred_mask, true_mask, num_classes):
    precisions = []
    recalls = []
    pred_mask = pred_mask.view(-1)
    true_mask = true_mask.view(-1)
    
    if num_classes == 1 :
        pred_inds = (pred_mask == 1)
        true_inds = (true_mask == 1)
        true_positive = (pred_inds & true_inds).sum().float()
        predicted_positive = pred_inds.sum().float()
        actual_positive = true_inds.sum().float()
        if predicted_positive == 0:
            precisions.append(float('nan'))
        else:
            precisions.append((true_positive / predicted_positive).item())
        if actual_positive == 0:
            recalls.append(float('nan'))
        else:
            recalls.append((true_positive / actual_positive).item())
            
    else:
        for cls in range(num_classes):
            pred_inds = (pred_mask == cls)
            true_inds = (true_mask == cls)
            true_positive = (pred_inds & true_inds).sum().float()
            predicted_positive = pred_inds.sum().float()
            actual_positive = true_inds.sum().float()
            if predicted_positive == 0:
                precisions.append(float('nan'))
            else:
                precisions.append((true_positive / predicted_positive).item())
            if actual_positive == 0:
                recalls.append(float('nan'))
            else:
                recalls.append((true_positive / actual_positive).item())
                
    return precisions, recalls


def calculate_hm_score(pred_mask, true_mask, num_classes):
    """
    Calculate HM (Hamming-like) score for segmentation masks.
    HM score = (union - intersection) / union
    """
    hm_scores = []
    pred_mask = pred_mask.view(-1)
    true_mask = true_mask.view(-1)
    
    if num_classes == 1:
        pred_inds = (pred_mask == 1).cpu().numpy()  # Convert to numpy
        true_inds = (true_mask == 1).cpu().numpy()
        intersection = np.sum(pred_inds * true_inds)
        union = np.sum(np.logical_or(pred_inds, true_inds))
        if union == 0:
            hm_scores.append(float('nan'))
        else:
            hm_score = (union - intersection) / union
            hm_scores.append(hm_score)
    else:
        for cls in range(num_classes):
            pred_inds = (pred_mask == cls).cpu().numpy()
            true_inds = (true_mask == cls).cpu().numpy()
            intersection = np.sum(pred_inds * true_inds)
            union = np.sum(np.logical_or(pred_inds, true_inds))
            if union == 0:
                hm_scores.append(float('nan'))
            else:
                hm_score = (union - intersection) / union
                hm_scores.append(hm_score)
                
    return hm_scores


def calculate_xor_score(pred_mask, true_mask, num_classes):
    """
    Calculate XOR score for segmentation masks.
    XOR score = (union - intersection) / sum(ground_truth)
    """
    xor_scores = []
    pred_mask = pred_mask.view(-1)
    true_mask = true_mask.view(-1)
    
    if num_classes == 1:
        pred_inds = (pred_mask == 1).cpu().numpy()  # Convert to numpy
        true_inds = (true_mask == 1).cpu().numpy()
        intersection = np.sum(pred_inds * true_inds)
        union = np.sum(np.logical_or(pred_inds, true_inds))
        gt_sum = np.sum(true_inds)
        if gt_sum == 0:
            xor_scores.append(float('nan'))
        else:
            xor_score = (union - intersection) / gt_sum
            xor_scores.append(xor_score)
    else:
        for cls in range(num_classes):
            pred_inds = (pred_mask == cls).cpu().numpy()
            true_inds = (true_mask == cls).cpu().numpy()
            intersection = np.sum(pred_inds * true_inds)
            union = np.sum(np.logical_or(pred_inds, true_inds))
            gt_sum = np.sum(true_inds)
            if gt_sum == 0:
                xor_scores.append(float('nan'))
            else:
                xor_score = (union - intersection) / gt_sum
                xor_scores.append(xor_score)
                
    return xor_scores


def calculate_f1_score(precisions, recalls):
    f1_scores = []
    for p, r in zip(precisions, recalls):
        if p == 0 or r == 0 or p != p or r != r:  # Check for zero or NaN
            f1_scores.append(float('nan'))
        else:
            f1_scores.append(2 * p * r / (p + r))
    return f1_scores


def evaluate_metrics(pred_mask, true_mask, num_classes):
    ious = calculate_iou(pred_mask, true_mask, num_classes)
    dices = calculate_dice(pred_mask, true_mask, num_classes)
    precisions, recalls = calculate_precision_recall(pred_mask, true_mask, num_classes)
    f1_scores = calculate_f1_score(precisions, recalls)
    hm_scores = calculate_hm_score(pred_mask, true_mask, num_classes)
    xor_scores = calculate_xor_score(pred_mask, true_mask, num_classes)

    metrics = {
        'IoU': np.nanmean(ious) if len(ious) > 0 else np.nan,
        'Dice': np.nanmean(dices) if len(dices) > 0 else np.nan,
        'Precision': np.nanmean(precisions) if len(precisions) > 0 else np.nan,
        'Recall': np.nanmean(recalls) if len(recalls) > 0 else np.nan,
        'F1 Score': np.nanmean(f1_scores) if len(f1_scores) > 0 else np.nan,
        'HM Score': np.nanmean(hm_scores) if len(hm_scores) > 0 else np.nan,
        'XOR Score': np.nanmean(xor_scores) if len(xor_scores) > 0 else np.nan
    }

    return metrics


def save_metrics(epoch, train_loss, train_metrics, val_loss, val_metrics, output_csv):
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
        writer.writerow([epoch] + [train_loss] +  train_metrics + [val_loss] + val_metrics)
