# utils.py
import torch
import torch.nn.functional as F
import os

def save_checkpoint(state, is_best, checkpoint_dir='checkpoints', filename='checkpoint.pth.tar'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save(state, best_path)

def load_checkpoint(model, optimizer, checkpoint_dir='checkpoints', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint_dir, filename)
    if os.path.isfile(filepath):
        print(f"Loading checkpoint '{filepath}'")
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Loaded checkpoint '{filepath}' (epoch {checkpoint['epoch']})")
    else:
        print(f"No checkpoint found at '{filepath}'")

# Additional utility functions
def adjust_learning_rate(optimizer, epoch, initial_lr, lr_decay_epoch=30):
    """Sets the learning rate to the initial LR decayed by 10 every lr_decay_epoch epochs"""
    lr = initial_lr * (0.1 ** (epoch // lr_decay_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


import numpy as np
import cv2


def dice_metric(A, B):
    intersect = np.sum(A * B)
    fsum = np.sum(A)
    ssum = np.sum(B)
    dice = (2 * intersect ) / (fsum + ssum)
    
    return dice    


def hm_metric(A, B):
    intersection = A * B
    union = np.logical_or(A, B)
    hm_score = (np.sum(union) - np.sum(intersection)) / np.sum(union)
    
    return hm_score


def xor_metric(A, GT):
    intersection = A * GT
    union = np.logical_or(A, GT)
    xor_score = (np.sum(union) - np.sum(intersection)) / np.sum(GT)
    
    return xor_score


# Original working one
# def create_mask(pred, GT):
    
#     kernel = np.ones((7,7),np.uint8) 
#     dilated_GT = cv2.dilate(GT, kernel, iterations = 2)

#     mult = pred * GT        
#     unique, count = np.unique(mult[mult !=0], return_counts=True)
#     cls= unique[np.argmax(count)]
    
#     lesion = np.where(pred==cls, 1, 0) * dilated_GT
    
#     return lesion


# def create_mask(pred: np.ndarray, GT: np.ndarray,
#                 dilate: bool = False,
#                 kernel_size: int = 3,
#                 iterations: int = 1) -> np.ndarray:
#     """
#     pred: H×W array of predicted class labels
#     GT:   H×W binary ground‑truth mask (0 or 1)
#     dilate: whether to apply a small dilation to GT
#     kernel_size: size of the square dilation kernel
#     iterations:  number of dilation iterations
#     """
        
#     # 1) Optionally dilate GT
#     if dilate:
#         kernel = np.ones((kernel_size, kernel_size), np.uint8)
#         GT_proc = cv2.dilate(GT, kernel, iterations=iterations)
#     else:
#         GT_proc = GT

#     # 2) If GT has no foreground, return all zeros
#     if GT_proc.sum() == 0:
#         return np.zeros_like(GT_proc, dtype=np.uint8)

#     # 3) Find the most frequent pred label within the GT region
#     region_preds = pred[GT_proc == 1]
#     if region_preds.size == 0:
#         return np.zeros_like(GT_proc, dtype=np.uint8)

#     labels, counts = np.unique(region_preds, return_counts=True)
#     cls = labels[np.argmax(counts)]

#     # 4) Build the mask: pixels == cls AND inside GT_proc
#     lesion = np.zeros_like(GT_proc, dtype=np.uint8)
#     lesion[(pred == cls) & (GT_proc == 1)] = 1
#     print('Shape', lesion.shape)
#     print('Maximum value', lesion.max())
#     print('Minimum value', lesion.min())
#     return lesion



# This was working but precision was always 1
# def create_mask(
#         pred: torch.Tensor,        # [H, W] class‑index tensor
#         GT:   torch.Tensor,        # [H, W] binary 0/1 tensor
#         dilate: bool = False,
#         kernel_size: int = 3,
#         iterations: int = 1
# ) -> torch.Tensor:
#     """
#     Returns a binary tensor (same shape as GT) where pixels are 1 iff
#       (a) they lie inside the (optionally dilated) GT mask **and**
#       (b) their predicted class equals the modal class within that GT.
#     """

#     # ---- 0.  Type & device consistency ------------------------------------
#     device = pred.device
#     GT     = GT.to(device).to(torch.uint8)      # be sure GT is 0/1 ints
#     pred   = pred.to(torch.int64)               # just in case

#     # ---- 1.  Optional dilation via max‑pool (GPU‑friendly) -----------------
#     if dilate:
#         pad = kernel_size // 2
#         x   = GT.unsqueeze(0).unsqueeze(0).float()          #  [1,1,H,W]
#         for _ in range(iterations):
#             x = F.max_pool2d(x, kernel_size, stride=1, padding=pad)
#         GT = x.squeeze(0).squeeze(0).to(torch.uint8)

#     # ---- 2.  Early exit if GT is empty -------------------------------------
#     if GT.sum() == 0:
#         return torch.zeros_like(GT, dtype=torch.uint8)

#     # ---- 3.  Modal class inside GT ----------------------------------------
#     region_preds = pred[GT == 1]                  # 1‑D tensor
#     modal_cls    = torch.mode(region_preds).values.item()

#     # ---- 4.  Lesion mask ---------------------------------------------------
#     lesion = ((pred == modal_cls) & (GT == 1)).to(torch.uint8)
#     return lesion                                # [H, W] tensor on same device



def create_mask(
        pred: torch.Tensor,        # [H, W] class indices
        GT:   torch.Tensor,        # [H, W] binary 0/1
        dilate: bool = False,
        kernel_size: int = 3,
        iterations: int = 1,
        restrict_to_gt: bool = False    # <‑‑ NEW FLAG
) -> torch.Tensor:
    """
    Returns a binary tensor (same shape as GT) where pixels are 1 iff
      (a) they lie inside the (optionally dilated) GT mask **and**
      (b) their predicted class equals the modal class within that GT.
    """

    # ---- 0.  Type & device consistency ------------------------------------
    device = pred.device
    GT     = GT.to(device).to(torch.uint8)      # be sure GT is 0/1 ints
    pred   = pred.to(torch.int64)               # just in case

    # ---- 1.  Optional dilation via max‑pool (GPU‑friendly) -----------------
    if dilate:
        pad = kernel_size // 2
        x   = GT.unsqueeze(0).unsqueeze(0).float()          #  [1,1,H,W]
        for _ in range(iterations):
            x = F.max_pool2d(x, kernel_size, stride=1, padding=pad)
        GT = x.squeeze(0).squeeze(0).to(torch.uint8)

    # ---- 2.  Early exit if GT is empty -------------------------------------
    if GT.sum() == 0:
        return torch.zeros_like(GT, dtype=torch.uint8)

    # 3.  modal class inside GT  (unchanged)  -------------------------------
    region_preds = pred[GT == 1]
    modal_cls    = torch.mode(region_preds).values.item()

    # 4.  lesion mask --------------------------------------------------------
    lesion = (pred == modal_cls)           # <-- keeps *all* pixels of that class
    if restrict_to_gt:                     #   optional old behaviour
        lesion = lesion & (GT == 1)

    return lesion.to(torch.uint8)                           # [H, W] tensor on same device

