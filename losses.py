import torch.nn as nn
import torch
import torch.nn.functional as F
import yaml

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)



class BinaryDiceLoss(nn.Module):
    
    def __init__(self, smooth=1e-5):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        logits = torch.sigmoid(logits)
        logits = logits.view(-1)
        targets = targets.view(-1)

        intersection = (logits * targets).sum()
        dice = (2. * intersection + self.smooth) / (logits.sum() + targets.sum() + self.smooth)
        loss = 1 - dice
        return loss
    
# Cross-Entropy loss
loss_ce = torch.nn.CrossEntropyLoss()
# Intra-scale Loss
loss_intra = torch.nn.L1Loss(reduction='mean')
# Inter-scale Loss
loss_inter = torch.nn.CosineEmbeddingLoss()