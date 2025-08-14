"""
Checkpoint utility functions for model saving and loading.
"""

import torch
import os


def save_checkpoint(state, is_best, checkpoint_dir='checkpoints', filename='checkpoint.pth.tar'):
    """
    Save model checkpoint.
    
    Args:
        state: Dictionary containing model state
        is_best: Whether this is the best model so far
        checkpoint_dir: Directory to save checkpoints
        filename: Name of the checkpoint file
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save(state, best_path)


def load_checkpoint(model, optimizer, checkpoint_dir='checkpoints', filename='checkpoint.pth.tar'):
    """
    Load model checkpoint.
    
    Args:
        model: Model to load state into
        optimizer: Optimizer to load state into
        checkpoint_dir: Directory containing checkpoints
        filename: Name of the checkpoint file
        
    Returns:
        epoch: The epoch number from the checkpoint
    """
    filepath = os.path.join(checkpoint_dir, filename)
    if os.path.isfile(filepath):
        print(f"Loading checkpoint '{filepath}'")
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        print(f"Loaded checkpoint '{filepath}' (epoch {epoch})")
        return epoch
    else:
        print(f"No checkpoint found at '{filepath}'")
        return 0


def adjust_learning_rate(optimizer, epoch, initial_lr, lr_decay_epoch=30):
    """
    Sets the learning rate to the initial LR decayed by 10 every lr_decay_epoch epochs.
    
    Args:
        optimizer: The optimizer to adjust
        epoch: Current epoch
        initial_lr: Initial learning rate
        lr_decay_epoch: Epochs between learning rate decays
    """
    lr = initial_lr * (0.1 ** (epoch // lr_decay_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
