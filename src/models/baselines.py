"""
Baseline models for crack segmentation comparison.

This module contains standard segmentation architectures:
- UNet: Classic encoder-decoder architecture
- FCN: Fully Convolutional Network
- DeepLabV3+: Advanced semantic segmentation model
- Custom baseline variants
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


def create_baseline_model(architecture='Unet', encoder_name='resnet50', 
                         in_channels=3, num_classes=1, encoder_weights='imagenet'):
    """
    Create a baseline segmentation model using segmentation-models-pytorch.
    
    Args:
        architecture (str): Model architecture ('Unet', 'FCN', 'DeepLabV3', etc.)
        encoder_name (str): Encoder backbone ('resnet18', 'resnet50', 'efficientnet-b0', etc.)
        in_channels (int): Number of input channels
        num_classes (int): Number of output classes
        encoder_weights (str): Pre-trained weights ('imagenet', 'ssl', 'swsl', None)
    
    Returns:
        nn.Module: Configured segmentation model
    """
    model = getattr(smp, architecture)(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes
    )
    return model


class UNet(nn.Module):
    """UNet baseline model for crack segmentation."""
    
    def __init__(self, in_channels=3, num_classes=1, encoder_name='resnet18', 
                 encoder_weights='imagenet'):
        super().__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes
        )
    
    def forward(self, x):
        return self.model(x)


class FCN(nn.Module):
    """FCN baseline model for crack segmentation."""
    
    def __init__(self, in_channels=3, num_classes=1, encoder_name='resnet18', 
                 encoder_weights='imagenet'):
        super().__init__()
        self.model = smp.FPN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes
        )
    
    def forward(self, x):
        return self.model(x)


class DeepLabV3(nn.Module):
    """DeepLabV3 baseline model for crack segmentation."""
    
    def __init__(self, in_channels=3, num_classes=1, encoder_name='resnet18', 
                 encoder_weights='imagenet'):
        super().__init__()
        self.model = smp.DeepLabV3(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes
        )
    
    def forward(self, x):
        return self.model(x)


class DeepLabV3Plus(nn.Module):
    """DeepLabV3+ baseline model for crack segmentation."""
    
    def __init__(self, in_channels=3, num_classes=1, encoder_name='resnet18', 
                 encoder_weights='imagenet'):
        super().__init__()
        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes
        )
    
    def forward(self, x):
        return self.model(x)


class PSPNet(nn.Module):
    """PSPNet baseline model for crack segmentation."""
    
    def __init__(self, in_channels=3, num_classes=1, encoder_name='resnet18', 
                 encoder_weights='imagenet'):
        super().__init__()
        self.model = smp.PSPNet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes
        )
    
    def forward(self, x):
        return self.model(x)


class LinkNet(nn.Module):
    """LinkNet baseline model for crack segmentation."""
    
    def __init__(self, in_channels=3, num_classes=1, encoder_name='resnet18', 
                 encoder_weights='imagenet'):
        super().__init__()
        self.model = smp.Linknet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes
        )
    
    def forward(self, x):
        return self.model(x)


class MSFormer_baseline(nn.Module):
    """Baseline CrackSegmenter model using UNet architecture."""
    
    def __init__(self, input_dim, num_classes=1):
        super().__init__()
        self.model = smp.Unet(
            encoder_name='resnet18',
            encoder_weights='imagenet',
            in_channels=input_dim,
            classes=num_classes
        )
    
    def forward(self, x):
        out = self.model(x)
        return out, None, None, None, None, None, None, None


# Dictionary mapping architecture names to model classes
BASELINE_MODELS = {
    'unet': UNet,
    'fcn': FCN,
    'deeplabv3': DeepLabV3,
    'deeplabv3plus': DeepLabV3Plus,
    'pspnet': PSPNet,
    'linknet': LinkNet,
    'msformer_baseline': MSFormer_baseline,
}


def get_baseline_model(architecture_name, **kwargs):
    """
    Get a baseline model by name.
    
    Args:
        architecture_name (str): Name of the architecture
        **kwargs: Additional arguments to pass to the model constructor
    
    Returns:
        nn.Module: Configured baseline model
    
    Raises:
        ValueError: If architecture name is not supported
    """
    architecture_name = architecture_name.lower()
    
    if architecture_name not in BASELINE_MODELS:
        raise ValueError(f"Unsupported baseline architecture: {architecture_name}. "
                       f"Supported: {list(BASELINE_MODELS.keys())}")
    
    return BASELINE_MODELS[architecture_name](**kwargs)
