#!/usr/bin/env python3
"""
Inference script for CrackSegmenter.

This script loads a trained model and performs inference on images.
"""

import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models import create_model
from data.transforms import get_inference_transforms
from utils.config import load_config


def load_model(model_path: str, config: dict, device: torch.device):
    """
    Load trained model from checkpoint.
    
    Args:
        model_path (str): Path to model checkpoint
        config (dict): Configuration dictionary
        device (torch.device): Device to load model on
        
    Returns:
        torch.nn.Module: Loaded model
    """
    # Create model
    if config['model']['baseline']:
        model = create_model(
            architecture=config['model']['architecture'],
            encoder_name=config['model']['backbone'],
            in_channels=3,
            num_classes=config['model']['num_classes'],
            encoder_weights=None  # Don't load pretrained weights
        )
    else:
        model = create_model(
            architecture=config['model']['architecture'],
            input_dim=3,
            embed_size=config['model']['nChannel']
        )
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model


def preprocess_image(image_path: str, config: dict, device: torch.device):
    """
    Preprocess image for inference.
    
    Args:
        image_path (str): Path to input image
        config (dict): Configuration dictionary
        device (torch.device): Device to put tensor on
        
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Get transforms
    transforms = get_inference_transforms(config['data']['image_size'])
    
    # Apply transforms
    transformed = transforms(image=image)
    image_tensor = transformed['image'].unsqueeze(0).to(device)
    
    return image_tensor, image


def predict(model: torch.nn.Module, image_tensor: torch.Tensor):
    """
    Perform prediction with the model.
    
    Args:
        model (torch.nn.Module): Trained model
        image_tensor (torch.Tensor): Preprocessed image tensor
        
    Returns:
        torch.Tensor: Prediction output
    """
    with torch.no_grad():
        outputs = model(image_tensor)
        
        # Handle different output formats
        if isinstance(outputs, tuple):
            main_output = outputs[0]  # Main prediction
        else:
            main_output = outputs
        
        # Apply sigmoid for binary segmentation
        probabilities = torch.sigmoid(main_output)
        
        # Convert to binary prediction
        predictions = (probabilities > 0.5).float()
        
        return predictions, probabilities


def visualize_results(original_image: Image.Image, predictions: torch.Tensor, 
                     probabilities: torch.Tensor, save_path: str = None):
    """
    Visualize prediction results.
    
    Args:
        original_image (Image.Image): Original input image
        predictions (torch.Tensor): Binary predictions
        probabilities (torch.Tensor): Probability maps
        save_path (str, optional): Path to save visualization
    """
    # Convert tensors to numpy arrays
    pred_np = predictions.squeeze().cpu().numpy()
    prob_np = probabilities.squeeze().cpu().numpy()
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Probability map
    im1 = axes[1].imshow(prob_np, cmap='hot', alpha=0.8)
    axes[1].set_title('Probability Map')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1])
    
    # Binary prediction
    axes[2].imshow(pred_np, cmap='gray')
    axes[2].set_title('Binary Prediction')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()


def save_predictions(predictions: torch.Tensor, probabilities: torch.Tensor, 
                    output_dir: str, base_name: str):
    """
    Save prediction results.
    
    Args:
        predictions (torch.Tensor): Binary predictions
        probabilities (torch.Tensor): Probability maps
        output_dir (str): Output directory
        base_name (str): Base name for output files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to numpy arrays
    pred_np = predictions.squeeze().cpu().numpy()
    prob_np = probabilities.squeeze().cpu().numpy()
    
    # Save binary prediction
    pred_path = os.path.join(output_dir, f"{base_name}_pred.png")
    pred_img = Image.fromarray((pred_np * 255).astype(np.uint8))
    pred_img.save(pred_path)
    
    # Save probability map
    prob_path = os.path.join(output_dir, f"{base_name}_prob.npy")
    np.save(prob_path, prob_np)
    
    print(f"Predictions saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='CrackSegmenter Inference')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--image_path', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default='./predictions',
                       help='Output directory for predictions')
    parser.add_argument('--visualize', action='store_true',
                       help='Show visualization')
    parser.add_argument('--save_results', action='store_true',
                       help='Save prediction results')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Image file not found: {args.image_path}")
    
    # Load configuration
    config = load_config(args.config)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = load_model(args.model_path, config, device)
    print("Model loaded successfully!")
    
    # Preprocess image
    print("Preprocessing image...")
    image_tensor, original_image = preprocess_image(
        args.image_path, config, device
    )
    
    # Perform inference
    print("Running inference...")
    predictions, probabilities = predict(model, image_tensor)
    
    # Get base name for output files
    base_name = os.path.splitext(os.path.basename(args.image_path))[0]
    
    # Visualize results
    if args.visualize:
        print("Visualizing results...")
        visualize_results(original_image, predictions, probabilities)
    
    # Save results
    if args.save_results:
        print("Saving results...")
        save_predictions(predictions, probabilities, args.output_dir, base_name)
    
    print("Inference completed!")


if __name__ == '__main__':
    main()
