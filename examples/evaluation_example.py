#!/usr/bin/env python3
"""
Example script demonstrating how to use the evaluation functionality.

This script shows different ways to evaluate models using the evaluation scripts.
"""

import os
import sys
import subprocess
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from utils.config import load_config, get_default_config


def run_command(command: str, description: str):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print('='*60)
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✅ Command executed successfully!")
        if result.stdout:
            print("Output:")
            print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with exit code {e.returncode}")
        if e.stderr:
            print("Error output:")
            print(e.stderr)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


def main():
    """Main function demonstrating evaluation usage."""
    print("🚀 CrackSegmenter Evaluation Examples")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists("scripts/evaluate.py"):
        print("❌ Please run this script from the repository root directory")
        return
    
    # Example 1: Single model evaluation
    print("\n📊 Example 1: Single Model Evaluation")
    print("This evaluates a single trained model on validation data.")
    
    # Check if we have a model to evaluate
    model_path = "trained_models/best_model.pth"
    if os.path.exists(model_path):
        run_command(
            f"python scripts/evaluate.py --model_path {model_path}",
            "Single model evaluation"
        )
    else:
        print(f"⚠️  Model not found at {model_path}")
        print("   This example requires a trained model to work.")
    
    # Example 2: Comprehensive evaluation
    print("\n📊 Example 2: Comprehensive Evaluation")
    print("This evaluates all available models on all available datasets.")
    
    # Check if evaluation config exists
    eval_config = "configs/evaluation_config.yaml"
    if os.path.exists(eval_config):
        run_command(
            f"python scripts/evaluate_all.py --config {eval_config}",
            "Comprehensive evaluation"
        )
    else:
        print(f"⚠️  Evaluation config not found at {eval_config}")
        print("   Creating a default evaluation configuration...")
        
        # Create default evaluation config
        default_config = get_default_config()
        default_config['evaluation'] = {
            'save_masks': True,
            'visualize': False,
            'mask_format': '.png',
            'compute_metrics': True
        }
        default_config['utils']['save_dir'] = './evaluation_results'
        default_config['utils']['checkpoint_root'] = './trained_models'
        default_config['utils']['dataset_root'] = './datasets'
        
        # Save config
        os.makedirs("configs", exist_ok=True)
        import yaml
        with open(eval_config, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False, indent=2)
        
        print(f"✅ Created {eval_config}")
        
        # Now try to run evaluation
        run_command(
            f"python scripts/evaluate_all.py --config {eval_config}",
            "Comprehensive evaluation with default config"
        )
    
    # Example 3: Evaluation with specific models and datasets
    print("\n📊 Example 3: Specific Model/Dataset Evaluation")
    print("This evaluates specific models on specific datasets.")
    
    run_command(
        "python scripts/evaluate_all.py --models Crack-Segmenter-v2 --datasets cracktree200 --save_masks",
        "Specific model/dataset evaluation with mask saving"
    )
    
    # Example 4: Evaluation with custom output directory
    print("\n📊 Example 4: Custom Output Directory")
    print("This saves evaluation results to a custom directory.")
    
    custom_output = "./my_evaluation_results"
    run_command(
        f"python scripts/evaluate_all.py --output_dir {custom_output} --save_masks",
        "Evaluation with custom output directory"
    )
    
    # Example 5: Evaluation with visualization
    print("\n📊 Example 5: Evaluation with Visualization")
    print("This shows real-time visualization during evaluation.")
    
    run_command(
        "python scripts/evaluate_all.py --visualize --save_masks",
        "Evaluation with visualization enabled"
    )
    
    # Summary
    print("\n" + "="*60)
    print("🎉 Evaluation Examples Completed!")
    print("\n📁 Output Locations:")
    print("   - Single evaluation: Results printed to console")
    print("   - Comprehensive evaluation: ./evaluation_results/")
    print("   - Predicted masks: ./evaluation_results/predicted_masks/")
    print("   - Evaluation logs: ./evaluation_results/evaluation.log")
    print("   - Results CSV: ./evaluation_results/evaluation_results.csv")
    
    print("\n🔧 Available Options:")
    print("   --config: Specify configuration file")
    print("   --models: Specify models to evaluate")
    print("   --datasets: Specify datasets to evaluate on")
    print("   --save_masks: Save predicted masks")
    print("   --visualize: Show real-time visualization")
    print("   --output_dir: Specify output directory")
    
    print("\n📚 For more information, see:")
    print("   - README.md: General usage and examples")
    print("   - configs/evaluation_config.yaml: Configuration options")
    print("   - scripts/evaluate_all.py --help: Command-line options")


if __name__ == '__main__':
    main()
