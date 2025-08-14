#!/usr/bin/env python3
"""
Example script demonstrating how to use the validation metrics functionality.

This script shows different ways to compute validation metrics for predicted masks.
"""

import os
import sys
import subprocess
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from utils.config import load_config


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
    """Main function demonstrating validation metrics usage."""
    print("🚀 CrackSegmenter Validation Metrics Examples")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists("scripts/validate_metrics.py"):
        print("❌ Please run this script from the repository root directory")
        return
    
    # Example 1: Basic validation metrics computation
    print("\n📊 Example 1: Basic Validation Metrics")
    print("This computes metrics for all models on all datasets using default paths.")
    
    run_command(
        "python scripts/validate_metrics.py --pred_base_path ./evaluation_results/predicted_masks --gt_base_path ./datasets",
        "Basic validation metrics computation"
    )
    
    # Example 2: Specific models and datasets
    print("\n📊 Example 2: Specific Models and Datasets")
    print("This evaluates specific models on specific datasets.")
    
    run_command(
        "python scripts/validate_metrics.py --pred_base_path ./evaluation_results/predicted_masks --gt_base_path ./datasets --models Crack-Segmenter-v2 Crack-Segmenter-v1 --datasets cracktree200 cfd",
        "Specific model/dataset validation"
    )
    
    # Example 3: Custom output directory
    print("\n📊 Example 3: Custom Output Directory")
    print("This saves results to a custom directory.")
    
    custom_output = "./my_validation_results"
    run_command(
        f"python scripts/validate_metrics.py --pred_base_path ./evaluation_results/predicted_masks --gt_base_path ./datasets --output_dir {custom_output}",
        "Validation with custom output directory"
    )
    
    # Example 4: Detailed metrics with custom paths
    print("\n📊 Example 4: Detailed Metrics with Custom Paths")
    print("This saves detailed per-image metrics and uses custom paths.")
    
    run_command(
        "python scripts/validate_metrics.py --pred_base_path ./my_predictions --gt_base_path ./my_datasets --save_detailed --output_dir ./detailed_validation",
        "Detailed validation with custom paths"
    )
    
    # Example 5: Debug logging
    print("\n📊 Example 5: Debug Logging")
    print("This runs with debug logging for troubleshooting.")
    
    run_command(
        "python scripts/validate_metrics.py --pred_base_path ./evaluation_results/predicted_masks --gt_base_path ./datasets --log_level DEBUG",
        "Validation with debug logging"
    )
    
    # Summary
    print("\n" + "="*60)
    print("🎉 Validation Metrics Examples Completed!")
    print("\n📁 Output Locations:")
    print("   - Default: ./validation_results/")
    print("   - Summary: ./validation_results/validation_metrics_summary.csv")
    print("   - Detailed: ./validation_results/validation_metrics_detailed.csv")
    print("   - Model Performance: ./validation_results/model_performance_summary.csv")
    print("   - Dataset Analysis: ./validation_results/dataset_difficulty_analysis.csv")
    
    print("\n🔧 Available Options:")
    print("   --pred_base_path: Path to predicted masks")
    print("   --gt_base_path: Path to ground truth masks")
    print("   --models: Specific models to evaluate")
    print("   --datasets: Specific datasets to evaluate")
    print("   --output_dir: Output directory for results")
    print("   --save_detailed: Save detailed per-image metrics")
    print("   --log_level: Logging level (DEBUG, INFO, WARNING, ERROR)")
    
    print("\n📚 For more information, see:")
    print("   - README.md: General usage and examples")
    print("   - configs/validation_metrics_config.yaml: Configuration options")
    print("   - scripts/validate_metrics.py --help: Command-line options")
    
    print("\n💡 Tips:")
    print("   - Ensure predicted masks and ground truth have matching filenames")
    print("   - Use --save_detailed for per-image analysis")
    print("   - Check logs for any file matching issues")
    print("   - Results are automatically saved as CSV files")


if __name__ == '__main__':
    main()
