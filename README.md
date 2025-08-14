# CrackSegmenter: Self-Supervised Multi-Scale Transformer with Attention-Guided Fusion for Efficient Crack Detection

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A state-of-the-art self-supervised crack segmentation framework that leverages multi-scale feature fusion and attention mechanisms for robust crack detection in various materials and surfaces.

## 🚀 Features

- **Multi-Scale Architecture**: Implements SAE (Scale-Aware Embedding) and AGF (Attention-Guided Fusion) for robust feature extraction
- **Self-Supervised Learning**: Leverages unlabeled data for improved generalization
- **Multiple Backbones**: Support for ResNet, EfficientNet, and other popular architectures
- **Baseline Models**: Includes UNet, FCN, DeepLabV3+ for comparison
- **Comprehensive Metrics**: IoU, Dice, Precision, Recall, F1-Score evaluation
- **Easy Configuration**: YAML-based configuration for quick experimentation
- **Multiple Datasets**: Support for CrackTree200, CFD, Forest, and custom datasets

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage](#usage)
- [Architecture](#architecture)
- [Datasets](#datasets)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- CUDA (for GPU training)

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/Blessing988/cracksegmenter.git
cd cracksegmenter

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### Requirements

```bash
pip install torch torchvision
pip install segmentation-models-pytorch
pip install albumentations
pip install opencv-python
pip install tqdm
pip install pyyaml
pip install einops
pip install scikit-image
```

## 🚀 Quick Start

### 1. Basic Training

```bash
# Train with default configuration
python train.py

# Train with custom config
python train.py --config configs/example_cracksegmenter.yaml
```

### 2. Evaluation

```bash
# Evaluate a single model
python scripts/evaluate.py --model_path trained_models/best_model.pth

# Run comprehensive evaluation
python scripts/evaluate_all.py --config configs/evaluation_config.yaml

# Run evaluation examples
python examples/evaluation_example.py

# Run validation metrics examples
python examples/validation_metrics_example.py
```

### 2. Baseline Model Training

```bash
# Set baseline=True in config.yaml and specify architecture
python train.py
```

### 3. Inference

```bash
python inference.py --model_path trained_models/best_model.pth --image_path test_image.jpg
```

## ⚙️ Configuration

The framework uses YAML configuration files for easy parameter management. Key configuration options:

### Model Configuration

```yaml
model:
  num_classes: 1
  nChannel: 100
  backbone: resnet18
  pretrained: True
  use_cbam: True
  use_transformer: True
  architecture: Crack-Segmenter-v2  # or UNet, FCN, DeepLabV3+
  use_dice: True
  use_bce: True
  baseline: False  # Set to True for baseline models
```

### Training Configuration

```yaml
training:
  batch_size: 4
  num_epochs: 500
  learning_rate: 0.0001
  weight_decay: 0.00001
  early_stopping_patience: 50
```

### Dataset Configuration

```yaml
data:
  image_size: 448
  root_dir: '/path/to/datasets'
  dataset_name: 'cracktree200'  # cracktree200, cfd, forest, gaps_384
  num_workers: 4
  mask_ext: '.png'
```

## 🏗️ Architecture

### CrackSegmenter (Multi-Scale Transformer)

The proposed architecture consists of three main components:

1. **Scale-Aware Embedding (SAE)**: Multi-scale patch embeddings for capturing features at different resolutions
2. **Attention-Guided Fusion (AGF)**: Attention mechanism for intelligent feature fusion
3. **Multi-Scale Feature Integration**: Hierarchical feature combination for robust segmentation

### Baseline Models

- **UNet**: Classic encoder-decoder architecture
- **FCN**: Fully Convolutional Network
- **DeepLabV3+**: Advanced semantic segmentation model
- **Custom Variants**: Modified architectures for specific use cases

## 📊 Datasets

### Supported Datasets

- **CrackTree200**: 200 pavement crack images with annotations
- **CFD**: Concrete crack detection dataset
- **Forest**: Forest crack dataset
- **GAPS**: Generic crack dataset with 384x384 resolution

#### Download and Preparation

- Download the datasets from the provided Google Drive link: [Google Drive](https://drive.google.com/file/d/1xrOqv0-3uMHjZyEUrerOYiYXW_E8SUMP/view).
- Extract them into the `datasets/` directory, using one subfolder per dataset (e.g., `datasets/cracktree200`, `datasets/cfd`, `datasets/forest`, `datasets/gaps_384`). Ensure each dataset follows the structure shown below.

### Custom Dataset Format

```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── masks/
    ├── train/
    ├── val/
    └── test/
```

## 🎯 Training

### Training Modes

1. **Baseline Training**: Set `baseline: True` in config
2. **CrackSegmenter Training**: Set `baseline: False` and choose architecture
3. **Ablation Studies**: Use `train_ablation.py` for component analysis

### Training Commands

```bash
# Standard training
python scripts/train.py

# Ablation study
python scripts/train_ablation.py

# Multi-GPU training
python -m torch.distributed.launch --nproc_per_node=4 scripts/train.py
```

### Training Tips

- Start with smaller datasets for quick experimentation
- Use mixed precision training for faster training
- Monitor validation metrics for early stopping
- Use data augmentation for better generalization

## 📈 Evaluation

### Metrics

- **IoU (Intersection over Union)**: Primary segmentation metric
- **Dice Coefficient**: Alternative overlap measure
- **Precision/Recall**: Classification performance
- **F1-Score**: Balanced precision-recall measure
- **XOR Metric**: Exclusive OR-based segmentation measure
- **HM Metric**: Hausdorff-based segmentation measure

### Evaluation Scripts

The repository provides three evaluation scripts:

1. **`scripts/evaluate.py`**: Simple evaluation of a single model
2. **`scripts/evaluate_all.py`**: Comprehensive evaluation of all models on all datasets
3. **`scripts/validate_metrics.py`**: File-based validation metrics computation for predicted masks

### Evaluation Commands

```bash
# Evaluate single trained model
python scripts/evaluate.py --model_path trained_models/best_model.pth

# Comprehensive evaluation of all models on all datasets
python scripts/evaluate_all.py --config configs/evaluation_config.yaml

# Evaluate specific models on specific datasets
python scripts/evaluate_all.py --models Crack-Segmenter-v2 Crack-Segmenter-v1 --datasets cracktree200 cfd

# Save predicted masks during evaluation
python scripts/evaluate_all.py --save_masks --output_dir ./evaluation_results

# Generate predictions
python scripts/inference.py --model_path trained_models/best_model.pth --output_dir predictions/

# Compute validation metrics for predicted masks
python scripts/validate_metrics.py --pred_base_path ./evaluation_results/predicted_masks --gt_base_path ./datasets

# Validate specific models and datasets
python scripts/validate_metrics.py --pred_base_path ./my_predictions --gt_base_path ./my_datasets --models Crack-Segmenter-v2 --datasets cracktree200 cfd
```

### Evaluation Features

- **Automatic Model Discovery**: Automatically finds available trained models
- **Dataset Validation**: Checks dataset structure and availability
- **Comprehensive Metrics**: Computes IoU, Dice, Precision, Recall, F1-Score, XOR, and HM metrics
- **Mask Saving**: Optionally saves predicted masks for analysis
- **Visualization**: Optional real-time visualization during evaluation
- **Robust Error Handling**: Continues evaluation even if individual samples fail
- **Detailed Logging**: Comprehensive logging for debugging and monitoring

### Evaluation Workflow

1. **Prepare Datasets**: Ensure datasets follow the expected structure with `val/images/` and `val/masks/` subdirectories
2. **Train Models**: Train models using the training scripts to generate checkpoints
3. **Configure Evaluation**: Update `configs/evaluation_config.yaml` with your paths and preferences
4. **Run Evaluation**: Use `scripts/evaluate_all.py` for comprehensive evaluation or `scripts/evaluate.py` for single model evaluation
5. **Analyze Results**: Review the generated CSV results and saved predicted masks

### Expected Directory Structure

```bash
project_root/
├── trained_models/              # Model checkpoints
│   ├── cracktree200/
│   │   ├── Crack-Segmenter-v2/
│   │   │   └── best_model.pth
│   │   └── Crack-Segmenter-v1/
│   │       └── best_model.pth
│   └── cfd/
│       └── Crack-Segmenter-v2/
│           └── best_model.pth
├── datasets/                    # Validation datasets
│   ├── cracktree200/
│   │   └── val/
│   │       ├── images/         # Validation images
│   │       └── masks/          # Ground truth masks
│   └── cfd/
│       └── val/
│           ├── images/
│           └── masks/
└── evaluation_results/          # Output directory
    ├── evaluation_results.csv   # Results summary
    ├── evaluation.log          # Evaluation logs
    └── predicted_masks/        # Saved predicted masks
        ├── cracktree200/
        │   └── Crack-Segmenter-v2/
        └── cfd/
            └── Crack-Segmenter-v2/
```
## 🔍 Validation Metrics

### Overview

The validation metrics functionality provides comprehensive evaluation of predicted masks against ground truth, supporting both existing metrics (IoU, Dice, Precision, Recall, F1-Score) and new specialized metrics (XOR, HM).

### Validation Metrics Script

**`scripts/validate_metrics.py`**: Computes validation metrics for predicted masks stored as files.

### Key Features

- **File-Based Validation**: Works with saved predicted masks and ground truth files
- **Multiple Metrics**: IoU, Dice, XOR, HM, and more
- **Flexible Paths**: Configurable prediction and ground truth directories
- **Detailed Reporting**: Per-image and summary-level metrics
- **Performance Analysis**: Model comparison and dataset difficulty analysis

### Usage Examples

```bash
# Basic validation metrics computation
python scripts/validate_metrics.py --pred_base_path ./evaluation_results/predicted_masks --gt_base_path ./datasets

# Specific models and datasets
python scripts/validate_metrics.py --pred_base_path ./my_predictions --gt_base_path ./my_datasets --models Crack-Segmenter-v2 --datasets cracktree200 cfd

# Save detailed per-image metrics
python scripts/validate_metrics.py --pred_base_path ./evaluation_results/predicted_masks --gt_base_path ./datasets --save_detailed

# Custom output directory
python scripts/validate_metrics.py --pred_base_path ./evaluation_results/predicted_masks --gt_base_path ./datasets --output_dir ./my_validation_results
```

### Expected Directory Structure

```
project_root/
├── evaluation_results/
│   └── predicted_masks/        # Predicted masks from evaluation
│       ├── Crack-Segmenter-v2/
│       │   ├── cracktree200/
│       │   │   ├── image1.png
│       │   │   └── image2.png
│       │   └── cfd/
│       │       ├── image1.png
│       │       └── image2.png
│       └── Crack-Segmenter-v1/
│           └── ...
├── datasets/                    # Ground truth datasets
│   ├── cracktree200/
│   │   └── val/
│   │       └── masks/
│   │           ├── image1.png
│   │           ├── image2.png
│   │           └── ...
│   └── cfd/
│       └── val/
│           └── masks/
│               ├── image1.png
│               └── image2.png
└── validation_results/          # Output directory
    ├── validation_metrics_summary.csv      # Summary results
    ├── validation_metrics_detailed.csv     # Detailed per-image results
    ├── model_performance_summary.csv       # Model performance analysis
    └── dataset_difficulty_analysis.csv     # Dataset difficulty analysis
```

### Configuration

Use `configs/validation_metrics_config.yaml` to configure:
- Paths to predicted masks and ground truth
- Models and datasets to evaluate
- File extensions and evaluation settings
- Output directory and logging options


### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Run specific test files
python tests/test_basic.py
python tests/test_evaluation.py
python tests/test_validation_metrics.py

# Format code
black src/ scripts/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

<!--
## 🙏 Acknowledgments

- Original paper authors for the foundational research
- PyTorch community for the excellent framework
- Segmentation Models PyTorch for baseline implementations

## 📞 Contact

- **Maintainer**: [Blessing Agyei Kyem](mailto:your.email@example.com)
- **Project Link**: [https://github.com/Blessing988/cracksegmenter](https://github.com/Blessing988/cracksegmenter)
- **Issues**: [GitHub Issues](https://github.com/Blessing988/cracksegmenter/issues)

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{cracksegmenter2024,
  title={CrackSegmenter: Self-Supervised Crack Segmentation with Multi-Scale Feature Fusion},
  author={Your Name and Co-authors},
  journal={arXiv preprint},
  year={2024}
}
```

---

**Star this repository if you find it useful! ⭐**
-->
