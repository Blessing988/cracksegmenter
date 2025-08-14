"""
Setup script for CrackSegmenter package.
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="cracksegmenter",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Self-Supervised Crack Segmentation with Multi-Scale Feature Fusion",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/cracksegmenter",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/cracksegmenter/issues",
        "Source": "https://github.com/yourusername/cracksegmenter",
        "Documentation": "https://github.com/yourusername/cracksegmenter#readme",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Framework :: PyTorch",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "pre-commit>=2.20.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.17.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipywidgets>=7.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "cracksegmenter-train=scripts.train:main",
            "cracksegmenter-inference=scripts.inference:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "crack-segmentation",
        "computer-vision",
        "deep-learning",
        "pytorch",
        "segmentation",
        "multi-scale",
        "attention",
        "transformer",
    ],
)
