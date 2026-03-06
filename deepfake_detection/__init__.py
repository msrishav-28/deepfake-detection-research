"""
Deepfake Detection Research Project

A comprehensive deepfake detection system using stacked ensemble of Vision Transformers
with explainable AI capabilities.

This project transforms the PyTorch Image Models (timm) codebase into a sophisticated
deepfake detection system that combines three pre-trained Vision Transformer models
using stacked generalization (meta-learning).
"""

__version__ = "1.0.0"
__author__ = "msrishav-28"

from . import data
from . import models
from . import utils
from . import evaluation

__all__ = [
    "data",
    "models", 
    "utils",
    "evaluation"
]
