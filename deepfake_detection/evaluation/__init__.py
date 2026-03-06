"""
Evaluation module for deepfake detection models.

This module contains evaluation metrics, explainability tools,
and performance comparison utilities.
"""

from .metrics import EvaluationMetrics, ModelComparator
from .explainability import GradCAMVisualizer, ExplainabilityAnalyzer
from .benchmarking import BenchmarkRunner, PerformanceAnalyzer

__all__ = [
    "EvaluationMetrics",
    "ModelComparator",
    "GradCAMVisualizer", 
    "ExplainabilityAnalyzer",
    "BenchmarkRunner",
    "PerformanceAnalyzer"
]
