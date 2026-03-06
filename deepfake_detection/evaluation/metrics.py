"""
Comprehensive evaluation metrics for deepfake detection models.

This module provides detailed evaluation metrics including accuracy, precision,
recall, F1-score, AUC, confusion matrices, and statistical significance tests.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class EvaluationMetrics:
    """Comprehensive evaluation metrics for binary classification."""
    
    def __init__(self, class_names: List[str] = None):
        """
        Args:
            class_names: Names of the classes (default: ['Real', 'Fake'])
        """
        self.class_names = class_names or ['Real', 'Fake']
        self.results = {}
    
    def calculate_all_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Calculate all evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional)
            
        Returns:
            Dictionary containing all metrics
        """
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='binary', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='binary', zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, average='binary', zero_division=0)
        metrics['specificity'] = self._calculate_specificity(y_true, y_pred)
        
        # Additional metrics
        metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
        metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        metrics['tn'], metrics['fp'], metrics['fn'], metrics['tp'] = cm.ravel()
        
        # Class-wise metrics
        metrics['precision_per_class'] = precision_score(y_true, y_pred, average=None, zero_division=0).tolist()
        metrics['recall_per_class'] = recall_score(y_true, y_pred, average=None, zero_division=0).tolist()
        metrics['f1_per_class'] = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()
        
        # Probability-based metrics (if probabilities provided)
        if y_proba is not None:
            if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                # Binary classification with 2 columns
                y_proba_positive = y_proba[:, 1]
            else:
                # Single column of positive class probabilities
                y_proba_positive = y_proba.flatten()
            
            metrics['auc_roc'] = roc_auc_score(y_true, y_proba_positive)
            metrics['auc_pr'] = average_precision_score(y_true, y_proba_positive)
            
            # ROC curve data
            fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba_positive)
            metrics['roc_curve'] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': roc_thresholds.tolist()
            }
            
            # Precision-Recall curve data
            precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_proba_positive)
            metrics['pr_curve'] = {
                'precision': precision_curve.tolist(),
                'recall': recall_curve.tolist(),
                'thresholds': pr_thresholds.tolist()
            }
        
        # Classification report
        metrics['classification_report'] = classification_report(
            y_true, y_pred, target_names=self.class_names, output_dict=True
        )
        
        self.results = metrics
        return metrics
    
    def _calculate_specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate specificity (true negative rate)."""
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    def print_summary(self, metrics: Optional[Dict] = None) -> None:
        """Print a summary of the evaluation metrics."""
        if metrics is None:
            metrics = self.results
        
        if not metrics:
            logger.warning("No metrics to display")
            return
        
        print("\n" + "="*60)
        print("EVALUATION METRICS SUMMARY")
        print("="*60)
        
        # Main metrics
        print(f"Accuracy:     {metrics['accuracy']:.4f}")
        print(f"Precision:    {metrics['precision']:.4f}")
        print(f"Recall:       {metrics['recall']:.4f}")
        print(f"F1-Score:     {metrics['f1']:.4f}")
        print(f"Specificity:  {metrics['specificity']:.4f}")
        
        if 'auc_roc' in metrics:
            print(f"AUC-ROC:      {metrics['auc_roc']:.4f}")
            print(f"AUC-PR:       {metrics['auc_pr']:.4f}")
        
        print(f"Cohen's κ:    {metrics['cohen_kappa']:.4f}")
        print(f"Matthews CC:  {metrics['matthews_corrcoef']:.4f}")
        
        # Confusion matrix
        print(f"\nConfusion Matrix:")
        cm = np.array(metrics['confusion_matrix'])
        print(f"              Predicted")
        print(f"              {self.class_names[0]:<8} {self.class_names[1]:<8}")
        print(f"Actual {self.class_names[0]:<8} {cm[0,0]:<8} {cm[0,1]:<8}")
        print(f"       {self.class_names[1]:<8} {cm[1,0]:<8} {cm[1,1]:<8}")
        
        # Per-class metrics
        print(f"\nPer-Class Metrics:")
        for i, class_name in enumerate(self.class_names):
            print(f"{class_name}:")
            print(f"  Precision: {metrics['precision_per_class'][i]:.4f}")
            print(f"  Recall:    {metrics['recall_per_class'][i]:.4f}")
            print(f"  F1-Score:  {metrics['f1_per_class'][i]:.4f}")
    
    def plot_confusion_matrix(
        self,
        metrics: Optional[Dict] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 6)
    ) -> plt.Figure:
        """
        Plot confusion matrix.
        
        Args:
            metrics: Metrics dictionary (uses self.results if None)
            save_path: Path to save the plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if metrics is None:
            metrics = self.results
        
        cm = np.array(metrics['confusion_matrix'])
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax
        )
        
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        return fig
    
    def plot_roc_curve(
        self,
        metrics: Optional[Dict] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 6)
    ) -> plt.Figure:
        """
        Plot ROC curve.
        
        Args:
            metrics: Metrics dictionary (uses self.results if None)
            save_path: Path to save the plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if metrics is None:
            metrics = self.results
        
        if 'roc_curve' not in metrics:
            raise ValueError("ROC curve data not available. Provide prediction probabilities.")
        
        roc_data = metrics['roc_curve']
        auc_score = metrics['auc_roc']
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot ROC curve
        ax.plot(
            roc_data['fpr'], roc_data['tpr'],
            linewidth=2, label=f'ROC Curve (AUC = {auc_score:.3f})'
        )
        
        # Plot diagonal line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}")
        
        return fig
    
    def plot_precision_recall_curve(
        self,
        metrics: Optional[Dict] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 6)
    ) -> plt.Figure:
        """
        Plot Precision-Recall curve.
        
        Args:
            metrics: Metrics dictionary (uses self.results if None)
            save_path: Path to save the plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if metrics is None:
            metrics = self.results
        
        if 'pr_curve' not in metrics:
            raise ValueError("PR curve data not available. Provide prediction probabilities.")
        
        pr_data = metrics['pr_curve']
        auc_pr = metrics['auc_pr']
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot PR curve
        ax.plot(
            pr_data['recall'], pr_data['precision'],
            linewidth=2, label=f'PR Curve (AUC = {auc_pr:.3f})'
        )
        
        # Plot baseline
        baseline = np.sum(metrics['confusion_matrix'][1]) / np.sum(metrics['confusion_matrix'])
        ax.axhline(y=baseline, color='k', linestyle='--', linewidth=1, label=f'Baseline ({baseline:.3f})')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"PR curve saved to {save_path}")
        
        return fig
    
    def save_metrics_to_file(
        self,
        filepath: str,
        metrics: Optional[Dict] = None
    ) -> None:
        """
        Save metrics to a file.
        
        Args:
            filepath: Path to save the metrics
            metrics: Metrics dictionary (uses self.results if None)
        """
        if metrics is None:
            metrics = self.results
        
        if filepath.endswith('.json'):
            import json
            with open(filepath, 'w') as f:
                json.dump(metrics, f, indent=2)
        elif filepath.endswith('.yaml') or filepath.endswith('.yml'):
            import yaml
            with open(filepath, 'w') as f:
                yaml.dump(metrics, f, default_flow_style=False)
        else:
            raise ValueError("Unsupported file format. Use .json or .yaml")
        
        logger.info(f"Metrics saved to {filepath}")


class ModelComparator:
    """Compare performance between multiple models."""
    
    def __init__(self):
        self.model_results = {}
    
    def add_model_results(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> None:
        """
        Add results for a model.
        
        Args:
            model_name: Name of the model
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional)
        """
        evaluator = EvaluationMetrics()
        metrics = evaluator.calculate_all_metrics(y_true, y_pred, y_proba)
        self.model_results[model_name] = metrics
        
        logger.info(f"Added results for {model_name}")
    
    def compare_models(self) -> pd.DataFrame:
        """
        Compare models across key metrics.
        
        Returns:
            DataFrame with comparison results
        """
        if not self.model_results:
            raise ValueError("No model results to compare")
        
        # Key metrics to compare
        key_metrics = ['accuracy', 'precision', 'recall', 'f1', 'specificity']
        if 'auc_roc' in list(self.model_results.values())[0]:
            key_metrics.extend(['auc_roc', 'auc_pr'])
        
        # Create comparison DataFrame
        comparison_data = {}
        for metric in key_metrics:
            comparison_data[metric] = [
                self.model_results[model][metric]
                for model in self.model_results.keys()
            ]
        
        df = pd.DataFrame(comparison_data, index=list(self.model_results.keys()))
        return df
    
    def statistical_significance_test(
        self,
        model1: str,
        model2: str,
        metric: str = 'accuracy'
    ) -> Dict[str, float]:
        """
        Perform statistical significance test between two models.
        
        Args:
            model1: Name of first model
            model2: Name of second model
            metric: Metric to compare
            
        Returns:
            Dictionary with test results
        """
        if model1 not in self.model_results or model2 not in self.model_results:
            raise ValueError("Model results not found")
        
        # This is a simplified test - in practice, you'd need multiple runs
        # or bootstrap sampling for proper statistical testing
        value1 = self.model_results[model1][metric]
        value2 = self.model_results[model2][metric]
        
        # Placeholder for actual statistical test
        # In practice, use McNemar's test for paired predictions
        difference = abs(value1 - value2)
        
        return {
            'model1': model1,
            'model2': model2,
            'metric': metric,
            'value1': value1,
            'value2': value2,
            'difference': difference,
            'significant': difference > 0.01  # Placeholder threshold
        }
    
    def plot_model_comparison(
        self,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Plot model comparison.
        
        Args:
            save_path: Path to save the plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        df = self.compare_models()
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.ravel()
        
        # Plot key metrics
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
        
        for i, metric in enumerate(metrics_to_plot):
            if metric in df.columns:
                df[metric].plot(kind='bar', ax=axes[i], title=metric.capitalize())
                axes[i].set_ylabel('Score')
                axes[i].set_ylim(0, 1)
                axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model comparison plot saved to {save_path}")
        
        return fig
