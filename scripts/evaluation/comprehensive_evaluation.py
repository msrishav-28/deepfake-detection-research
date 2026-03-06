#!/usr/bin/env python3
"""
Comprehensive Evaluation System for Deepfake Detection Models

This script provides a professional evaluation framework for deepfake detection
research, including individual model assessment, ensemble evaluation, and
explainability analysis through Grad-CAM visualizations.

Usage:
    python scripts/evaluation/comprehensive_evaluation.py --config config.yaml --explainability
"""

import os
import sys
import time
import argparse
import yaml
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json
import logging
from tqdm import tqdm

# Grad-CAM for explainability analysis
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from deepfake_detection.models.ensemble import StackedEnsemble
from deepfake_detection.data.datasets import FaceForensicsDataset, CelebDFDataset
from deepfake_detection.evaluation.metrics import DeepfakeMetrics
from deepfake_detection.utils.model_utils import load_model_weights

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DeepfakeEvaluationFramework:
    """Professional evaluation framework for deepfake detection research."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the evaluation framework."""
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.metrics = DeepfakeMetrics()
        
        # Model configurations
        self.model_configs = {
            'vit': 'Vision Transformer (ViT-Base)',
            'deit': 'Data-efficient Image Transformer (DeiT-Base)',
            'swin': 'Swin Transformer (Swin-Base)',
            'ensemble': 'Stacked Ensemble'
        }
        
        logger.info("Deepfake Detection Evaluation Framework initialized")
        logger.info(f"Device: {self.device}")
    
    def load_trained_models(self) -> Dict[str, torch.nn.Module]:
        """Load all trained models for evaluation."""
        models = {}
        model_dir = Path(self.config['paths']['model_dir'])
        
        logger.info("Loading trained models...")
        
        # Load individual base models
        base_models = ['vit', 'deit', 'swin']
        
        for model_name in base_models:
            model_path = model_dir / 'base_models' / f'{model_name}_best.pth'
            if model_path.exists():
                try:
                    model = load_model_weights(model_name, str(model_path), self.device)
                    model.eval()
                    models[model_name] = model
                    logger.info(f"✓ Loaded {self.model_configs[model_name]}")
                except Exception as e:
                    logger.error(f"✗ Failed to load {model_name}: {e}")
            else:
                logger.warning(f"Model not found: {model_path}")
        
        # Load stacked ensemble
        ensemble_path = model_dir / 'ensemble' / 'stacked_ensemble.pth'
        if ensemble_path.exists():
            try:
                ensemble = StackedEnsemble(
                    base_models=list(models.values()),
                    meta_learner_type='logistic_regression'
                )
                ensemble.load_state_dict(torch.load(ensemble_path, map_location=self.device))
                ensemble.eval()
                models['ensemble'] = ensemble
                logger.info(f"✓ Loaded {self.model_configs['ensemble']}")
            except Exception as e:
                logger.error(f"✗ Failed to load ensemble: {e}")
        
        logger.info(f"Successfully loaded {len(models)} models")
        return models
    
    def load_test_datasets(self) -> Dict[str, torch.utils.data.DataLoader]:
        """Load test datasets for evaluation."""
        datasets = {}
        
        # FaceForensics++ test set
        try:
            ff_dataset = FaceForensicsDataset(
                data_dir=self.config['data']['faceforensics']['path'],
                split='test',
                use_extracted_faces=True,
                image_size=self.config['data']['preprocessing']['face_crop_size']
            )
            
            ff_loader = torch.utils.data.DataLoader(
                ff_dataset,
                batch_size=32,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
            
            datasets['faceforensics'] = ff_loader
            logger.info(f"✓ Loaded FaceForensics++ test set: {len(ff_dataset)} samples")
            
        except Exception as e:
            logger.error(f"✗ Failed to load FaceForensics++ test set: {e}")
        
        # CelebDF test set
        try:
            celebdf_dataset = CelebDFDataset(
                data_dir=self.config['data']['celebdf']['path'],
                split='test',
                use_extracted_faces=True,
                image_size=self.config['data']['preprocessing']['face_crop_size']
            )
            
            celebdf_loader = torch.utils.data.DataLoader(
                celebdf_dataset,
                batch_size=32,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
            
            datasets['celebdf'] = celebdf_loader
            logger.info(f"✓ Loaded CelebDF test set: {len(celebdf_dataset)} samples")
            
        except Exception as e:
            logger.error(f"✗ Failed to load CelebDF test set: {e}")
        
        return datasets
    
    def evaluate_model(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        model_name: str,
        dataset_name: str
    ) -> Dict[str, Any]:
        """
        Evaluate a single model on a dataset.
        
        Args:
            model: Model to evaluate
            dataloader: Test data loader
            model_name: Name of the model
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info(f"Evaluating {self.model_configs.get(model_name, model_name)} on {dataset_name}")
        
        model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        inference_times = []
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc=f"Evaluating {model_name}")):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Measure inference time
                start_time = time.time()
                
                if 'ensemble' in model_name:
                    outputs = model(images)
                else:
                    outputs = model(images)
                
                end_time = time.time()
                
                # Process outputs
                if outputs.dim() == 1:
                    probabilities = torch.sigmoid(outputs)
                    predictions = (probabilities > 0.5).long()
                else:
                    probabilities = torch.softmax(outputs, dim=1)[:, 1]
                    predictions = torch.argmax(outputs, dim=1)
                
                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
                # Store inference time (per sample)
                batch_time = (end_time - start_time) / len(images)
                inference_times.extend([batch_time] * len(images))
        
        # Calculate comprehensive metrics
        metrics_result = self.metrics.calculate_all_metrics(
            y_true=all_labels,
            y_pred=all_predictions,
            y_prob=all_probabilities
        )
        
        # Add performance metrics
        metrics_result.update({
            'avg_inference_time_ms': np.mean(inference_times) * 1000,
            'std_inference_time_ms': np.std(inference_times) * 1000,
            'total_samples': len(all_labels),
            'throughput_samples_per_sec': 1.0 / np.mean(inference_times)
        })
        
        logger.info(f"✓ {model_name}: Accuracy={metrics_result['accuracy']:.4f}, "
                   f"AUC={metrics_result['auc']:.4f}, "
                   f"Inference={metrics_result['avg_inference_time_ms']:.2f}ms")
        
        return metrics_result
    
    def generate_explainability_analysis(
        self,
        model: torch.nn.Module,
        sample_images: torch.Tensor,
        sample_labels: torch.Tensor,
        model_name: str,
        output_dir: Path,
        num_samples: int = 8
    ):
        """
        Generate explainability analysis using Grad-CAM.
        
        Args:
            model: Model to analyze
            sample_images: Sample images for analysis
            sample_labels: Ground truth labels
            model_name: Name of the model
            output_dir: Output directory for visualizations
            num_samples: Number of samples to analyze
        """
        
        logger.info(f"Generating explainability analysis for {model_name}")
        
        if 'ensemble' in model_name:
            logger.info("Skipping explainability for ensemble model (complex architecture)")
            return
        
        try:
            # Get target layers for Grad-CAM based on architecture
            if 'vit' in model_name or 'deit' in model_name:
                target_layers = [model.blocks[-1].norm1]
            elif 'swin' in model_name:
                target_layers = [model.layers[-1].blocks[-1].norm1]
            else:
                logger.warning(f"Unknown architecture for {model_name}, skipping explainability")
                return
            
            # Initialize Grad-CAM
            cam = GradCAM(model=model, target_layers=target_layers)
            
            # Create visualization
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            fig.suptitle(f'Explainability Analysis: {self.model_configs.get(model_name, model_name)}', 
                        fontsize=16, fontweight='bold')
            
            for i in range(min(num_samples, len(sample_images))):
                if i >= 8:
                    break
                
                image = sample_images[i:i+1].to(self.device)
                label = sample_labels[i].item()
                
                # Generate Grad-CAM
                targets = [ClassifierOutputTarget(1)]  # Focus on 'fake' class
                grayscale_cam = cam(input_tensor=image, targets=targets)
                grayscale_cam = grayscale_cam[0, :]
                
                # Convert image for visualization
                img_np = image[0].cpu().permute(1, 2, 0).numpy()
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
                
                # Create heatmap overlay
                visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
                
                # Plot
                row = i // 4
                col = i % 4
                axes[row, col].imshow(visualization)
                
                # Title with ground truth
                truth = "Real" if label == 0 else "Fake"
                axes[row, col].set_title(f'Ground Truth: {truth}', fontsize=10)
                axes[row, col].axis('off')
            
            # Hide unused subplots
            for i in range(num_samples, 8):
                row = i // 4
                col = i % 4
                axes[row, col].axis('off')
            
            plt.tight_layout()
            
            # Save explainability analysis
            explainability_file = output_dir / f'explainability_{model_name}.png'
            plt.savefig(explainability_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✓ Explainability analysis saved: {explainability_file}")
            
        except Exception as e:
            logger.error(f"✗ Explainability analysis failed for {model_name}: {e}")
    
    def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """Run comprehensive evaluation on all models and datasets."""
        logger.info("Starting comprehensive deepfake detection evaluation")
        
        # Load models and datasets
        models = self.load_trained_models()
        datasets = self.load_test_datasets()
        
        if not models:
            raise ValueError("No models loaded for evaluation")
        
        if not datasets:
            raise ValueError("No datasets loaded for evaluation")
        
        # Run evaluations
        results = {}
        
        for dataset_name, dataloader in datasets.items():
            results[dataset_name] = {}
            
            for model_name, model in models.items():
                try:
                    evaluation_result = self.evaluate_model(
                        model, dataloader, model_name, dataset_name
                    )
                    results[dataset_name][model_name] = evaluation_result
                    
                except Exception as e:
                    logger.error(f"✗ Error evaluating {model_name} on {dataset_name}: {e}")
                    results[dataset_name][model_name] = {'error': str(e)}
        
        return results
    
    def generate_evaluation_report(self, results: Dict, output_dir: Path, include_explainability: bool = False):
        """Generate comprehensive evaluation report."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Generating evaluation report")
        
        # Save raw results
        results_file = output_dir / 'evaluation_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Generate CSV summary (timm-style)
        self._generate_csv_summary(results, output_dir)
        
        # Generate performance comparison
        self._generate_performance_comparison(results, output_dir)
        
        # Generate statistical analysis
        self._generate_statistical_analysis(results, output_dir)
        
        # Generate explainability analysis if requested
        if include_explainability:
            self._generate_explainability_report(output_dir)
        
        logger.info(f"✓ Evaluation report generated in: {output_dir}")
    
    def _generate_csv_summary(self, results: Dict, output_dir: Path):
        """Generate CSV summary similar to timm's benchmark results."""
        summary_data = []
        
        for dataset_name, dataset_results in results.items():
            for model_name, metrics in dataset_results.items():
                if 'error' not in metrics:
                    summary_data.append({
                        'model': self.model_configs.get(model_name, model_name),
                        'dataset': dataset_name,
                        'accuracy': metrics.get('accuracy', 0),
                        'precision': metrics.get('precision', 0),
                        'recall': metrics.get('recall', 0),
                        'f1_score': metrics.get('f1_score', 0),
                        'auc': metrics.get('auc', 0),
                        'avg_inference_time_ms': metrics.get('avg_inference_time_ms', 0),
                        'throughput_samples_per_sec': metrics.get('throughput_samples_per_sec', 0),
                        'total_samples': metrics.get('total_samples', 0)
                    })
        
        df = pd.DataFrame(summary_data)
        
        # Sort by AUC score (descending)
        df = df.sort_values(['dataset', 'auc'], ascending=[True, False])
        
        # Save CSV
        csv_file = output_dir / 'deepfake_detection_benchmark.csv'
        df.to_csv(csv_file, index=False, float_format='%.4f')
        
        logger.info(f"✓ CSV benchmark summary saved: {csv_file}")
    
    def _generate_performance_comparison(self, results: Dict, output_dir: Path):
        """Generate performance comparison visualizations."""
        # Create performance comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Deepfake Detection Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # Prepare data for plotting
        plot_data = []
        for dataset_name, dataset_results in results.items():
            for model_name, metrics in dataset_results.items():
                if 'error' not in metrics:
                    plot_data.append({
                        'Model': self.model_configs.get(model_name, model_name),
                        'Dataset': dataset_name.title(),
                        'Accuracy': metrics.get('accuracy', 0),
                        'AUC': metrics.get('auc', 0),
                        'F1_Score': metrics.get('f1_score', 0),
                        'Inference_Time': metrics.get('avg_inference_time_ms', 0)
                    })
        
        df_plot = pd.DataFrame(plot_data)
        
        if not df_plot.empty:
            # Accuracy comparison
            sns.barplot(data=df_plot, x='Model', y='Accuracy', hue='Dataset', ax=axes[0,0])
            axes[0,0].set_title('Model Accuracy Comparison')
            axes[0,0].tick_params(axis='x', rotation=45)
            
            # AUC comparison
            sns.barplot(data=df_plot, x='Model', y='AUC', hue='Dataset', ax=axes[0,1])
            axes[0,1].set_title('AUC Score Comparison')
            axes[0,1].tick_params(axis='x', rotation=45)
            
            # F1-Score comparison
            sns.barplot(data=df_plot, x='Model', y='F1_Score', hue='Dataset', ax=axes[1,0])
            axes[1,0].set_title('F1-Score Comparison')
            axes[1,0].tick_params(axis='x', rotation=45)
            
            # Inference time comparison
            sns.barplot(data=df_plot, x='Model', y='Inference_Time', hue='Dataset', ax=axes[1,1])
            axes[1,1].set_title('Inference Time Comparison (ms)')
            axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save performance comparison
        comparison_file = output_dir / 'performance_comparison.png'
        plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Performance comparison saved: {comparison_file}")
    
    def _generate_statistical_analysis(self, results: Dict, output_dir: Path):
        """Generate statistical analysis of results."""
        analysis = {
            'summary_statistics': {},
            'model_rankings': {},
            'ensemble_analysis': {},
            'dataset_analysis': {}
        }
        
        # Calculate summary statistics
        all_results = []
        for dataset_name, dataset_results in results.items():
            for model_name, metrics in dataset_results.items():
                if 'error' not in metrics:
                    all_results.append({
                        'dataset': dataset_name,
                        'model': model_name,
                        'model_display': self.model_configs.get(model_name, model_name),
                        **metrics
                    })
        
        if all_results:
            df_analysis = pd.DataFrame(all_results)
            
            # Model rankings
            for metric in ['accuracy', 'auc', 'f1_score']:
                if metric in df_analysis.columns:
                    ranking = df_analysis.groupby('model_display')[metric].mean().sort_values(ascending=False)
                    analysis['model_rankings'][metric] = ranking.to_dict()
            
            # Ensemble analysis
            ensemble_results = df_analysis[df_analysis['model'] == 'ensemble']
            individual_results = df_analysis[df_analysis['model'] != 'ensemble']
            
            if not ensemble_results.empty and not individual_results.empty:
                ensemble_avg = ensemble_results['accuracy'].mean()
                individual_avg = individual_results['accuracy'].mean()
                best_individual = individual_results['accuracy'].max()
                
                analysis['ensemble_analysis'] = {
                    'ensemble_accuracy': ensemble_avg,
                    'average_individual_accuracy': individual_avg,
                    'best_individual_accuracy': best_individual,
                    'improvement_over_average': ensemble_avg - individual_avg,
                    'improvement_over_best': ensemble_avg - best_individual
                }
        
        # Save statistical analysis
        analysis_file = output_dir / 'statistical_analysis.json'
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        logger.info(f"✓ Statistical analysis saved: {analysis_file}")
    
    def _generate_explainability_report(self, output_dir: Path):
        """Generate explainability analysis report."""
        explainability_files = list(output_dir.glob('explainability_*.png'))
        
        if explainability_files:
            logger.info(f"✓ Generated {len(explainability_files)} explainability visualizations")
        else:
            logger.info("No explainability visualizations found")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive deepfake detection evaluation')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Configuration file')
    parser.add_argument('--output-dir', type=str, default='results/evaluation',
                        help='Output directory for evaluation results')
    parser.add_argument('--explainability', action='store_true',
                        help='Include explainability analysis (Grad-CAM)')
    parser.add_argument('--explainability-samples', type=int, default=8,
                        help='Number of samples for explainability analysis')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("="*60)
    print("DEEPFAKE DETECTION MODEL EVALUATION")
    print("="*60)
    print("Professional evaluation framework for deepfake detection research")
    print("Individual models and ensemble performance assessment")
    if args.explainability:
        print("Including explainability analysis through Grad-CAM")
    print("="*60)
    
    # Initialize evaluation framework
    evaluator = DeepfakeEvaluationFramework(config)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load models
        models = evaluator.load_trained_models()
        
        if not models:
            logger.error("No models loaded. Check model paths in configuration.")
            return 1
        
        # Load datasets
        datasets = evaluator.load_test_datasets()
        
        if not datasets:
            logger.error("No datasets loaded. Check data paths in configuration.")
            return 1
        
        # Run comprehensive evaluation
        results = evaluator.run_comprehensive_evaluation()
        
        # Generate explainability analysis if requested
        if args.explainability:
            logger.info("Generating explainability analysis...")
            
            for dataset_name, dataloader in datasets.items():
                # Get sample batch for explainability
                sample_batch = next(iter(dataloader))
                sample_images, sample_labels = sample_batch
                
                for model_name, model in models.items():
                    if model_name != 'ensemble':  # Skip ensemble for explainability
                        evaluator.generate_explainability_analysis(
                            model, sample_images, sample_labels,
                            model_name, output_dir, args.explainability_samples
                        )
        
        # Generate comprehensive report
        evaluator.generate_evaluation_report(results, output_dir, args.explainability)
        
        print("\n" + "="*60)
        print("EVALUATION COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Results saved to: {output_dir}")
        print("\nGenerated files:")
        print("  - deepfake_detection_benchmark.csv (main results)")
        print("  - evaluation_results.json (detailed metrics)")
        print("  - performance_comparison.png (visualizations)")
        print("  - statistical_analysis.json (statistical summary)")
        if args.explainability:
            print("  - explainability_*.png (Grad-CAM visualizations)")
        
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
