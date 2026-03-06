#!/usr/bin/env python3
"""
Comprehensive Benchmarking System for Deepfake Detection Models

Inspired by timm's benchmarking approach but specialized for deepfake detection.
This script evaluates individual models and ensemble performance.

Usage:
    python scripts/evaluation/benchmark_deepfake_models.py --config config.yaml
"""

import os
import sys
import time
import argparse
import yaml
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json
import logging
from tqdm import tqdm

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


class DeepfakeBenchmark:
    """Comprehensive benchmarking system for deepfake detection models."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the benchmark system.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        
        # Initialize metrics calculator
        self.metrics = DeepfakeMetrics()
        
        logger.info(f"Benchmark initialized on device: {self.device}")
    
    def load_models(self) -> Dict[str, torch.nn.Module]:
        """Load all trained models for benchmarking."""
        models = {}
        model_dir = Path(self.config['paths']['model_dir'])
        
        # Load individual base models
        base_models = ['vit', 'deit', 'swin']
        
        for model_name in base_models:
            model_path = model_dir / 'base_models' / f'{model_name}_best.pth'
            if model_path.exists():
                try:
                    model = load_model_weights(model_name, str(model_path), self.device)
                    model.eval()
                    models[f'base_{model_name}'] = model
                    logger.info(f"✅ Loaded base model: {model_name}")
                except Exception as e:
                    logger.error(f"❌ Failed to load {model_name}: {e}")
            else:
                logger.warning(f"⚠️  Model not found: {model_path}")
        
        # Load ensemble model
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
                logger.info("✅ Loaded ensemble model")
            except Exception as e:
                logger.error(f"❌ Failed to load ensemble: {e}")
        
        return models
    
    def load_test_datasets(self) -> Dict[str, torch.utils.data.DataLoader]:
        """Load test datasets for benchmarking."""
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
            logger.info(f"✅ Loaded FaceForensics++ test set: {len(ff_dataset)} samples")
            
        except Exception as e:
            logger.error(f"❌ Failed to load FaceForensics++ test set: {e}")
        
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
            logger.info(f"✅ Loaded CelebDF test set: {len(celebdf_dataset)} samples")
            
        except Exception as e:
            logger.error(f"❌ Failed to load CelebDF test set: {e}")
        
        return datasets
    
    def benchmark_model(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        model_name: str,
        dataset_name: str
    ) -> Dict[str, Any]:
        """
        Benchmark a single model on a dataset.
        
        Args:
            model: Model to benchmark
            dataloader: Test data loader
            model_name: Name of the model
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary containing benchmark results
        """
        logger.info(f"Benchmarking {model_name} on {dataset_name}...")
        
        model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        inference_times = []
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc=f"Testing {model_name}")):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Measure inference time
                start_time = time.time()
                
                if 'ensemble' in model_name:
                    # Ensemble model returns different output format
                    outputs = model(images)
                else:
                    outputs = model(images)
                
                end_time = time.time()
                
                # Calculate probabilities and predictions
                if outputs.dim() == 1:
                    # Binary classification with single output
                    probabilities = torch.sigmoid(outputs)
                    predictions = (probabilities > 0.5).long()
                else:
                    # Multi-class output
                    probabilities = torch.softmax(outputs, dim=1)[:, 1]  # Probability of fake
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
        
        # Add timing information
        metrics_result.update({
            'avg_inference_time_ms': np.mean(inference_times) * 1000,
            'std_inference_time_ms': np.std(inference_times) * 1000,
            'total_samples': len(all_labels),
            'throughput_samples_per_sec': 1.0 / np.mean(inference_times)
        })
        
        logger.info(f"✅ {model_name} on {dataset_name}: "
                   f"Accuracy={metrics_result['accuracy']:.4f}, "
                   f"AUC={metrics_result['auc']:.4f}, "
                   f"Inference={metrics_result['avg_inference_time_ms']:.2f}ms")
        
        return metrics_result
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive benchmark on all models and datasets."""
        logger.info("Starting comprehensive deepfake detection benchmark...")
        
        # Load models and datasets
        models = self.load_models()
        datasets = self.load_test_datasets()
        
        if not models:
            raise ValueError("No models loaded for benchmarking")
        
        if not datasets:
            raise ValueError("No datasets loaded for benchmarking")
        
        # Run benchmarks
        results = {}
        
        for dataset_name, dataloader in datasets.items():
            results[dataset_name] = {}
            
            for model_name, model in models.items():
                try:
                    benchmark_result = self.benchmark_model(
                        model, dataloader, model_name, dataset_name
                    )
                    results[dataset_name][model_name] = benchmark_result
                    
                except Exception as e:
                    logger.error(f"❌ Error benchmarking {model_name} on {dataset_name}: {e}")
                    results[dataset_name][model_name] = {'error': str(e)}
        
        self.results = results
        return results
    
    def generate_benchmark_report(self, output_dir: str):
        """Generate comprehensive benchmark report."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save raw results
        results_file = output_dir / 'benchmark_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Generate CSV summary
        self._generate_csv_summary(output_dir)
        
        # Generate comparison tables
        self._generate_comparison_tables(output_dir)
        
        # Generate performance analysis
        self._generate_performance_analysis(output_dir)
        
        logger.info(f"📊 Benchmark report generated in: {output_dir}")
    
    def _generate_csv_summary(self, output_dir: Path):
        """Generate CSV summary similar to timm's benchmark results."""
        summary_data = []
        
        for dataset_name, dataset_results in self.results.items():
            for model_name, metrics in dataset_results.items():
                if 'error' not in metrics:
                    summary_data.append({
                        'model': model_name,
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
        csv_file = output_dir / 'deepfake_benchmark_summary.csv'
        df.to_csv(csv_file, index=False, float_format='%.4f')
        
        logger.info(f"📄 CSV summary saved: {csv_file}")
    
    def _generate_comparison_tables(self, output_dir: Path):
        """Generate detailed comparison tables."""
        for dataset_name, dataset_results in self.results.items():
            # Create comparison DataFrame
            comparison_data = []
            
            for model_name, metrics in dataset_results.items():
                if 'error' not in metrics:
                    comparison_data.append({
                        'Model': model_name.replace('_', ' ').title(),
                        'Accuracy': f"{metrics.get('accuracy', 0):.4f}",
                        'Precision': f"{metrics.get('precision', 0):.4f}",
                        'Recall': f"{metrics.get('recall', 0):.4f}",
                        'F1-Score': f"{metrics.get('f1_score', 0):.4f}",
                        'AUC': f"{metrics.get('auc', 0):.4f}",
                        'Inference (ms)': f"{metrics.get('avg_inference_time_ms', 0):.2f}",
                        'Throughput (samples/s)': f"{metrics.get('throughput_samples_per_sec', 0):.1f}"
                    })
            
            df = pd.DataFrame(comparison_data)
            
            # Save detailed comparison
            comparison_file = output_dir / f'{dataset_name}_detailed_comparison.csv'
            df.to_csv(comparison_file, index=False)
            
            logger.info(f"📊 Detailed comparison saved: {comparison_file}")
    
    def _generate_performance_analysis(self, output_dir: Path):
        """Generate performance analysis summary."""
        analysis = {
            'benchmark_summary': {
                'total_models_tested': 0,
                'total_datasets': len(self.results),
                'best_overall_model': None,
                'best_accuracy': 0,
                'best_auc': 0,
                'fastest_model': None,
                'fastest_inference_ms': float('inf')
            },
            'dataset_analysis': {},
            'model_analysis': {}
        }
        
        all_results = []
        
        # Collect all results
        for dataset_name, dataset_results in self.results.items():
            analysis['dataset_analysis'][dataset_name] = {
                'models_tested': len([m for m in dataset_results if 'error' not in dataset_results[m]]),
                'best_model': None,
                'best_auc': 0
            }
            
            for model_name, metrics in dataset_results.items():
                if 'error' not in metrics:
                    all_results.append({
                        'dataset': dataset_name,
                        'model': model_name,
                        **metrics
                    })
                    
                    # Update dataset best
                    if metrics.get('auc', 0) > analysis['dataset_analysis'][dataset_name]['best_auc']:
                        analysis['dataset_analysis'][dataset_name]['best_auc'] = metrics.get('auc', 0)
                        analysis['dataset_analysis'][dataset_name]['best_model'] = model_name
                    
                    # Update overall best
                    if metrics.get('auc', 0) > analysis['benchmark_summary']['best_auc']:
                        analysis['benchmark_summary']['best_auc'] = metrics.get('auc', 0)
                        analysis['benchmark_summary']['best_overall_model'] = f"{model_name} on {dataset_name}"
                    
                    if metrics.get('accuracy', 0) > analysis['benchmark_summary']['best_accuracy']:
                        analysis['benchmark_summary']['best_accuracy'] = metrics.get('accuracy', 0)
                    
                    # Update fastest
                    inference_time = metrics.get('avg_inference_time_ms', float('inf'))
                    if inference_time < analysis['benchmark_summary']['fastest_inference_ms']:
                        analysis['benchmark_summary']['fastest_inference_ms'] = inference_time
                        analysis['benchmark_summary']['fastest_model'] = f"{model_name} on {dataset_name}"
        
        analysis['benchmark_summary']['total_models_tested'] = len(set(r['model'] for r in all_results))
        
        # Save analysis
        analysis_file = output_dir / 'performance_analysis.json'
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        logger.info(f"📈 Performance analysis saved: {analysis_file}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark deepfake detection models')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Configuration file')
    parser.add_argument('--output-dir', type=str, default='results/benchmarks',
                        help='Output directory for benchmark results')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("="*60)
    print("DEEPFAKE DETECTION MODEL BENCHMARK")
    print("="*60)
    print("Inspired by timm's benchmarking approach")
    print("Individual model and ensemble performance evaluation")
    print("="*60)
    
    # Initialize and run benchmark
    benchmark = DeepfakeBenchmark(config)
    
    try:
        # Run comprehensive benchmark
        results = benchmark.run_comprehensive_benchmark()
        
        # Generate reports
        benchmark.generate_benchmark_report(args.output_dir)
        
        print("\n🎉 Benchmark completed successfully!")
        print(f"📊 Results saved to: {args.output_dir}")
        print("\nKey files generated:")
        print("  - deepfake_benchmark_summary.csv (main results)")
        print("  - benchmark_results.json (detailed results)")
        print("  - performance_analysis.json (analysis summary)")
        
    except Exception as e:
        logger.error(f"❌ Benchmark failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
