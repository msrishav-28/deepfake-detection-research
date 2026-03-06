#!/usr/bin/env python3
"""
Comprehensive evaluation script for deepfake detection models.

This script evaluates individual base models and the ensemble on the test set,
generates detailed metrics, and creates visualizations.

Usage:
    python scripts/evaluation/evaluate_models.py --config config.yaml
    python scripts/evaluation/evaluate_models.py --ensemble-dir models/ensemble --data-dir data/processed
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from deepfake_detection.models.model_factory import ModelFactory
from deepfake_detection.models.ensemble import StackedEnsemble
from deepfake_detection.data.timm_integration import DeepfakeDataModule
from deepfake_detection.evaluation.metrics import EvaluationMetrics, ModelComparator
from deepfake_detection.evaluation.explainability import ExplainabilityAnalyzer
from deepfake_detection.utils.training_utils import setup_logging

logger = logging.getLogger(__name__)


def load_ensemble(ensemble_dir: str, config: dict, device: torch.device) -> StackedEnsemble:
    """Load the trained ensemble."""
    logger.info(f"Loading ensemble from {ensemble_dir}")
    
    # Create base models
    factory = ModelFactory(config)
    base_models = factory.create_all_base_models(device)
    
    # Create ensemble
    ensemble = StackedEnsemble(base_models, device=device)
    
    # Load ensemble weights and meta-learner
    ensemble.load_ensemble(ensemble_dir)
    ensemble.eval()
    
    logger.info("Ensemble loaded successfully")
    return ensemble


def evaluate_single_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    model_name: str
) -> tuple:
    """
    Evaluate a single model.
    
    Returns:
        Tuple of (y_true, y_pred, y_proba)
    """
    logger.info(f"Evaluating {model_name}")
    
    model.eval()
    all_predictions = []
    all_probabilities = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, desc=f"Evaluating {model_name}"):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            # Get model outputs
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = outputs.argmax(dim=1)
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    return np.array(all_targets), np.array(all_predictions), np.array(all_probabilities)


def evaluate_ensemble(
    ensemble: StackedEnsemble,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device
) -> tuple:
    """
    Evaluate the ensemble model.
    
    Returns:
        Tuple of (y_true, y_pred, y_proba, individual_predictions)
    """
    logger.info("Evaluating ensemble")
    
    ensemble.eval()
    all_predictions = []
    all_probabilities = []
    all_targets = []
    all_individual_predictions = {name: [] for name in ensemble.model_names}
    
    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, desc="Evaluating ensemble"):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            # Get ensemble predictions
            predictions, probabilities = ensemble.predict(inputs)
            
            # Get individual model predictions
            meta_features, individual_preds = ensemble.extract_meta_features(
                inputs, return_individual_predictions=True
            )
            
            # Store results
            all_predictions.extend(predictions)
            all_probabilities.extend(probabilities)
            all_targets.extend(targets.cpu().numpy())
            
            # Store individual predictions
            for model_name in ensemble.model_names:
                all_individual_predictions[model_name].extend(individual_preds[model_name])
    
    # Convert individual predictions to numpy arrays
    for model_name in all_individual_predictions:
        all_individual_predictions[model_name] = np.vstack(all_individual_predictions[model_name])
    
    return (
        np.array(all_targets),
        np.array(all_predictions),
        np.array(all_probabilities),
        all_individual_predictions
    )


def create_evaluation_report(
    results: dict,
    save_dir: str,
    config: dict
) -> None:
    """Create comprehensive evaluation report."""
    logger.info("Creating evaluation report")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize model comparator
    comparator = ModelComparator()
    
    # Add results for each model
    for model_name, (y_true, y_pred, y_proba) in results.items():
        comparator.add_model_results(model_name, y_true, y_pred, y_proba)
    
    # Generate comparison DataFrame
    comparison_df = comparator.compare_models()
    
    # Save comparison table
    comparison_df.to_csv(os.path.join(save_dir, 'model_comparison.csv'))
    comparison_df.to_html(os.path.join(save_dir, 'model_comparison.html'))
    
    # Create detailed metrics for each model
    detailed_results = {}
    
    for model_name, (y_true, y_pred, y_proba) in results.items():
        evaluator = EvaluationMetrics()
        metrics = evaluator.calculate_all_metrics(y_true, y_pred, y_proba)
        detailed_results[model_name] = metrics
        
        # Save individual model metrics
        model_save_dir = os.path.join(save_dir, model_name)
        os.makedirs(model_save_dir, exist_ok=True)
        
        # Save metrics
        evaluator.save_metrics_to_file(
            os.path.join(model_save_dir, 'metrics.json')
        )
        
        # Plot confusion matrix
        evaluator.plot_confusion_matrix(
            save_path=os.path.join(model_save_dir, 'confusion_matrix.png')
        )
        
        # Plot ROC curve if probabilities available
        if y_proba is not None:
            evaluator.plot_roc_curve(
                save_path=os.path.join(model_save_dir, 'roc_curve.png')
            )
            evaluator.plot_precision_recall_curve(
                save_path=os.path.join(model_save_dir, 'pr_curve.png')
            )
    
    # Save detailed results
    with open(os.path.join(save_dir, 'detailed_results.json'), 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    # Create model comparison plot
    comparator.plot_model_comparison(
        save_path=os.path.join(save_dir, 'model_comparison.png')
    )
    
    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(comparison_df.round(4))
    
    # Find best model
    best_model = comparison_df['f1'].idxmax()
    best_f1 = comparison_df.loc[best_model, 'f1']
    print(f"\nBest performing model: {best_model} (F1-Score: {best_f1:.4f})")
    
    logger.info(f"Evaluation report saved to {save_dir}")


def generate_explainability_analysis(
    ensemble: StackedEnsemble,
    data_loader: torch.utils.data.DataLoader,
    save_dir: str,
    num_samples: int = 20
) -> None:
    """Generate explainability analysis with Grad-CAM."""
    logger.info("Generating explainability analysis")
    
    # Create explainability analyzer
    analyzer = ExplainabilityAnalyzer(ensemble.base_models)
    
    # Collect samples for analysis
    images = []
    input_tensors = []
    true_labels = []
    
    sample_count = 0
    for inputs, targets in data_loader:
        batch_size = inputs.size(0)
        
        for i in range(batch_size):
            if sample_count >= num_samples:
                break
            
            # Convert tensor back to image for visualization
            input_tensor = inputs[i:i+1]  # Keep batch dimension
            
            # Denormalize for visualization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image_tensor = inputs[i] * std + mean
            image_tensor = torch.clamp(image_tensor, 0, 1)
            
            # Convert to numpy
            image = image_tensor.permute(1, 2, 0).numpy()
            image = (image * 255).astype(np.uint8)
            
            images.append(image)
            input_tensors.append(input_tensor)
            true_labels.append(targets[i].item())
            
            sample_count += 1
        
        if sample_count >= num_samples:
            break
    
    # Perform batch analysis
    explainability_dir = os.path.join(save_dir, 'explainability')
    analysis_results = analyzer.batch_analysis(
        images, input_tensors, true_labels, explainability_dir
    )
    
    # Save analysis summary
    summary = {
        'total_samples': len(analysis_results),
        'agreement_stats': {},
        'confidence_stats': {}
    }
    
    # Aggregate statistics
    agreement_ratios = [r['agreement_analysis']['agreement_ratio'] for r in analysis_results]
    mean_confidences = [r['confidence_analysis']['mean_confidence'] for r in analysis_results]
    
    summary['agreement_stats'] = {
        'mean_agreement_ratio': float(np.mean(agreement_ratios)),
        'std_agreement_ratio': float(np.std(agreement_ratios))
    }
    
    summary['confidence_stats'] = {
        'mean_confidence': float(np.mean(mean_confidences)),
        'std_confidence': float(np.std(mean_confidences))
    }
    
    # Save summary
    with open(os.path.join(explainability_dir, 'analysis_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Explainability analysis saved to {explainability_dir}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate deepfake detection models')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--ensemble-dir', type=str,
                        help='Directory containing trained ensemble')
    parser.add_argument('--data-dir', type=str,
                        help='Directory containing processed datasets')
    parser.add_argument('--save-dir', type=str,
                        help='Directory to save evaluation results')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cuda, cpu, or auto)')
    parser.add_argument('--explainability', action='store_true',
                        help='Generate explainability analysis')
    parser.add_argument('--num-samples', type=int, default=20,
                        help='Number of samples for explainability analysis')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    setup_logging()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Set paths from config if not provided
    if args.ensemble_dir is None:
        args.ensemble_dir = os.path.join(config['paths']['models_dir'], 'ensemble')
    
    if args.data_dir is None:
        args.data_dir = config['paths']['data_dir']
    
    if args.save_dir is None:
        args.save_dir = os.path.join(config['paths']['results_dir'], 'evaluation')
    
    # Setup data module
    data_module = DeepfakeDataModule(config, args.data_dir)
    data_module.setup('faceforensics')
    
    # Get test data loader
    test_loader, _ = data_module.get_loader('test')
    
    if len(test_loader.dataset) == 0:
        logger.error("Test dataset is empty. Please check data preparation.")
        return
    
    # Load ensemble
    ensemble = load_ensemble(args.ensemble_dir, config, device)
    
    # Evaluate ensemble
    y_true, y_pred_ensemble, y_proba_ensemble, individual_preds = evaluate_ensemble(
        ensemble, test_loader, device
    )
    
    # Prepare results dictionary
    results = {
        'ensemble': (y_true, y_pred_ensemble, y_proba_ensemble)
    }
    
    # Add individual model results
    for model_name in ensemble.model_names:
        model_proba = individual_preds[model_name]
        model_pred = np.argmax(model_proba, axis=1)
        results[model_name] = (y_true, model_pred, model_proba)
    
    # Create evaluation report
    create_evaluation_report(results, args.save_dir, config)
    
    # Generate explainability analysis if requested
    if args.explainability:
        generate_explainability_analysis(
            ensemble, test_loader, args.save_dir, args.num_samples
        )
    
    logger.info("Evaluation completed successfully!")


if __name__ == '__main__':
    main()
