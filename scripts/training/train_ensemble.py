#!/usr/bin/env python3
"""
Training script for the stacked ensemble meta-learner.

This script generates meta-features from trained base models using the hold-out set
and trains a meta-learner to optimally combine their predictions.

Usage:
    python scripts/training/train_ensemble.py --config config.yaml
    python scripts/training/train_ensemble.py --models-dir models/base_models --data-dir data/processed
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
import pickle

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from deepfake_detection.models.model_factory import ModelFactory
from deepfake_detection.models.ensemble import MetaLearner, StackedEnsemble, create_stacked_ensemble
from deepfake_detection.data.timm_integration import DeepfakeDataModule
from deepfake_detection.utils.training_utils import setup_logging, calculate_metrics

logger = logging.getLogger(__name__)


def load_trained_models(
    models_dir: str,
    config: dict,
    device: torch.device
) -> dict:
    """
    Load trained base models from checkpoints.
    
    Args:
        models_dir: Directory containing model checkpoints
        config: Configuration dictionary
        device: Device to load models on
        
    Returns:
        Dictionary of loaded models
    """
    factory = ModelFactory(config)
    models = {}
    
    # Expected model files
    model_files = {
        'vit': 'vit.pth',
        'deit': 'deit.pth',
        'swin': 'swin.pth'
    }
    
    for model_type, filename in model_files.items():
        checkpoint_path = os.path.join(models_dir, filename)
        
        if os.path.exists(checkpoint_path):
            # Create model
            model = factory.create_model(model_type, device)
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Handle different checkpoint formats
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            model.model.load_state_dict(state_dict)
            model.eval()
            
            models[model_type] = model
            logger.info(f"Loaded {model_type} model from {checkpoint_path}")
        else:
            logger.warning(f"Model checkpoint not found: {checkpoint_path}")
    
    if len(models) == 0:
        raise ValueError("No trained models found. Please train base models first.")
    
    logger.info(f"Loaded {len(models)} base models")
    return models


def generate_meta_features(
    models: dict,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device
) -> tuple:
    """
    Generate meta-features from base models.
    
    Args:
        models: Dictionary of trained models
        data_loader: Data loader for the hold-out set
        device: Device to run inference on
        
    Returns:
        Tuple of (meta_features, targets, individual_predictions)
    """
    logger.info("Generating meta-features from base models")
    
    all_meta_features = []
    all_targets = []
    individual_predictions = {model_name: [] for model_name in models.keys()}
    
    # Set all models to evaluation mode
    for model in models.values():
        model.eval()
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(data_loader, desc="Generating meta-features")):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            batch_meta_features = []
            
            # Get predictions from each model
            for model_name, model in models.items():
                outputs = model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                
                # Store individual predictions
                individual_predictions[model_name].append(probabilities.cpu().numpy())
                
                # Add to meta-features
                batch_meta_features.append(probabilities.cpu().numpy())
            
            # Concatenate meta-features from all models
            batch_meta_features = np.concatenate(batch_meta_features, axis=1)
            all_meta_features.append(batch_meta_features)
            all_targets.append(targets.cpu().numpy())
    
    # Concatenate all batches
    meta_features = np.vstack(all_meta_features)
    targets = np.concatenate(all_targets)
    
    # Concatenate individual predictions
    for model_name in individual_predictions:
        individual_predictions[model_name] = np.vstack(individual_predictions[model_name])
    
    logger.info(f"Generated meta-features: {meta_features.shape}")
    logger.info(f"Target distribution: {np.bincount(targets)}")
    
    return meta_features, targets, individual_predictions


def evaluate_base_models(
    individual_predictions: dict,
    targets: np.ndarray
) -> dict:
    """
    Evaluate individual base model performance.
    
    Args:
        individual_predictions: Dictionary of model predictions
        targets: True targets
        
    Returns:
        Dictionary of evaluation results
    """
    results = {}
    
    for model_name, predictions in individual_predictions.items():
        # Get predicted labels
        predicted_labels = np.argmax(predictions, axis=1)
        
        # Calculate metrics
        metrics = calculate_metrics(targets.tolist(), predicted_labels.tolist())
        results[model_name] = metrics
        
        logger.info(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}, "
                   f"F1: {metrics['f1']:.4f}")
    
    return results


def train_meta_learner(
    meta_features: np.ndarray,
    targets: np.ndarray,
    config: dict
) -> MetaLearner:
    """
    Train the meta-learner.
    
    Args:
        meta_features: Meta-features from base models
        targets: Target labels
        config: Configuration dictionary
        
    Returns:
        Trained meta-learner
    """
    logger.info("Training meta-learner")
    
    # Get meta-learner configuration
    ensemble_config = config.get('models', {}).get('ensemble', {})
    meta_learner_type = ensemble_config.get('meta_learner', 'LogisticRegression')
    
    # Convert to sklearn-compatible name
    if meta_learner_type == 'LogisticRegression':
        meta_learner_type = 'logistic_regression'
    elif meta_learner_type == 'RandomForest':
        meta_learner_type = 'random_forest'
    
    # Create and train meta-learner
    meta_learner = MetaLearner(model_type=meta_learner_type)
    
    # Get cross-validation folds from config
    cv_folds = config.get('training', {}).get('ensemble', {}).get('cv_folds', 5)
    
    # Train meta-learner
    cv_results = meta_learner.fit(meta_features, targets, cv_folds=cv_folds)
    
    logger.info(f"Meta-learner training completed")
    logger.info(f"Cross-validation accuracy: {cv_results['cv_mean']:.4f} ± {cv_results['cv_std']:.4f}")
    
    return meta_learner


def evaluate_ensemble(
    ensemble: StackedEnsemble,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device
) -> dict:
    """
    Evaluate the complete ensemble.
    
    Args:
        ensemble: Trained ensemble
        data_loader: Data loader for evaluation
        device: Device to run evaluation on
        
    Returns:
        Evaluation metrics
    """
    logger.info("Evaluating ensemble")
    
    ensemble.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, desc="Evaluating ensemble"):
            inputs = inputs.to(device, non_blocking=True)
            
            # Get ensemble predictions
            predicted_labels, _ = ensemble.predict(inputs)
            
            all_predictions.extend(predicted_labels.tolist())
            all_targets.extend(targets.numpy().tolist())
    
    # Calculate metrics
    metrics = calculate_metrics(all_targets, all_predictions)
    
    logger.info(f"Ensemble - Accuracy: {metrics['accuracy']:.4f}, "
               f"F1: {metrics['f1']:.4f}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train ensemble meta-learner')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--models-dir', type=str,
                        help='Directory containing trained base models')
    parser.add_argument('--data-dir', type=str,
                        help='Directory containing processed datasets')
    parser.add_argument('--save-dir', type=str,
                        help='Directory to save ensemble')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cuda, cpu, or auto)')
    
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
    if args.models_dir is None:
        args.models_dir = os.path.join(config['paths']['models_dir'], 'base_models')
    
    if args.data_dir is None:
        args.data_dir = config['paths']['data_dir']
    
    if args.save_dir is None:
        args.save_dir = os.path.join(config['paths']['models_dir'], 'ensemble')
    
    # Load trained base models
    models = load_trained_models(args.models_dir, config, device)
    
    # Setup data module
    data_module = DeepfakeDataModule(config, args.data_dir)
    data_module.setup('faceforensics')
    
    # Get hold-out data loader for meta-feature generation
    holdout_loader, _ = data_module.get_loader('holdout')
    
    # Generate meta-features
    meta_features, targets, individual_predictions = generate_meta_features(
        models, holdout_loader, device
    )
    
    # Evaluate base models
    base_model_results = evaluate_base_models(individual_predictions, targets)
    
    # Train meta-learner
    meta_learner = train_meta_learner(meta_features, targets, config)
    
    # Create ensemble
    ensemble = StackedEnsemble(models, meta_learner, device)
    
    # Evaluate ensemble on hold-out set
    ensemble_results = evaluate_ensemble(ensemble, holdout_loader, device)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Save ensemble
    ensemble.save_ensemble(args.save_dir)
    
    # Save training results
    results = {
        'base_models': base_model_results,
        'ensemble': ensemble_results,
        'meta_learner_cv': meta_learner.model.cv_results_ if hasattr(meta_learner.model, 'cv_results_') else None
    }
    
    results_path = os.path.join(args.save_dir, 'ensemble_results.yaml')
    with open(results_path, 'w') as f:
        yaml.dump(results, f, default_flow_style=False)
    
    # Save meta-features for analysis
    meta_features_path = os.path.join(args.save_dir, 'meta_features.npz')
    np.savez(
        meta_features_path,
        meta_features=meta_features,
        targets=targets,
        **individual_predictions
    )
    
    logger.info("Ensemble training completed!")
    logger.info(f"Results saved to {args.save_dir}")
    
    # Print summary
    print("\n" + "="*50)
    print("ENSEMBLE TRAINING SUMMARY")
    print("="*50)
    print("\nBase Model Performance:")
    for model_name, metrics in base_model_results.items():
        print(f"  {model_name}: Accuracy = {metrics['accuracy']:.4f}, F1 = {metrics['f1']:.4f}")
    
    print(f"\nEnsemble Performance:")
    print(f"  Accuracy = {ensemble_results['accuracy']:.4f}, F1 = {ensemble_results['f1']:.4f}")
    
    # Calculate improvement
    avg_base_acc = np.mean([metrics['accuracy'] for metrics in base_model_results.values()])
    improvement = ensemble_results['accuracy'] - avg_base_acc
    print(f"\nImprovement over average base model: {improvement:.4f}")


if __name__ == '__main__':
    main()
