"""
Model factory for creating and managing deepfake detection models.

This module provides utilities for creating, loading, and managing
the three base models (ViT, DeiT, Swin) used in the ensemble.
"""

import os
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple
import logging

from .base_models import ViTModel, DeiTModel, SwinModel, BaseDeepfakeModel
from .base_models import create_base_model, load_model_weights, save_model_weights

logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory class for creating and managing deepfake detection models."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Configuration dictionary containing model specifications
        """
        self.config = config
        self.models_config = config.get('models', {})
        self.base_models_config = self.models_config.get('base_models', {})
        
    def create_model(
        self,
        model_type: str,
        device: Optional[torch.device] = None,
        **kwargs
    ) -> BaseDeepfakeModel:
        """
        Create a single model.
        
        Args:
            model_type: Type of model ('vit', 'deit', 'swin')
            device: Device to place the model on
            **kwargs: Additional arguments to override config
            
        Returns:
            Created model instance
        """
        if model_type not in self.base_models_config:
            raise ValueError(f"Model type {model_type} not found in configuration")
        
        # Get model configuration
        model_config = self.base_models_config[model_type].copy()
        model_config.update(kwargs)  # Override with any provided kwargs
        
        # Create model
        model = create_base_model(model_type, model_config)
        
        # Move to device if specified
        if device is not None:
            model = model.to(device)
        
        logger.info(f"Created {model_type} model: {model.model_name}")
        return model
    
    def create_all_base_models(
        self,
        device: Optional[torch.device] = None
    ) -> Dict[str, BaseDeepfakeModel]:
        """
        Create all three base models.
        
        Args:
            device: Device to place models on
            
        Returns:
            Dictionary mapping model types to model instances
        """
        models = {}
        
        for model_type in ['vit', 'deit', 'swin']:
            if model_type in self.base_models_config:
                models[model_type] = self.create_model(model_type, device)
            else:
                logger.warning(f"Configuration for {model_type} not found, skipping")
        
        logger.info(f"Created {len(models)} base models")
        return models
    
    def load_pretrained_models(
        self,
        models_dir: str,
        device: Optional[torch.device] = None
    ) -> Dict[str, BaseDeepfakeModel]:
        """
        Load pretrained models from saved checkpoints.
        
        Args:
            models_dir: Directory containing model checkpoints
            device: Device to place models on
            
        Returns:
            Dictionary of loaded models
        """
        models = {}
        
        # Expected checkpoint files
        checkpoint_files = {
            'vit': 'vit.pth',
            'deit': 'deit.pth',
            'swin': 'swin.pth'
        }
        
        for model_type, filename in checkpoint_files.items():
            checkpoint_path = os.path.join(models_dir, filename)
            
            if os.path.exists(checkpoint_path):
                # Create model
                model = self.create_model(model_type, device)
                
                # Load weights
                model = load_model_weights(model, checkpoint_path)
                models[model_type] = model
                
                logger.info(f"Loaded {model_type} model from {checkpoint_path}")
            else:
                logger.warning(f"Checkpoint not found: {checkpoint_path}")
        
        return models
    
    def save_models(
        self,
        models: Dict[str, BaseDeepfakeModel],
        save_dir: str,
        epoch: Optional[int] = None,
        metrics: Optional[Dict[str, Dict]] = None
    ) -> None:
        """
        Save multiple models to disk.
        
        Args:
            models: Dictionary of models to save
            save_dir: Directory to save models
            epoch: Current epoch number
            metrics: Training metrics for each model
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Expected save files
        save_files = {
            'vit': 'vit.pth',
            'deit': 'deit.pth',
            'swin': 'swin.pth'
        }
        
        for model_type, model in models.items():
            if model_type in save_files:
                save_path = os.path.join(save_dir, save_files[model_type])
                model_metrics = metrics.get(model_type) if metrics else None
                
                save_model_weights(
                    model=model,
                    save_path=save_path,
                    epoch=epoch,
                    metrics=model_metrics
                )
    
    def get_model_info(
        self,
        models: Dict[str, BaseDeepfakeModel]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all models.
        
        Args:
            models: Dictionary of models
            
        Returns:
            Dictionary containing model information
        """
        info = {}
        
        for model_type, model in models.items():
            info[model_type] = {
                'model_name': model.model_name,
                'num_parameters': model.get_num_parameters(),
                'config': model.get_config()
            }
        
        return info


def create_deepfake_model(
    model_type: str,
    config: Dict[str, Any],
    device: Optional[torch.device] = None,
    checkpoint_path: Optional[str] = None
) -> BaseDeepfakeModel:
    """
    Convenience function to create a single deepfake detection model.
    
    Args:
        model_type: Type of model ('vit', 'deit', 'swin')
        config: Configuration dictionary
        device: Device to place model on
        checkpoint_path: Path to checkpoint file (optional)
        
    Returns:
        Created model instance
    """
    factory = ModelFactory(config)
    model = factory.create_model(model_type, device)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        model = load_model_weights(model, checkpoint_path)
    
    return model


def load_pretrained_weights(
    model: BaseDeepfakeModel,
    weights_path: str,
    strict: bool = True
) -> BaseDeepfakeModel:
    """
    Load pretrained weights into a model.
    
    Args:
        model: Model instance
        weights_path: Path to weights file
        strict: Whether to strictly enforce key matching
        
    Returns:
        Model with loaded weights
    """
    return load_model_weights(model, weights_path, strict)


def prepare_models_for_training(
    models: Dict[str, BaseDeepfakeModel],
    freeze_backbone: bool = False,
    learning_rates: Optional[Dict[str, float]] = None
) -> Dict[str, torch.optim.Optimizer]:
    """
    Prepare models for training by setting up optimizers.
    
    Args:
        models: Dictionary of models
        freeze_backbone: Whether to freeze backbone parameters
        learning_rates: Learning rates for each model
        
    Returns:
        Dictionary of optimizers
    """
    if learning_rates is None:
        learning_rates = {'vit': 1e-4, 'deit': 1e-4, 'swin': 1e-4}
    
    optimizers = {}
    
    for model_type, model in models.items():
        # Freeze backbone if requested
        if freeze_backbone:
            model.freeze_backbone(True)
        
        # Create optimizer
        lr = learning_rates.get(model_type, 1e-4)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=1e-5,
            betas=(0.9, 0.999)
        )
        
        optimizers[model_type] = optimizer
        logger.info(f"Created optimizer for {model_type} with lr={lr}")
    
    return optimizers


def get_models_summary(models: Dict[str, BaseDeepfakeModel]) -> Dict[str, Any]:
    """
    Get a comprehensive summary of all models.
    
    Args:
        models: Dictionary of models
        
    Returns:
        Summary dictionary
    """
    summary = {
        'total_models': len(models),
        'total_parameters': sum(model.get_num_parameters() for model in models.values()),
        'models': {}
    }
    
    for model_type, model in models.items():
        summary['models'][model_type] = {
            'model_name': model.model_name,
            'parameters': model.get_num_parameters(),
            'config': model.get_config()
        }
    
    return summary


def validate_model_compatibility(models: Dict[str, BaseDeepfakeModel]) -> bool:
    """
    Validate that all models are compatible for ensemble use.
    
    Args:
        models: Dictionary of models
        
    Returns:
        True if all models are compatible
    """
    if not models:
        logger.error("No models provided for validation")
        return False
    
    # Check that all models have the same number of classes
    num_classes_set = set(model.num_classes for model in models.values())
    if len(num_classes_set) > 1:
        logger.error(f"Models have different number of classes: {num_classes_set}")
        return False
    
    # Check that all expected model types are present
    expected_types = {'vit', 'deit', 'swin'}
    present_types = set(models.keys())
    
    if not expected_types.issubset(present_types):
        missing_types = expected_types - present_types
        logger.warning(f"Missing model types: {missing_types}")
    
    logger.info("Model compatibility validation passed")
    return True
