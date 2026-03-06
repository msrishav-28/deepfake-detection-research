"""
Ensemble models for deepfake detection.

This module implements the stacked ensemble approach that combines
predictions from multiple base models using meta-learning.
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from typing import Dict, List, Tuple, Optional, Any, Union
import logging

from .base_models import BaseDeepfakeModel

logger = logging.getLogger(__name__)


class MetaLearner:
    """Meta-learner for combining base model predictions."""
    
    def __init__(
        self,
        model_type: str = 'logistic_regression',
        **kwargs
    ):
        """
        Args:
            model_type: Type of meta-learner ('logistic_regression', 'random_forest')
            **kwargs: Additional arguments for the meta-learner
        """
        self.model_type = model_type
        self.model = None
        self.is_fitted = False
        
        # Create meta-learner model
        if model_type == 'logistic_regression':
            self.model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                **kwargs
            )
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(
                random_state=42,
                n_estimators=100,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown meta-learner type: {model_type}")
        
        logger.info(f"Created {model_type} meta-learner")
    
    def fit(
        self,
        meta_features: np.ndarray,
        targets: np.ndarray,
        cv_folds: int = 5
    ) -> Dict[str, float]:
        """
        Train the meta-learner.
        
        Args:
            meta_features: Features from base models (N, num_models * num_classes)
            targets: Target labels (N,)
            cv_folds: Number of cross-validation folds
            
        Returns:
            Cross-validation scores
        """
        logger.info(f"Training meta-learner with {len(meta_features)} samples")
        
        # Perform cross-validation
        cv_scores = cross_val_score(
            self.model, meta_features, targets, cv=cv_folds, scoring='accuracy'
        )
        
        # Fit on full dataset
        self.model.fit(meta_features, targets)
        self.is_fitted = True
        
        cv_results = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores.tolist()
        }
        
        logger.info(f"Meta-learner CV accuracy: {cv_results['cv_mean']:.4f} ± {cv_results['cv_std']:.4f}")
        return cv_results
    
    def predict(self, meta_features: np.ndarray) -> np.ndarray:
        """
        Make predictions using the meta-learner.
        
        Args:
            meta_features: Features from base models
            
        Returns:
            Predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Meta-learner must be fitted before making predictions")
        
        return self.model.predict(meta_features)
    
    def predict_proba(self, meta_features: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities from the meta-learner.
        
        Args:
            meta_features: Features from base models
            
        Returns:
            Prediction probabilities
        """
        if not self.is_fitted:
            raise ValueError("Meta-learner must be fitted before making predictions")
        
        return self.model.predict_proba(meta_features)
    
    def save(self, filepath: str) -> None:
        """Save the meta-learner to disk."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted meta-learner")
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'model_type': self.model_type,
                'is_fitted': self.is_fitted
            }, f)
        
        logger.info(f"Meta-learner saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load the meta-learner from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.model_type = data['model_type']
        self.is_fitted = data['is_fitted']
        
        logger.info(f"Meta-learner loaded from {filepath}")


class StackedEnsemble(nn.Module):
    """Stacked ensemble combining multiple base models with a meta-learner."""
    
    def __init__(
        self,
        base_models: Dict[str, BaseDeepfakeModel],
        meta_learner: Optional[MetaLearner] = None,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            base_models: Dictionary of base models
            meta_learner: Trained meta-learner (optional)
            device: Device to run models on
        """
        super().__init__()
        
        self.base_models = nn.ModuleDict(base_models)
        self.meta_learner = meta_learner
        self.device = device or torch.device('cpu')
        self.model_names = list(base_models.keys())
        
        # Move base models to device
        for model in self.base_models.values():
            model.to(self.device)
        
        logger.info(f"Created stacked ensemble with {len(base_models)} base models")
    
    def extract_meta_features(
        self,
        inputs: torch.Tensor,
        return_individual_predictions: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, np.ndarray]]]:
        """
        Extract meta-features from base models.
        
        Args:
            inputs: Input tensor
            return_individual_predictions: Whether to return individual predictions
            
        Returns:
            Meta-features array or tuple of (meta_features, individual_predictions)
        """
        self.eval()
        individual_predictions = {}
        meta_features_list = []
        
        with torch.no_grad():
            for model_name, model in self.base_models.items():
                # Get predictions from base model
                outputs = model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                
                # Store individual predictions
                individual_predictions[model_name] = probabilities.cpu().numpy()
                
                # Add to meta-features
                meta_features_list.append(probabilities.cpu().numpy())
        
        # Concatenate all meta-features
        meta_features = np.concatenate(meta_features_list, axis=1)
        
        if return_individual_predictions:
            return meta_features, individual_predictions
        else:
            return meta_features
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ensemble.
        
        Args:
            inputs: Input tensor
            
        Returns:
            Ensemble predictions
        """
        if self.meta_learner is None or not self.meta_learner.is_fitted:
            # If no meta-learner, return average of base model predictions
            return self._average_predictions(inputs)
        
        # Extract meta-features
        meta_features = self.extract_meta_features(inputs)
        
        # Get meta-learner predictions
        ensemble_probs = self.meta_learner.predict_proba(meta_features)
        
        # Convert back to torch tensor
        return torch.tensor(ensemble_probs, dtype=torch.float32, device=self.device)
    
    def _average_predictions(self, inputs: torch.Tensor) -> torch.Tensor:
        """Average predictions from all base models."""
        predictions = []
        
        with torch.no_grad():
            for model in self.base_models.values():
                outputs = model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                predictions.append(probabilities)
        
        # Average all predictions
        return torch.stack(predictions).mean(dim=0)
    
    def predict(self, inputs: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with the ensemble.
        
        Args:
            inputs: Input tensor
            
        Returns:
            Tuple of (predicted_labels, prediction_probabilities)
        """
        self.eval()
        
        with torch.no_grad():
            probabilities = self.forward(inputs)
            predicted_labels = torch.argmax(probabilities, dim=1)
        
        return predicted_labels.cpu().numpy(), probabilities.cpu().numpy()
    
    def get_model_contributions(
        self,
        inputs: torch.Tensor
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze individual model contributions to ensemble predictions.
        
        Args:
            inputs: Input tensor
            
        Returns:
            Dictionary with model contributions
        """
        meta_features, individual_preds = self.extract_meta_features(
            inputs, return_individual_predictions=True
        )
        
        # Get ensemble predictions
        if self.meta_learner and self.meta_learner.is_fitted:
            ensemble_probs = self.meta_learner.predict_proba(meta_features)
        else:
            ensemble_probs = np.mean([preds for preds in individual_preds.values()], axis=0)
        
        contributions = {}
        
        for i, model_name in enumerate(self.model_names):
            model_probs = individual_preds[model_name]
            
            # Calculate agreement with ensemble
            agreement = np.mean(np.abs(model_probs - ensemble_probs), axis=1)
            
            contributions[model_name] = {
                'mean_agreement': float(np.mean(agreement)),
                'confidence': float(np.mean(np.max(model_probs, axis=1))),
                'predictions': model_probs.tolist()
            }
        
        return contributions
    
    def save_ensemble(self, save_dir: str) -> None:
        """
        Save the complete ensemble.
        
        Args:
            save_dir: Directory to save ensemble components
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Save base models
        for model_name, model in self.base_models.items():
            model_path = os.path.join(save_dir, f'{model_name}.pth')
            torch.save(model.state_dict(), model_path)
        
        # Save meta-learner
        if self.meta_learner:
            meta_learner_path = os.path.join(save_dir, 'meta_learner.pkl')
            self.meta_learner.save(meta_learner_path)
        
        # Save ensemble configuration
        config_path = os.path.join(save_dir, 'ensemble_config.pkl')
        with open(config_path, 'wb') as f:
            pickle.dump({
                'model_names': self.model_names,
                'device': str(self.device)
            }, f)
        
        logger.info(f"Ensemble saved to {save_dir}")
    
    def load_ensemble(self, save_dir: str) -> None:
        """
        Load the complete ensemble.
        
        Args:
            save_dir: Directory containing ensemble components
        """
        import os
        
        # Load ensemble configuration
        config_path = os.path.join(save_dir, 'ensemble_config.pkl')
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
        
        # Load base models
        for model_name in config['model_names']:
            model_path = os.path.join(save_dir, f'{model_name}.pth')
            if os.path.exists(model_path):
                self.base_models[model_name].load_state_dict(torch.load(model_path))
        
        # Load meta-learner
        meta_learner_path = os.path.join(save_dir, 'meta_learner.pkl')
        if os.path.exists(meta_learner_path):
            if self.meta_learner is None:
                self.meta_learner = MetaLearner()
            self.meta_learner.load(meta_learner_path)
        
        logger.info(f"Ensemble loaded from {save_dir}")


def create_stacked_ensemble(
    base_models: Dict[str, BaseDeepfakeModel],
    meta_learner_type: str = 'logistic_regression',
    device: Optional[torch.device] = None
) -> StackedEnsemble:
    """
    Create a stacked ensemble.
    
    Args:
        base_models: Dictionary of base models
        meta_learner_type: Type of meta-learner
        device: Device to run on
        
    Returns:
        Stacked ensemble instance
    """
    meta_learner = MetaLearner(model_type=meta_learner_type)
    ensemble = StackedEnsemble(base_models, meta_learner, device)
    
    return ensemble
