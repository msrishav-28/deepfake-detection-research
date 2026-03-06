"""
Training utilities for deepfake detection models.

This module contains helper functions for training, validation, checkpointing,
and other training-related utilities.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import time

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = 'INFO', log_file: Optional[str] = None) -> None:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level
        log_file: Optional log file path
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, Any],
    filepath: str,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        metrics: Training metrics
        filepath: Path to save checkpoint
        scheduler: Optional learning rate scheduler
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'timestamp': time.time()
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    torch.save(checkpoint, filepath)
    logger.info(f"Checkpoint saved to {filepath}")


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    filepath: str,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        model: Model to load weights into
        optimizer: Optimizer to load state into
        filepath: Path to checkpoint file
        scheduler: Optional learning rate scheduler
        device: Device to load checkpoint on
        
    Returns:
        Checkpoint metadata
    """
    if device is None:
        device = torch.device('cpu')
    
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    logger.info(f"Checkpoint loaded from {filepath}")
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'metrics': checkpoint.get('metrics', {}),
        'timestamp': checkpoint.get('timestamp', 0)
    }


def calculate_metrics(
    y_true: List[int],
    y_pred: List[int],
    average: str = 'binary'
) -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging strategy for metrics
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0)
    }
    
    return metrics


def print_classification_report(
    y_true: List[int],
    y_pred: List[int],
    target_names: Optional[List[str]] = None
) -> str:
    """
    Generate and print classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Names of target classes
        
    Returns:
        Classification report string
    """
    if target_names is None:
        target_names = ['Real', 'Fake']
    
    report = classification_report(y_true, y_pred, target_names=target_names)
    print(report)
    return report


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
        mode: str = 'min'
    ):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore best weights
            mode: 'min' for loss, 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.mode = mode
        
        self.best_value = None
        self.counter = 0
        self.best_weights = None
        
        if mode == 'min':
            self.is_better = lambda current, best: current < best - min_delta
            self.best_value = float('inf')
        else:
            self.is_better = lambda current, best: current > best + min_delta
            self.best_value = float('-inf')
    
    def __call__(self, current_value: float, model: Optional[nn.Module] = None) -> bool:
        """
        Check if training should stop.
        
        Args:
            current_value: Current metric value
            model: Model to save best weights from
            
        Returns:
            True if training should stop
        """
        if self.is_better(current_value, self.best_value):
            self.best_value = current_value
            self.counter = 0
            
            if self.restore_best_weights and model is not None:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights and model is not None and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
                logger.info("Restored best weights")
            return True
        
        return False


class AverageMeter:
    """Utility class to track running averages."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: torch.device,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    early_stopping: Optional[EarlyStopping] = None,
    save_dir: Optional[str] = None,
    model_name: str = 'model'
) -> Dict[str, List[float]]:
    """
    Train a model with validation.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Number of epochs
        device: Device to train on
        scheduler: Learning rate scheduler
        early_stopping: Early stopping callback
        save_dir: Directory to save checkpoints
        model_name: Name for saving checkpoints
        
    Returns:
        Training history dictionary
    """
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = AverageMeter()
        train_acc = AverageMeter()
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            acc = (predicted == targets).float().mean()
            
            train_loss.update(loss.item(), inputs.size(0))
            train_acc.update(acc.item(), inputs.size(0))
        
        # Validation phase
        val_metrics = validate_model(model, val_loader, criterion, device)
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
        
        # Record history
        history['train_loss'].append(train_loss.avg)
        history['train_acc'].append(train_acc.avg)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            if save_dir:
                checkpoint_path = os.path.join(save_dir, f'{model_name}_best.pth')
                save_checkpoint(model, optimizer, epoch, val_metrics, checkpoint_path, scheduler)
        
        # Early stopping
        if early_stopping and early_stopping(val_metrics['loss'], model):
            logger.info(f"Early stopping at epoch {epoch}")
            break
        
        # Log progress
        logger.info(
            f"Epoch {epoch}: Train Loss: {train_loss.avg:.4f}, "
            f"Train Acc: {train_acc.avg:.4f}, Val Loss: {val_metrics['loss']:.4f}, "
            f"Val Acc: {val_metrics['accuracy']:.4f}"
        )
    
    return history


def validate_model(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """
    Validate a model.
    
    Args:
        model: Model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        
    Returns:
        Validation metrics
    """
    model.eval()
    
    val_loss = AverageMeter()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            _, predicted = torch.max(outputs.data, 1)
            
            val_loss.update(loss.item(), inputs.size(0))
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate metrics
    metrics = calculate_metrics(all_targets, all_predictions)
    metrics['loss'] = val_loss.avg
    
    return metrics
