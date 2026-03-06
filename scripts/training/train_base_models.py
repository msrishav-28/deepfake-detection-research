#!/usr/bin/env python3
"""
Training script for base models (ViT, DeiT, Swin) in deepfake detection.

This script fine-tunes the three Vision Transformer models for binary
deepfake classification using the training set.

Usage:
    python scripts/training/train_base_models.py --config config.yaml
    python scripts/training/train_base_models.py --model vit --data-dir data/processed --epochs 50
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import logging
from tqdm import tqdm
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from deepfake_detection.models.model_factory import ModelFactory, create_deepfake_model
from deepfake_detection.data.timm_integration import DeepfakeDataModule
from deepfake_detection.utils.training_utils import (
    setup_logging, save_checkpoint, load_checkpoint, 
    calculate_metrics, EarlyStopping
)

logger = logging.getLogger(__name__)


def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    mixup_fn=None,
    epoch: int = 0
) -> dict:
    """Train model for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_samples = 0
    correct_predictions = 0
    
    progress_bar = tqdm(loader, desc=f'Training Epoch {epoch}')
    
    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        # Apply MixUp/CutMix if available
        if mixup_fn is not None:
            inputs, targets = mixup_fn(inputs, targets)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Calculate loss
        if mixup_fn is not None and isinstance(targets, tuple):
            # MixUp loss calculation
            targets_a, targets_b, lam = targets
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
        else:
            loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        total_samples += inputs.size(0)
        
        # Calculate accuracy (only for non-mixup batches)
        if not (mixup_fn is not None and isinstance(targets, tuple)):
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == targets).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Avg Loss': f'{total_loss / (batch_idx + 1):.4f}'
        })
    
    avg_loss = total_loss / len(loader)
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'samples': total_samples
    }


def validate_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int = 0
) -> dict:
    """Validate model for one epoch."""
    model.eval()
    
    total_loss = 0.0
    total_samples = 0
    correct_predictions = 0
    all_predictions = []
    all_targets = []
    
    progress_bar = tqdm(loader, desc=f'Validation Epoch {epoch}')
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Statistics
            total_loss += loss.item()
            total_samples += inputs.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == targets).sum().item()
            
            # Store predictions for detailed metrics
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss / (batch_idx + 1):.4f}'
            })
    
    avg_loss = total_loss / len(loader)
    accuracy = correct_predictions / total_samples
    
    # Calculate detailed metrics
    metrics = calculate_metrics(all_targets, all_predictions)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'samples': total_samples,
        **metrics
    }


def train_single_model(
    model_type: str,
    config: dict,
    data_module: DeepfakeDataModule,
    device: torch.device,
    save_dir: str
) -> dict:
    """Train a single model."""
    logger.info(f"Starting training for {model_type} model")
    
    # Create model
    factory = ModelFactory(config)
    model = factory.create_model(model_type, device)
    
    # Setup training components
    training_config = config['training']['base_models']
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config['learning_rate'],
        weight_decay=training_config['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=training_config['epochs']
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=10,
        min_delta=0.001,
        restore_best_weights=True
    )
    
    # Get data loaders
    train_loader, mixup_fn = data_module.get_loader('train')
    val_loader, _ = data_module.get_loader('holdout')  # Use holdout for validation
    
    # Setup logging
    log_dir = os.path.join(save_dir, 'logs', model_type)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    # Training loop
    best_val_acc = 0.0
    training_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rates': []
    }
    
    for epoch in range(training_config['epochs']):
        start_time = time.time()
        
        # Training
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, mixup_fn, epoch
        )
        
        # Validation
        val_metrics = validate_epoch(
            model, val_loader, criterion, device, epoch
        )
        
        # Learning rate scheduling
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log metrics
        writer.add_scalar('Loss/Train', train_metrics['loss'], epoch)
        writer.add_scalar('Loss/Validation', val_metrics['loss'], epoch)
        writer.add_scalar('Accuracy/Train', train_metrics['accuracy'], epoch)
        writer.add_scalar('Accuracy/Validation', val_metrics['accuracy'], epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # Store history
        training_history['train_loss'].append(train_metrics['loss'])
        training_history['train_acc'].append(train_metrics['accuracy'])
        training_history['val_loss'].append(val_metrics['loss'])
        training_history['val_acc'].append(val_metrics['accuracy'])
        training_history['learning_rates'].append(current_lr)
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            model_save_path = os.path.join(save_dir, f'{model_type}.pth')
            save_checkpoint(
                model, optimizer, epoch, val_metrics, model_save_path
            )
        
        # Early stopping check
        if early_stopping(val_metrics['loss']):
            logger.info(f"Early stopping triggered for {model_type} at epoch {epoch}")
            break
        
        # Log progress
        epoch_time = time.time() - start_time
        logger.info(
            f"{model_type} Epoch {epoch}: "
            f"Train Loss: {train_metrics['loss']:.4f}, "
            f"Train Acc: {train_metrics['accuracy']:.4f}, "
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"Val Acc: {val_metrics['accuracy']:.4f}, "
            f"Time: {epoch_time:.2f}s"
        )
    
    writer.close()
    
    # Final evaluation
    final_metrics = validate_epoch(model, val_loader, criterion, device)
    
    logger.info(f"Completed training for {model_type}")
    logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
    
    return {
        'best_val_acc': best_val_acc,
        'final_metrics': final_metrics,
        'training_history': training_history
    }


def main():
    parser = argparse.ArgumentParser(description='Train base models for deepfake detection')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--model', type=str, choices=['vit', 'deit', 'swin', 'all'],
                        default='all', help='Model to train')
    parser.add_argument('--data-dir', type=str,
                        help='Directory containing processed datasets')
    parser.add_argument('--save-dir', type=str,
                        help='Directory to save trained models')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cuda, cpu, or auto)')
    parser.add_argument('--epochs', type=int,
                        help='Number of training epochs (overrides config)')
    
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
    if args.data_dir is None:
        args.data_dir = config['paths']['data_dir']
    
    if args.save_dir is None:
        args.save_dir = os.path.join(config['paths']['models_dir'], 'base_models')
    
    # Override epochs if provided
    if args.epochs is not None:
        config['training']['base_models']['epochs'] = args.epochs
    
    # Setup data module
    data_module = DeepfakeDataModule(config, args.data_dir)
    data_module.setup('faceforensics')  # Use FaceForensics++ dataset
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Train models
    results = {}
    
    if args.model == 'all':
        models_to_train = ['vit', 'deit', 'swin']
    else:
        models_to_train = [args.model]
    
    for model_type in models_to_train:
        try:
            result = train_single_model(
                model_type, config, data_module, device, args.save_dir
            )
            results[model_type] = result
        except Exception as e:
            logger.error(f"Error training {model_type}: {e}")
            continue
    
    # Save training summary
    summary_path = os.path.join(args.save_dir, 'training_summary.yaml')
    with open(summary_path, 'w') as f:
        yaml.dump(results, f, default_flow_style=False)
    
    logger.info("Training completed!")
    logger.info(f"Results saved to {args.save_dir}")


if __name__ == '__main__':
    main()
