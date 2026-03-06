"""
Integration with timm's data loading and augmentation framework.

This module provides seamless integration between our custom datasets
and timm's existing data loading, augmentation, and training infrastructure.
"""

import os
import torch
from torch.utils.data import DataLoader
from timm.data import create_loader, Mixup, FastCollateMixup
from timm.data.transforms_factory import create_transform
from typing import Optional, Dict, Any, Tuple
import logging

from .datasets import FaceForensicsDataset, CelebDFDataset
from .augmentations import get_timm_mixup_transforms

logger = logging.getLogger(__name__)


def create_deepfake_dataset(
    dataset_type: str,
    data_dir: str,
    split: str = 'train',
    config: Optional[Dict[str, Any]] = None
) -> torch.utils.data.Dataset:
    """
    Create a deepfake dataset compatible with timm.
    
    Args:
        dataset_type: Type of dataset ('faceforensics' or 'celebdf')
        data_dir: Root directory containing the dataset
        split: Dataset split ('train', 'holdout', 'test')
        config: Configuration dictionary
        
    Returns:
        Dataset instance
    """
    if config is None:
        config = {}
    
    # Get image size from config
    image_size = config.get('image_size', 224)
    
    # Create transforms using timm's factory
    transform = create_transform(
        input_size=(3, image_size, image_size),
        is_training=(split == 'train'),
        use_prefetcher=False,
        no_aug=False,
        scale=(0.08, 1.0),
        ratio=(3./4., 4./3.),
        hflip=0.5,
        vflip=0.0,
        color_jitter=0.4,
        auto_augment='rand-m9-mstd0.5-inc1',
        interpolation='bicubic',
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        re_prob=0.25,
        re_mode='pixel',
        re_count=1,
        crop_pct=0.875,
    )
    
    # Create dataset based on type
    if dataset_type == 'faceforensics':
        dataset = FaceForensicsDataset(
            data_dir=data_dir,
            split=split,
            transform=transform,
            image_size=image_size,
            categories=config.get('categories', None),
            videos_per_category=config.get('videos_per_category', 100)
        )
    elif dataset_type == 'celebdf':
        dataset = CelebDFDataset(
            data_dir=data_dir,
            split=split,
            transform=transform,
            image_size=image_size
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    logger.info(f"Created {dataset_type} dataset for {split} split with {len(dataset)} samples")
    return dataset


def create_deepfake_loader(
    dataset: torch.utils.data.Dataset,
    batch_size: int = 32,
    is_training: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    use_mixup: bool = True,
    mixup_config: Optional[Dict[str, Any]] = None
) -> Tuple[DataLoader, Optional[Mixup]]:
    """
    Create a data loader with optional MixUp/CutMix augmentation.
    
    Args:
        dataset: Dataset instance
        batch_size: Batch size
        is_training: Whether this is for training
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        use_mixup: Whether to use MixUp/CutMix
        mixup_config: MixUp configuration
        
    Returns:
        Tuple of (DataLoader, Mixup function or None)
    """
    # Default MixUp configuration
    if mixup_config is None:
        mixup_config = get_timm_mixup_transforms()
    
    # Create MixUp function if requested and training
    mixup_fn = None
    collate_fn = None
    
    if use_mixup and is_training:
        mixup_fn = Mixup(**mixup_config)
        # Use FastCollateMixup for better performance
        collate_fn = FastCollateMixup(**mixup_config)
    
    # Create data loader using timm's create_loader
    loader = create_loader(
        dataset,
        input_size=(3, 224, 224),  # Will be overridden by dataset transforms
        batch_size=batch_size,
        is_training=is_training,
        use_prefetcher=True,
        no_aug=False,
        re_prob=0.25,
        re_mode='pixel',
        re_count=1,
        re_split=False,
        scale=(0.08, 1.0),
        ratio=(3./4., 4./3.),
        hflip=0.5,
        vflip=0.0,
        color_jitter=0.4,
        auto_augment='rand-m9-mstd0.5-inc1',
        num_aug_repeats=0,
        num_aug_splits=0,
        interpolation='bicubic',
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        num_workers=num_workers,
        distributed=False,
        crop_pct=0.875,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        use_multi_epochs_loader=False,
        persistent_workers=True if num_workers > 0 else False
    )
    
    logger.info(f"Created data loader with batch_size={batch_size}, mixup={use_mixup}")
    return loader, mixup_fn


def create_deepfake_loaders(
    config: Dict[str, Any],
    data_dir: str,
    dataset_type: str = 'faceforensics'
) -> Dict[str, Tuple[DataLoader, Optional[Mixup]]]:
    """
    Create data loaders for all splits (train, holdout, test).
    
    Args:
        config: Configuration dictionary
        data_dir: Root directory containing datasets
        dataset_type: Type of dataset
        
    Returns:
        Dictionary mapping split names to (loader, mixup_fn) tuples
    """
    loaders = {}
    
    # Training configuration
    training_config = config.get('training', {})
    batch_size = training_config.get('batch_size', 32)
    num_workers = config.get('hardware', {}).get('num_workers', 4)
    pin_memory = config.get('hardware', {}).get('pin_memory', True)
    
    # MixUp configuration
    use_mixup = training_config.get('augmentation', {}).get('mixup_prob', 0) > 0
    mixup_config = None
    if use_mixup:
        mixup_config = get_timm_mixup_transforms(
            mixup_alpha=training_config.get('augmentation', {}).get('mixup_alpha', 0.2),
            cutmix_alpha=training_config.get('augmentation', {}).get('cutmix_alpha', 1.0),
            mixup_prob=training_config.get('augmentation', {}).get('mixup_prob', 0.5)
        )
    
    # Create loaders for each split
    for split in ['train', 'holdout', 'test']:
        # Create dataset
        dataset = create_deepfake_dataset(
            dataset_type=dataset_type,
            data_dir=os.path.join(data_dir, dataset_type),
            split=split,
            config=config.get('data', {})
        )
        
        # Skip if dataset is empty
        if len(dataset) == 0:
            logger.warning(f"Empty dataset for {split} split, skipping")
            continue
        
        # Create loader
        is_training = (split == 'train')
        loader, mixup_fn = create_deepfake_loader(
            dataset=dataset,
            batch_size=batch_size,
            is_training=is_training,
            num_workers=num_workers,
            pin_memory=pin_memory,
            use_mixup=use_mixup and is_training,
            mixup_config=mixup_config
        )
        
        loaders[split] = (loader, mixup_fn)
        logger.info(f"Created {split} loader with {len(dataset)} samples")
    
    return loaders


def get_data_config_for_timm(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert our configuration to timm-compatible data configuration.
    
    Args:
        config: Our configuration dictionary
        
    Returns:
        timm-compatible data configuration
    """
    data_config = config.get('data', {})
    models_config = config.get('models', {})
    
    # Get image size from model configuration
    image_size = 224  # Default
    for model_name, model_config in models_config.get('base_models', {}).items():
        if 'input_size' in model_config:
            image_size = model_config['input_size']
            break
    
    return {
        'input_size': (3, image_size, image_size),
        'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406),
        'std': (0.229, 0.224, 0.225),
        'crop_pct': 0.875,
    }


class DeepfakeDataModule:
    """Data module for managing deepfake detection datasets and loaders."""
    
    def __init__(self, config: Dict[str, Any], data_dir: str):
        """
        Args:
            config: Configuration dictionary
            data_dir: Root directory containing datasets
        """
        self.config = config
        self.data_dir = data_dir
        self.loaders = {}
        self.datasets = {}
    
    def setup(self, dataset_type: str = 'faceforensics'):
        """Setup datasets and loaders."""
        logger.info(f"Setting up data module for {dataset_type}")
        
        # Create loaders
        self.loaders = create_deepfake_loaders(
            config=self.config,
            data_dir=self.data_dir,
            dataset_type=dataset_type
        )
        
        # Store datasets for easy access
        for split, (loader, _) in self.loaders.items():
            self.datasets[split] = loader.dataset
    
    def get_loader(self, split: str) -> Tuple[DataLoader, Optional[Mixup]]:
        """Get loader for a specific split."""
        if split not in self.loaders:
            raise ValueError(f"Split {split} not available. Available splits: {list(self.loaders.keys())}")
        return self.loaders[split]
    
    def get_dataset(self, split: str) -> torch.utils.data.Dataset:
        """Get dataset for a specific split."""
        if split not in self.datasets:
            raise ValueError(f"Split {split} not available. Available splits: {list(self.datasets.keys())}")
        return self.datasets[split]
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get timm-compatible data configuration."""
        return get_data_config_for_timm(self.config)
