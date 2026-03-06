"""
Data augmentation utilities for deepfake detection.

This module provides advanced augmentation techniques including MixUp and CutMix
implementations compatible with the timm framework.
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from torchvision import transforms
from typing import Tuple, Optional, Dict, Any
import random


class MixUpAugmentation:
    """MixUp augmentation implementation."""
    
    def __init__(self, alpha: float = 0.2, prob: float = 0.5, seed: int = None):
        """
        Args:
            alpha: Beta distribution parameter for mixing coefficient
            prob: Probability of applying MixUp
            seed: Optional seed for reproducible augmentation
        """
        self.alpha = alpha
        self.prob = prob
        self.rng = np.random.default_rng(seed)
    
    def __call__(self, batch: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply MixUp augmentation to a batch.
        
        Args:
            batch: Input batch of shape (N, C, H, W)
            targets: Target labels of shape (N,)
            
        Returns:
            Tuple of (mixed_batch, mixed_targets)
        """
        if self.rng.random() > self.prob:
            return batch, targets
        
        batch_size = batch.size(0)
        
        # Sample mixing coefficient
        if self.alpha > 0:
            lam = self.rng.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        # Ensure dominant sample retains majority weight (Zhang et al., 2018)
        lam = max(lam, 1 - lam)
        
        # Create random permutation
        index = torch.randperm(batch_size).to(batch.device)
        
        # Mix inputs
        mixed_batch = lam * batch + (1 - lam) * batch[index, :]
        
        # Mix targets (for soft labels)
        targets_a, targets_b = targets, targets[index]
        mixed_targets = (targets_a, targets_b, lam)
        
        return mixed_batch, mixed_targets


class CutMixAugmentation:
    """CutMix augmentation implementation."""
    
    def __init__(self, alpha: float = 1.0, prob: float = 0.5, seed: int = None):
        """
        Args:
            alpha: Beta distribution parameter for mixing coefficient
            prob: Probability of applying CutMix
            seed: Optional seed for reproducible augmentation
        """
        self.alpha = alpha
        self.prob = prob
        self.rng = np.random.default_rng(seed)
    
    def __call__(self, batch: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply CutMix augmentation to a batch.
        
        Args:
            batch: Input batch of shape (N, C, H, W)
            targets: Target labels of shape (N,)
            
        Returns:
            Tuple of (mixed_batch, mixed_targets)
        """
        if self.rng.random() > self.prob:
            return batch, targets
        
        batch_size = batch.size(0)
        
        # Sample mixing coefficient
        if self.alpha > 0:
            lam = self.rng.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        # Create random permutation
        index = torch.randperm(batch_size).to(batch.device)
        
        # Generate random bounding box
        W, H = batch.size(3), batch.size(2)
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        # Uniform sampling
        cx = self.rng.integers(W)
        cy = self.rng.integers(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # Apply CutMix
        mixed_batch = batch.clone()
        mixed_batch[:, :, bby1:bby2, bbx1:bbx2] = batch[index, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        # Mix targets
        targets_a, targets_b = targets, targets[index]
        mixed_targets = (targets_a, targets_b, lam)
        
        return mixed_batch, mixed_targets


def get_augmentation_transforms(
    split: str = 'train',
    image_size: int = 224,
    augmentation_config: Optional[Dict[str, Any]] = None
) -> transforms.Compose:
    """
    Get augmentation transforms for different dataset splits.
    
    Args:
        split: Dataset split ('train', 'holdout', 'test')
        image_size: Target image size
        augmentation_config: Configuration for augmentation parameters
        
    Returns:
        Composed transforms
    """
    if augmentation_config is None:
        augmentation_config = {
            'horizontal_flip_prob': 0.5,
            'rotation_degrees': 10,
            'color_jitter': {
                'brightness': 0.2,
                'contrast': 0.2,
                'saturation': 0.2,
                'hue': 0.1
            },
            'gaussian_blur_prob': 0.1,
            'gaussian_blur_sigma': (0.1, 2.0)
        }
    
    # Base transforms for all splits
    base_transforms = [
        transforms.Resize((image_size, image_size)),
    ]
    
    # Training-specific augmentations
    if split == 'train':
        train_transforms = [
            transforms.RandomHorizontalFlip(p=augmentation_config['horizontal_flip_prob']),
            transforms.RandomRotation(degrees=augmentation_config['rotation_degrees']),
            transforms.ColorJitter(
                brightness=augmentation_config['color_jitter']['brightness'],
                contrast=augmentation_config['color_jitter']['contrast'],
                saturation=augmentation_config['color_jitter']['saturation'],
                hue=augmentation_config['color_jitter']['hue']
            ),
            transforms.RandomApply([
                transforms.GaussianBlur(
                    kernel_size=3,
                    sigma=augmentation_config['gaussian_blur_sigma']
                )
            ], p=augmentation_config['gaussian_blur_prob']),
        ]
        base_transforms.extend(train_transforms)
    
    # Final transforms for all splits
    final_transforms = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet means
            std=[0.229, 0.224, 0.225]    # ImageNet stds
        )
    ]
    base_transforms.extend(final_transforms)
    
    return transforms.Compose(base_transforms)


def get_timm_mixup_transforms(
    mixup_alpha: float = 0.2,
    cutmix_alpha: float = 1.0,
    mixup_prob: float = 0.5,
    switch_prob: float = 0.5
) -> Dict[str, Any]:
    """
    Get MixUp/CutMix configuration compatible with timm.
    
    Args:
        mixup_alpha: MixUp alpha parameter
        cutmix_alpha: CutMix alpha parameter
        mixup_prob: Probability of applying MixUp/CutMix
        switch_prob: Probability of switching between MixUp and CutMix
        
    Returns:
        Configuration dictionary for timm Mixup
    """
    return {
        'mixup_alpha': mixup_alpha,
        'cutmix_alpha': cutmix_alpha,
        'cutmix_minmax': None,
        'prob': mixup_prob,
        'switch_prob': switch_prob,
        'mode': 'batch',
        'label_smoothing': 0.1,
        'num_classes': 2
    }


class FaceSpecificAugmentation:
    """Face-specific augmentation techniques for deepfake detection."""
    
    def __init__(self, prob: float = 0.3):
        """
        Args:
            prob: Probability of applying face-specific augmentations
        """
        self.prob = prob
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply face-specific augmentations.
        
        Args:
            image: Input image tensor of shape (C, H, W) in [0, 1] range
            
        Returns:
            Augmented image tensor
        """
        # Safety assertion: must be applied before normalization (Bug #14)
        assert image.max() <= 1.0 + 1e-5 and image.min() >= -1e-5, \
            "FaceSpecificAugmentation must be applied before normalization. " \
            f"Got range [{image.min():.3f}, {image.max():.3f}]"
        
        if random.random() > self.prob:
            return image
        
        # Apply random face-specific augmentations
        augmentations = [
            self._add_compression_artifacts,
            self._add_gaussian_noise,
            self._simulate_lighting_changes,
        ]
        
        # Randomly select and apply augmentation
        aug_func = random.choice(augmentations)
        return aug_func(image)
    
    def _add_compression_artifacts(self, image: torch.Tensor) -> torch.Tensor:
        """
        Simulate JPEG compression via actual DCT-quantization round-trip.
        Quality factor drawn from U[20, 75] to span low-to-medium compression.
        """
        # Convert [0,1] float tensor -> uint8 HWC numpy
        img_np = (image.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        quality = random.randint(20, 75)
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        _, buf = cv2.imencode('.jpg', img_bgr, encode_params)
        decoded_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        decoded_rgb = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)
        
        # Back to [0,1] float CHW tensor
        return torch.from_numpy(decoded_rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
    
    def _add_gaussian_noise(self, image: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to simulate camera sensor noise."""
        noise_std = random.uniform(0.01, 0.05)
        noise = torch.randn_like(image) * noise_std
        return torch.clamp(image + noise, 0, 1)
    
    def _simulate_lighting_changes(self, image: torch.Tensor) -> torch.Tensor:
        """Simulate lighting variations."""
        # Random brightness adjustment
        brightness_factor = random.uniform(0.8, 1.2)
        return torch.clamp(image * brightness_factor, 0, 1)


def create_augmentation_pipeline(
    split: str,
    config: Dict[str, Any]
) -> transforms.Compose:
    """
    Create a complete augmentation pipeline.
    
    Args:
        split: Dataset split
        config: Augmentation configuration
        
    Returns:
        Complete augmentation pipeline
    """
    # Get base transforms
    base_transforms = get_augmentation_transforms(
        split=split,
        image_size=config.get('image_size', 224),
        augmentation_config=config.get('augmentation', {})
    )
    
    # Add face-specific augmentations for training
    if split == 'train' and config.get('use_face_augmentation', False):
        face_aug = FaceSpecificAugmentation(prob=config.get('face_aug_prob', 0.3))
        
        # Insert face augmentation before normalization
        transform_list = base_transforms.transforms
        # Find the ToTensor transform
        tensor_idx = next(i for i, t in enumerate(transform_list) if isinstance(t, transforms.ToTensor))
        
        # Insert face augmentation after ToTensor but before Normalize
        transform_list.insert(tensor_idx + 1, face_aug)
        
        return transforms.Compose(transform_list)
    
    return base_transforms
