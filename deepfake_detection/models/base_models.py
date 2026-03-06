"""
Base model implementations for deepfake detection.

This module contains wrapper classes for ViT, DeiT, and Swin Transformer models
that are fine-tuned for binary deepfake classification.
"""

import torch
import torch.nn as nn
from timm import create_model
from timm.models import VisionTransformer
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class BaseDeepfakeModel(nn.Module):
    """Base class for deepfake detection models."""
    
    def __init__(
        self,
        model_name: str,
        num_classes: int = 2,
        pretrained: bool = True,
        drop_rate: float = 0.1,
        drop_path_rate: float = 0.1,
        **kwargs
    ):
        """
        Args:
            model_name: Name of the timm model
            num_classes: Number of output classes (2 for binary classification)
            pretrained: Whether to use pretrained weights
            drop_rate: Dropout rate
            drop_path_rate: Drop path rate for regularization
        """
        super().__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Create the base model using timm
        self.model = create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            **kwargs
        )
        
        # Store model configuration
        self.config = {
            'model_name': model_name,
            'num_classes': num_classes,
            'pretrained': pretrained,
            'drop_rate': drop_rate,
            'drop_path_rate': drop_path_rate,
            **kwargs
        }
        
        logger.info(f"Created {model_name} with {self.get_num_parameters()} parameters")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before the classification head."""
        if hasattr(self.model, 'forward_features'):
            return self.model.forward_features(x)
        else:
            # Fallback for models without forward_features
            return self.model.forward_head(x, pre_logits=True)
    
    def get_num_parameters(self) -> int:
        """Get the number of trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def freeze_backbone(self, freeze: bool = True):
        """Freeze or unfreeze the backbone (all layers except classifier)."""
        for name, param in self.model.named_parameters():
            if 'head' not in name and 'classifier' not in name:
                param.requires_grad = not freeze
        
        logger.info(f"{'Froze' if freeze else 'Unfroze'} backbone parameters")
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.config.copy()


class ViTModel(BaseDeepfakeModel):
    """Vision Transformer model for deepfake detection."""
    
    def __init__(
        self,
        model_name: str = 'vit_base_patch16_224',
        num_classes: int = 2,
        pretrained: bool = True,
        **kwargs
    ):
        """
        Args:
            model_name: ViT model variant
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
        """
        super().__init__(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            **kwargs
        )
        
        # ViT-specific configurations
        self.patch_size = getattr(self.model, 'patch_embed', {}).patch_size if hasattr(self.model, 'patch_embed') else (16, 16)
        self.embed_dim = getattr(self.model, 'embed_dim', 768)
        
    def get_attention_maps(self, x: torch.Tensor, layer_idx: int = -1) -> torch.Tensor:
        """
        Extract attention maps from a specific transformer layer.
        
        Args:
            x: Input tensor
            layer_idx: Layer index (-1 for last layer)
            
        Returns:
            Attention maps tensor
        """
        # This is a simplified implementation
        # In practice, you'd need to hook into the attention layers
        with torch.no_grad():
            _ = self.model(x)
            # Return dummy attention maps for now
            # Real implementation would require modifying the forward pass
            batch_size = x.size(0)
            num_heads = 12  # Typical for base models
            seq_len = (224 // 16) ** 2 + 1  # Patches + CLS token
            return torch.zeros(batch_size, num_heads, seq_len, seq_len)


class DeiTModel(BaseDeepfakeModel):
    """Data-efficient Image Transformer model for deepfake detection."""
    
    def __init__(
        self,
        model_name: str = 'deit_base_distilled_patch16_224',
        num_classes: int = 2,
        pretrained: bool = True,
        **kwargs
    ):
        """
        Args:
            model_name: DeiT model variant
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
        """
        super().__init__(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            **kwargs
        )
        
        # Check if this is a distilled model
        self.is_distilled = 'distilled' in model_name
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass, handling distilled models appropriately."""
        output = self.model(x)
        
        # For distilled models during inference, average the two heads
        if self.is_distilled and isinstance(output, tuple):
            return (output[0] + output[1]) / 2
        
        return output


class SwinModel(BaseDeepfakeModel):
    """Swin Transformer model for deepfake detection."""
    
    def __init__(
        self,
        model_name: str = 'swin_base_patch4_window7_224',
        num_classes: int = 2,
        pretrained: bool = True,
        **kwargs
    ):
        """
        Args:
            model_name: Swin model variant
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
        """
        super().__init__(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            **kwargs
        )
        
        # Swin-specific configurations
        self.window_size = getattr(self.model, 'window_size', 7)
        self.patch_size = getattr(self.model, 'patch_size', 4)
    
    def get_feature_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract feature maps from different stages.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary of feature maps from different stages
        """
        features = {}
        
        # This would require modifying the forward pass to capture intermediate features
        # For now, return the final features
        with torch.no_grad():
            final_features = self.get_features(x)
            features['final'] = final_features
        
        return features


def create_base_model(
    model_type: str,
    model_config: Dict[str, Any]
) -> BaseDeepfakeModel:
    """
    Factory function to create base models.
    
    Args:
        model_type: Type of model ('vit', 'deit', 'swin')
        model_config: Model configuration dictionary
        
    Returns:
        Instantiated model
    """
    model_classes = {
        'vit': ViTModel,
        'deit': DeiTModel,
        'swin': SwinModel
    }
    
    if model_type not in model_classes:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(model_classes.keys())}")
    
    model_class = model_classes[model_type]
    return model_class(**model_config)


def load_model_weights(
    model: BaseDeepfakeModel,
    checkpoint_path: str,
    strict: bool = True
) -> BaseDeepfakeModel:
    """
    Load model weights from checkpoint.
    
    Args:
        model: Model instance
        checkpoint_path: Path to checkpoint file
        strict: Whether to strictly enforce key matching
        
    Returns:
        Model with loaded weights
    """
    logger.info(f"Loading weights from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Remove 'model.' prefix if present
    if any(key.startswith('model.') for key in state_dict.keys()):
        state_dict = {key.replace('model.', ''): value for key, value in state_dict.items()}
    
    # Load weights
    missing_keys, unexpected_keys = model.model.load_state_dict(state_dict, strict=strict)
    
    if missing_keys:
        logger.warning(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        logger.warning(f"Unexpected keys: {unexpected_keys}")
    
    logger.info("Successfully loaded model weights")
    return model


def save_model_weights(
    model: BaseDeepfakeModel,
    save_path: str,
    epoch: Optional[int] = None,
    optimizer_state: Optional[Dict] = None,
    metrics: Optional[Dict] = None
) -> None:
    """
    Save model weights and training state.
    
    Args:
        model: Model instance
        save_path: Path to save checkpoint
        epoch: Current epoch number
        optimizer_state: Optimizer state dict
        metrics: Training metrics
    """
    checkpoint = {
        'model': model.model.state_dict(),
        'config': model.get_config(),
        'model_type': model.__class__.__name__,
    }
    
    if epoch is not None:
        checkpoint['epoch'] = epoch
    
    if optimizer_state is not None:
        checkpoint['optimizer'] = optimizer_state
    
    if metrics is not None:
        checkpoint['metrics'] = metrics
    
    torch.save(checkpoint, save_path)
    logger.info(f"Saved model checkpoint to {save_path}")


def get_model_summary(model: BaseDeepfakeModel) -> Dict[str, Any]:
    """
    Get a summary of the model.
    
    Args:
        model: Model instance
        
    Returns:
        Model summary dictionary
    """
    return {
        'model_name': model.model_name,
        'model_type': model.__class__.__name__,
        'num_classes': model.num_classes,
        'num_parameters': model.get_num_parameters(),
        'config': model.get_config()
    }
