"""
Explainability tools for deepfake detection models.

This module provides Grad-CAM visualizations and other explainability
techniques to understand model decision-making processes.
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from typing import List, Dict, Optional, Tuple, Any, Union
import logging

from ..models.base_models import BaseDeepfakeModel

logger = logging.getLogger(__name__)


class GradCAMVisualizer:
    """Grad-CAM visualization for Vision Transformer models."""
    
    def __init__(
        self,
        model: BaseDeepfakeModel,
        target_layers: Optional[List[str]] = None,
        use_cuda: bool = True
    ):
        """
        Args:
            model: The model to visualize
            target_layers: List of target layer names for Grad-CAM
            use_cuda: Whether to use CUDA if available
        """
        self.model = model
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        # Get target layers
        if target_layers is None:
            target_layers = self._get_default_target_layers()
        
        self.target_layers = self._get_target_layer_objects(target_layers)
        
        # Initialize Grad-CAM
        self.grad_cam = GradCAM(
            model=self.model.model,
            target_layers=self.target_layers,
            use_cuda=use_cuda
        )
        
        logger.info(f"GradCAM initialized for {model.model_name}")
    
    def _get_default_target_layers(self) -> List[str]:
        """Get default target layers based on model type."""
        model_name = self.model.model_name.lower()
        
        if 'vit' in model_name or 'deit' in model_name:
            # For ViT/DeiT models, use the last transformer block
            return ['blocks.11.norm1']  # Typical for base models
        elif 'swin' in model_name:
            # For Swin models, use the last stage
            return ['layers.3.blocks.1.norm1']
        else:
            # Fallback - try to find the last normalization layer
            return ['norm']
    
    def _get_target_layer_objects(self, layer_names: List[str]) -> List[nn.Module]:
        """Convert layer names to actual layer objects."""
        target_layers = []
        
        for layer_name in layer_names:
            try:
                # Navigate through the model hierarchy
                layer = self.model.model
                for attr in layer_name.split('.'):
                    if attr.isdigit():
                        layer = layer[int(attr)]
                    else:
                        layer = getattr(layer, attr)
                target_layers.append(layer)
                logger.info(f"Found target layer: {layer_name}")
            except (AttributeError, IndexError) as e:
                logger.warning(f"Could not find layer {layer_name}: {e}")
        
        if not target_layers:
            # Fallback to the last layer before classification head
            try:
                if hasattr(self.model.model, 'norm'):
                    target_layers = [self.model.model.norm]
                elif hasattr(self.model.model, 'head'):
                    # Get the layer before the head
                    target_layers = [list(self.model.model.children())[-2]]
                else:
                    target_layers = [list(self.model.model.children())[-1]]
                logger.info("Using fallback target layer")
            except Exception as e:
                logger.error(f"Could not determine target layers: {e}")
                raise
        
        return target_layers
    
    def generate_gradcam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        eigen_smooth: bool = False,
        aug_smooth: bool = False
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: Input tensor (1, C, H, W)
            target_class: Target class for visualization (None for predicted class)
            eigen_smooth: Whether to use eigen smoothing
            aug_smooth: Whether to use augmentation smoothing
            
        Returns:
            Grad-CAM heatmap as numpy array
        """
        input_tensor = input_tensor.to(self.device)
        
        # Get target class if not specified
        if target_class is None:
            with torch.no_grad():
                output = self.model(input_tensor)
                target_class = output.argmax(dim=1).item()
        
        # Create target for Grad-CAM
        targets = [ClassifierOutputTarget(target_class)]
        
        # Generate Grad-CAM
        grayscale_cam = self.grad_cam(
            input_tensor=input_tensor,
            targets=targets,
            eigen_smooth=eigen_smooth,
            aug_smooth=aug_smooth
        )
        
        # Return the first (and only) heatmap
        return grayscale_cam[0]
    
    def visualize_gradcam(
        self,
        image: Union[np.ndarray, Image.Image, torch.Tensor],
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Create Grad-CAM visualization overlay.
        
        Args:
            image: Original image (RGB format)
            input_tensor: Preprocessed input tensor
            target_class: Target class for visualization
            alpha: Transparency for overlay
            colormap: OpenCV colormap for heatmap
            
        Returns:
            Tuple of (original_image, visualization, confidence_score)
        """
        # Convert image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        elif isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).cpu().numpy()
            image = (image * 255).astype(np.uint8)
        
        # Ensure image is in correct format
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        # Generate Grad-CAM heatmap
        grayscale_cam = self.generate_gradcam(input_tensor, target_class)
        
        # Get model confidence
        with torch.no_grad():
            output = self.model(input_tensor.to(self.device))
            confidence = torch.softmax(output, dim=1).max().item()
        
        # Resize image to match heatmap if needed
        if image.shape[:2] != grayscale_cam.shape:
            image = cv2.resize(image, (grayscale_cam.shape[1], grayscale_cam.shape[0]))
        
        # Normalize image to [0, 1]
        image_normalized = image.astype(np.float32) / 255.0
        
        # Create visualization
        visualization = show_cam_on_image(
            image_normalized,
            grayscale_cam,
            use_rgb=True,
            colormap=colormap,
            image_weight=1-alpha
        )
        
        return image, visualization, confidence
    
    def compare_models_gradcam(
        self,
        models: Dict[str, BaseDeepfakeModel],
        image: Union[np.ndarray, Image.Image],
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Dict[str, Tuple[np.ndarray, float]]:
        """
        Compare Grad-CAM visualizations across multiple models.
        
        Args:
            models: Dictionary of models to compare
            image: Original image
            input_tensor: Preprocessed input tensor
            target_class: Target class for visualization
            
        Returns:
            Dictionary mapping model names to (visualization, confidence) tuples
        """
        results = {}
        
        for model_name, model in models.items():
            try:
                # Create visualizer for this model
                visualizer = GradCAMVisualizer(model, use_cuda=self.device.type == 'cuda')
                
                # Generate visualization
                _, visualization, confidence = visualizer.visualize_gradcam(
                    image, input_tensor, target_class
                )
                
                results[model_name] = (visualization, confidence)
                logger.info(f"Generated Grad-CAM for {model_name}")
                
            except Exception as e:
                logger.error(f"Error generating Grad-CAM for {model_name}: {e}")
                continue
        
        return results
    
    def plot_gradcam_comparison(
        self,
        original_image: np.ndarray,
        gradcam_results: Dict[str, Tuple[np.ndarray, float]],
        class_names: List[str] = None,
        target_class: int = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 5)
    ) -> plt.Figure:
        """
        Plot Grad-CAM comparison across models.
        
        Args:
            original_image: Original input image
            gradcam_results: Results from compare_models_gradcam
            class_names: Names of classes
            target_class: Target class index
            save_path: Path to save the plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if class_names is None:
            class_names = ['Real', 'Fake']
        
        num_models = len(gradcam_results)
        fig, axes = plt.subplots(1, num_models + 1, figsize=figsize)
        
        if num_models == 0:
            axes = [axes]
        
        # Plot original image
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Plot Grad-CAM for each model
        for i, (model_name, (visualization, confidence)) in enumerate(gradcam_results.items()):
            axes[i + 1].imshow(visualization)
            
            # Create title with prediction info
            predicted_class = class_names[target_class] if target_class is not None else "Unknown"
            title = f'{model_name}\n{predicted_class} ({confidence:.3f})'
            axes[i + 1].set_title(title)
            axes[i + 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Grad-CAM comparison saved to {save_path}")
        
        return fig


class ExplainabilityAnalyzer:
    """Comprehensive explainability analysis for deepfake detection."""
    
    def __init__(self, models: Dict[str, BaseDeepfakeModel]):
        """
        Args:
            models: Dictionary of models to analyze
        """
        self.models = models
        self.visualizers = {}
        
        # Create visualizers for each model
        for model_name, model in models.items():
            try:
                self.visualizers[model_name] = GradCAMVisualizer(model)
                logger.info(f"Created visualizer for {model_name}")
            except Exception as e:
                logger.error(f"Failed to create visualizer for {model_name}: {e}")
    
    def analyze_sample(
        self,
        image: Union[np.ndarray, Image.Image],
        input_tensor: torch.Tensor,
        true_label: Optional[int] = None,
        class_names: List[str] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive explainability analysis on a single sample.
        
        Args:
            image: Original image
            input_tensor: Preprocessed input tensor
            true_label: True label (optional)
            class_names: Names of classes
            
        Returns:
            Dictionary containing analysis results
        """
        if class_names is None:
            class_names = ['Real', 'Fake']
        
        results = {
            'predictions': {},
            'gradcam_visualizations': {},
            'agreement_analysis': {},
            'confidence_analysis': {}
        }
        
        # Get predictions from all models
        for model_name, model in self.models.items():
            with torch.no_grad():
                output = model(input_tensor.to(model.device))
                probabilities = torch.softmax(output, dim=1)
                predicted_class = output.argmax(dim=1).item()
                confidence = probabilities.max().item()
                
                results['predictions'][model_name] = {
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'probabilities': probabilities.cpu().numpy().tolist()
                }
        
        # Generate Grad-CAM visualizations
        if self.visualizers:
            first_visualizer = list(self.visualizers.values())[0]
            gradcam_results = first_visualizer.compare_models_gradcam(
                self.models, image, input_tensor
            )
            results['gradcam_visualizations'] = gradcam_results
        
        # Analyze model agreement
        predictions = [results['predictions'][name]['predicted_class'] for name in self.models.keys()]
        confidences = [results['predictions'][name]['confidence'] for name in self.models.keys()]
        
        results['agreement_analysis'] = {
            'unanimous_agreement': len(set(predictions)) == 1,
            'majority_prediction': max(set(predictions), key=predictions.count),
            'agreement_ratio': predictions.count(max(set(predictions), key=predictions.count)) / len(predictions)
        }
        
        results['confidence_analysis'] = {
            'mean_confidence': np.mean(confidences),
            'std_confidence': np.std(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences)
        }
        
        return results
    
    def batch_analysis(
        self,
        images: List[Union[np.ndarray, Image.Image]],
        input_tensors: List[torch.Tensor],
        true_labels: Optional[List[int]] = None,
        save_dir: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform explainability analysis on a batch of samples.
        
        Args:
            images: List of original images
            input_tensors: List of preprocessed input tensors
            true_labels: List of true labels (optional)
            save_dir: Directory to save visualizations
            
        Returns:
            List of analysis results
        """
        results = []
        
        for i, (image, input_tensor) in enumerate(zip(images, input_tensors)):
            true_label = true_labels[i] if true_labels else None
            
            # Analyze sample
            sample_results = self.analyze_sample(image, input_tensor, true_label)
            
            # Save visualization if directory provided
            if save_dir and 'gradcam_visualizations' in sample_results:
                import os
                os.makedirs(save_dir, exist_ok=True)
                
                # Create comparison plot
                if self.visualizers:
                    first_visualizer = list(self.visualizers.values())[0]
                    save_path = os.path.join(save_dir, f'gradcam_comparison_{i:04d}.png')
                    
                    first_visualizer.plot_gradcam_comparison(
                        image,
                        sample_results['gradcam_visualizations'],
                        save_path=save_path
                    )
            
            results.append(sample_results)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(images)} samples")
        
        return results
