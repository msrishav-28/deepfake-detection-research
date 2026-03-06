#!/usr/bin/env python3
"""
Inference pipeline for deepfake detection ensemble.

This script provides a complete inference pipeline that loads the trained
ensemble and makes predictions on new data.

Usage:
    python scripts/evaluation/inference_pipeline.py --ensemble-dir models/ensemble --input-dir data/test
    python scripts/evaluation/inference_pipeline.py --config config.yaml --input path/to/image.jpg
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import logging
from tqdm import tqdm
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from deepfake_detection.models.model_factory import ModelFactory
from deepfake_detection.models.ensemble import StackedEnsemble, MetaLearner
from deepfake_detection.data.augmentations import get_augmentation_transforms
from deepfake_detection.utils.training_utils import setup_logging

logger = logging.getLogger(__name__)


class DeepfakeInferencePipeline:
    """Complete inference pipeline for deepfake detection."""
    
    def __init__(
        self,
        ensemble_dir: str,
        config: dict,
        device: torch.device = None
    ):
        """
        Args:
            ensemble_dir: Directory containing trained ensemble
            config: Configuration dictionary
            device: Device to run inference on
        """
        self.ensemble_dir = ensemble_dir
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.ensemble = None
        self.transform = None
        self.class_names = ['Real', 'Fake']
        
        self._load_ensemble()
        self._setup_transforms()
        
        logger.info(f"Inference pipeline initialized on {self.device}")
    
    def _load_ensemble(self):
        """Load the trained ensemble."""
        logger.info(f"Loading ensemble from {self.ensemble_dir}")
        
        # Create base models
        factory = ModelFactory(self.config)
        base_models = factory.create_all_base_models(self.device)
        
        # Create ensemble
        self.ensemble = StackedEnsemble(base_models, device=self.device)
        
        # Load ensemble weights and meta-learner
        self.ensemble.load_ensemble(self.ensemble_dir)
        
        # Set to evaluation mode
        self.ensemble.eval()
        
        logger.info("Ensemble loaded successfully")
    
    def _setup_transforms(self):
        """Setup image preprocessing transforms."""
        self.transform = get_augmentation_transforms(
            split='test',  # Use test transforms (no augmentation)
            image_size=224,
            augmentation_config=None
        )
        
        logger.info("Image transforms initialized")
    
    def predict_single_image(
        self,
        image_path: str,
        return_probabilities: bool = True,
        return_contributions: bool = False
    ) -> dict:
        """
        Make prediction on a single image.
        
        Args:
            image_path: Path to the image file
            return_probabilities: Whether to return class probabilities
            return_contributions: Whether to return individual model contributions
            
        Returns:
            Dictionary containing prediction results
        """
        # Load and preprocess image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return {'error': f"Failed to load image: {e}"}
        
        # Apply transforms
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            predicted_labels, probabilities = self.ensemble.predict(input_tensor)
        
        # Prepare results
        results = {
            'image_path': image_path,
            'predicted_class': self.class_names[predicted_labels[0]],
            'predicted_label': int(predicted_labels[0]),
            'confidence': float(np.max(probabilities[0]))
        }
        
        if return_probabilities:
            results['probabilities'] = {
                self.class_names[i]: float(probabilities[0][i])
                for i in range(len(self.class_names))
            }
        
        if return_contributions:
            contributions = self.ensemble.get_model_contributions(input_tensor)
            results['model_contributions'] = contributions
        
        return results
    
    def predict_batch(
        self,
        image_paths: list,
        batch_size: int = 32,
        return_probabilities: bool = True
    ) -> list:
        """
        Make predictions on a batch of images.
        
        Args:
            image_paths: List of image file paths
            batch_size: Batch size for processing
            return_probabilities: Whether to return class probabilities
            
        Returns:
            List of prediction results
        """
        results = []
        
        # Process in batches
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches"):
            batch_paths = image_paths[i:i + batch_size]
            batch_tensors = []
            valid_paths = []
            
            # Load and preprocess batch
            for image_path in batch_paths:
                try:
                    image = Image.open(image_path).convert('RGB')
                    tensor = self.transform(image)
                    batch_tensors.append(tensor)
                    valid_paths.append(image_path)
                except Exception as e:
                    logger.warning(f"Skipping {image_path}: {e}")
                    results.append({
                        'image_path': image_path,
                        'error': f"Failed to load image: {e}"
                    })
            
            if not batch_tensors:
                continue
            
            # Stack tensors and move to device
            batch_tensor = torch.stack(batch_tensors).to(self.device)
            
            # Make predictions
            with torch.no_grad():
                predicted_labels, probabilities = self.ensemble.predict(batch_tensor)
            
            # Process results
            for j, image_path in enumerate(valid_paths):
                result = {
                    'image_path': image_path,
                    'predicted_class': self.class_names[predicted_labels[j]],
                    'predicted_label': int(predicted_labels[j]),
                    'confidence': float(np.max(probabilities[j]))
                }
                
                if return_probabilities:
                    result['probabilities'] = {
                        self.class_names[k]: float(probabilities[j][k])
                        for k in range(len(self.class_names))
                    }
                
                results.append(result)
        
        return results
    
    def predict_directory(
        self,
        input_dir: str,
        output_file: str = None,
        batch_size: int = 32,
        file_extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp')
    ) -> list:
        """
        Make predictions on all images in a directory.
        
        Args:
            input_dir: Directory containing images
            output_file: Optional file to save results
            batch_size: Batch size for processing
            file_extensions: Supported file extensions
            
        Returns:
            List of prediction results
        """
        # Find all image files
        image_paths = []
        for ext in file_extensions:
            image_paths.extend(Path(input_dir).glob(f'**/*{ext}'))
            image_paths.extend(Path(input_dir).glob(f'**/*{ext.upper()}'))
        
        image_paths = [str(path) for path in image_paths]
        
        if not image_paths:
            logger.warning(f"No images found in {input_dir}")
            return []
        
        logger.info(f"Found {len(image_paths)} images in {input_dir}")
        
        # Make predictions
        results = self.predict_batch(image_paths, batch_size)
        
        # Save results if requested
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {output_file}")
        
        return results
    
    def analyze_predictions(self, results: list) -> dict:
        """
        Analyze prediction results.
        
        Args:
            results: List of prediction results
            
        Returns:
            Analysis summary
        """
        valid_results = [r for r in results if 'error' not in r]
        
        if not valid_results:
            return {'error': 'No valid predictions to analyze'}
        
        # Count predictions by class
        class_counts = {}
        confidence_scores = []
        
        for result in valid_results:
            predicted_class = result['predicted_class']
            confidence = result['confidence']
            
            class_counts[predicted_class] = class_counts.get(predicted_class, 0) + 1
            confidence_scores.append(confidence)
        
        # Calculate statistics
        analysis = {
            'total_predictions': len(valid_results),
            'class_distribution': class_counts,
            'class_percentages': {
                cls: (count / len(valid_results)) * 100
                for cls, count in class_counts.items()
            },
            'confidence_stats': {
                'mean': float(np.mean(confidence_scores)),
                'std': float(np.std(confidence_scores)),
                'min': float(np.min(confidence_scores)),
                'max': float(np.max(confidence_scores))
            },
            'errors': len(results) - len(valid_results)
        }
        
        return analysis


def main():
    parser = argparse.ArgumentParser(description='Deepfake detection inference pipeline')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--ensemble-dir', type=str,
                        help='Directory containing trained ensemble')
    parser.add_argument('--input', type=str, required=True,
                        help='Input image file or directory')
    parser.add_argument('--output', type=str,
                        help='Output file for results (JSON format)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for processing')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cuda, cpu, or auto)')
    parser.add_argument('--analyze', action='store_true',
                        help='Analyze prediction results')
    parser.add_argument('--contributions', action='store_true',
                        help='Include model contributions (single image only)')
    
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
    
    # Set ensemble directory from config if not provided
    if args.ensemble_dir is None:
        args.ensemble_dir = os.path.join(config['paths']['models_dir'], 'ensemble')
    
    # Initialize pipeline
    pipeline = DeepfakeInferencePipeline(args.ensemble_dir, config, device)
    
    # Make predictions
    if os.path.isfile(args.input):
        # Single image prediction
        logger.info(f"Processing single image: {args.input}")
        result = pipeline.predict_single_image(
            args.input,
            return_probabilities=True,
            return_contributions=args.contributions
        )
        results = [result]
        
        # Print result
        if 'error' not in result:
            print(f"\nPrediction for {args.input}:")
            print(f"  Class: {result['predicted_class']}")
            print(f"  Confidence: {result['confidence']:.4f}")
            if 'probabilities' in result:
                print("  Probabilities:")
                for cls, prob in result['probabilities'].items():
                    print(f"    {cls}: {prob:.4f}")
        else:
            print(f"Error: {result['error']}")
    
    elif os.path.isdir(args.input):
        # Directory prediction
        logger.info(f"Processing directory: {args.input}")
        results = pipeline.predict_directory(
            args.input,
            output_file=args.output,
            batch_size=args.batch_size
        )
        
        print(f"\nProcessed {len(results)} images from {args.input}")
    
    else:
        logger.error(f"Input path does not exist: {args.input}")
        return
    
    # Analyze results if requested
    if args.analyze and len(results) > 1:
        analysis = pipeline.analyze_predictions(results)
        
        print("\n" + "="*50)
        print("PREDICTION ANALYSIS")
        print("="*50)
        print(f"Total predictions: {analysis['total_predictions']}")
        print(f"Errors: {analysis['errors']}")
        print("\nClass distribution:")
        for cls, percentage in analysis['class_percentages'].items():
            print(f"  {cls}: {percentage:.1f}%")
        print(f"\nConfidence statistics:")
        print(f"  Mean: {analysis['confidence_stats']['mean']:.4f}")
        print(f"  Std:  {analysis['confidence_stats']['std']:.4f}")
        print(f"  Range: [{analysis['confidence_stats']['min']:.4f}, {analysis['confidence_stats']['max']:.4f}]")
    
    # Save results if output file specified and not already saved
    if args.output and os.path.isfile(args.input):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()
