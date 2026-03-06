"""
Dataset classes for deepfake detection.

This module contains custom dataset implementations for FaceForensics++
and CelebDF datasets with proper video frame extraction and preprocessing.
"""

import os
import cv2
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from typing import List, Tuple, Optional, Dict, Any
import json
import logging

logger = logging.getLogger(__name__)


class DeepfakeDataset(Dataset):
    """Base dataset class for deepfake detection with face extraction support."""

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform: Optional[transforms.Compose] = None,
        target_transform: Optional[transforms.Compose] = None,
        image_size: int = 224,
        max_frames_per_video: int = 10,
        use_extracted_faces: bool = True,
        min_face_quality: float = 0.3,
        faces_per_video: int = 5
    ):
        """
        Args:
            data_dir: Root directory containing the dataset
            split: Dataset split ('train', 'holdout', 'test')
            transform: Image transformations
            target_transform: Target transformations
            image_size: Target image size for resizing
            max_frames_per_video: Maximum frames to extract per video
            use_extracted_faces: Whether to use pre-extracted face crops
            min_face_quality: Minimum quality threshold for face selection
            faces_per_video: Number of faces to sample per video
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.image_size = image_size
        self.max_frames_per_video = max_frames_per_video
        self.use_extracted_faces = use_extracted_faces
        self.min_face_quality = min_face_quality
        self.faces_per_video = faces_per_video
        
        # Default transforms if none provided
        if self.transform is None:
            self.transform = self._get_default_transforms()
            
        self.samples = []
        self.labels = []
        self.class_to_idx = {'real': 0, 'fake': 1}
        
        self._load_dataset()
    
    def _get_default_transforms(self) -> transforms.Compose:
        """Get default image transformations."""
        if self.split == 'train':
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def _load_dataset(self):
        """Load dataset samples and labels. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _load_dataset method")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a sample from the dataset."""
        image_path = self.samples[idx]
        label = self.labels[idx]
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.warning(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (self.image_size, self.image_size), (0, 0, 0))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            label = self.target_transform(label)
            
        return image, label

    def _load_extracted_faces(self, video_dir: str, label: int) -> List[Tuple[str, int]]:
        """
        Load pre-extracted face crops from a video directory.

        Args:
            video_dir: Directory containing extracted face crops
            label: Label for the video (0=real, 1=fake)

        Returns:
            List of (face_path, label) tuples
        """
        face_samples = []

        if not os.path.exists(video_dir):
            return face_samples

        # Find all face crop files
        face_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            face_files.extend(Path(video_dir).glob(f'*{ext}'))

        # Filter by quality if specified
        quality_filtered = []
        for face_file in face_files:
            # Extract quality from filename (format: frame_XXX_face_XX_qX.XX.jpg)
            filename = face_file.stem
            if '_q' in filename:
                try:
                    quality_str = filename.split('_q')[-1]
                    quality = float(quality_str)
                    if quality >= self.min_face_quality:
                        quality_filtered.append((face_file, quality))
                except ValueError:
                    # If quality parsing fails, include the face
                    quality_filtered.append((face_file, 1.0))
            else:
                # No quality info, include the face
                quality_filtered.append((face_file, 1.0))

        # Sort by quality (highest first) and select top faces
        quality_filtered.sort(key=lambda x: x[1], reverse=True)
        selected_faces = quality_filtered[:self.faces_per_video]

        # Add to samples
        for face_file, quality in selected_faces:
            face_samples.append((str(face_file), label))

        return face_samples


class FaceForensicsDataset(DeepfakeDataset):
    """Dataset class for FaceForensics++ dataset."""
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        categories: List[str] = None,
        videos_per_category: int = 100,
        **kwargs
    ):
        """
        Args:
            data_dir: Root directory containing FaceForensics++ dataset
            split: Dataset split ('train', 'holdout', 'test')
            categories: List of categories to include
            videos_per_category: Number of videos per category
        """
        if categories is None:
            categories = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "Original"]
        
        self.categories = categories
        self.videos_per_category = videos_per_category
        
        super().__init__(data_dir, split, **kwargs)
    
    def _load_dataset(self):
        """Load FaceForensics++ dataset samples."""
        split_file = os.path.join(self.data_dir, 'splits', f'{self.split}_split.txt')
        
        if os.path.exists(split_file):
            # Load from existing split file
            with open(split_file, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    image_path, label = parts
                    if os.path.exists(image_path):
                        self.samples.append(image_path)
                        self.labels.append(int(label))
        else:
            # Create dataset from directory structure
            self._create_from_directory()
    
    def _create_from_directory(self):
        """Create dataset from directory structure."""
        processed_dir = os.path.join(self.data_dir, 'processed', self.split)
        
        for category in self.categories:
            category_dir = os.path.join(processed_dir, category)
            if not os.path.exists(category_dir):
                logger.warning(f"Category directory not found: {category_dir}")
                continue
            
            # Determine label (Original is real, others are fake)
            label = 0 if category == "Original" else 1
            
            # Get all image files in category directory
            for root, dirs, files in os.walk(category_dir):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_path = os.path.join(root, file)
                        self.samples.append(image_path)
                        self.labels.append(label)
        
        logger.info(f"Loaded {len(self.samples)} samples for {self.split} split")


class CelebDFDataset(DeepfakeDataset):
    """Dataset class for CelebDF dataset."""
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        **kwargs
    ):
        """
        Args:
            data_dir: Root directory containing CelebDF dataset
            split: Dataset split ('train', 'holdout', 'test')
        """
        super().__init__(data_dir, split, **kwargs)
    
    def _load_dataset(self):
        """Load CelebDF dataset samples."""
        split_file = os.path.join(self.data_dir, 'splits', f'{self.split}_split.txt')
        
        if os.path.exists(split_file):
            # Load from existing split file
            with open(split_file, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    image_path, label = parts
                    if os.path.exists(image_path):
                        self.samples.append(image_path)
                        self.labels.append(int(label))
        else:
            # Create dataset from directory structure
            self._create_from_directory()
    
    def _create_from_directory(self):
        """Create dataset from directory structure."""
        processed_dir = os.path.join(self.data_dir, 'processed', self.split)
        
        categories = ['Real', 'Fake']
        for category in categories:
            category_dir = os.path.join(processed_dir, category)
            if not os.path.exists(category_dir):
                logger.warning(f"Category directory not found: {category_dir}")
                continue
            
            # Determine label
            label = 0 if category == "Real" else 1
            
            # Get all image files in category directory
            for root, dirs, files in os.walk(category_dir):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_path = os.path.join(root, file)
                        self.samples.append(image_path)
                        self.labels.append(label)
        
        logger.info(f"Loaded {len(self.samples)} samples for {self.split} split")


def get_dataset_stats(dataset: Dataset) -> Dict[str, Any]:
    """Get statistics about a dataset."""
    labels = [dataset.labels[i] for i in range(len(dataset))]
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    stats = {
        'total_samples': len(dataset),
        'num_classes': len(unique_labels),
        'class_distribution': dict(zip(unique_labels, counts)),
        'class_balance': counts / len(labels)
    }
    
    return stats
