"""
Data splitting utilities for deepfake detection.

This module provides functionality to split datasets into train/holdout/test sets
with proper stratification and reproducibility.
"""

import os
import random
import numpy as np
from typing import List, Tuple, Dict, Any
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


class DataSplitter:
    """Utility class for splitting datasets into train/holdout/test sets."""
    
    def __init__(
        self,
        train_ratio: float = 0.6,
        holdout_ratio: float = 0.2,
        test_ratio: float = 0.2,
        random_seed: int = 42,
        stratify: bool = True
    ):
        """
        Args:
            train_ratio: Proportion of data for training base models
            holdout_ratio: Proportion of data for training meta-learner
            test_ratio: Proportion of data for final evaluation
            random_seed: Random seed for reproducibility
            stratify: Whether to maintain class balance across splits
        """
        # Validate ratios
        if abs(train_ratio + holdout_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Train, holdout, and test ratios must sum to 1.0")
        
        self.train_ratio = train_ratio
        self.holdout_ratio = holdout_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        self.stratify = stratify
        
        # Set random seeds for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    def split_dataset(
        self,
        samples: List[str],
        labels: List[int],
        output_dir: str
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Split dataset into train/holdout/test sets.
        
        Args:
            samples: List of sample file paths
            labels: List of corresponding labels
            output_dir: Directory to save split files
            
        Returns:
            Tuple of (train_files, holdout_files, test_files)
        """
        if len(samples) != len(labels):
            raise ValueError("Number of samples and labels must match")
        
        logger.info(f"Splitting {len(samples)} samples into train/holdout/test sets")
        logger.info(f"Ratios: train={self.train_ratio}, holdout={self.holdout_ratio}, test={self.test_ratio}")
        
        # Convert to numpy arrays for easier manipulation
        samples = np.array(samples)
        labels = np.array(labels)
        
        # First split: separate test set
        if self.stratify:
            train_holdout_samples, test_samples, train_holdout_labels, test_labels = train_test_split(
                samples, labels,
                test_size=self.test_ratio,
                random_state=self.random_seed,
                stratify=labels
            )
        else:
            train_holdout_samples, test_samples, train_holdout_labels, test_labels = train_test_split(
                samples, labels,
                test_size=self.test_ratio,
                random_state=self.random_seed
            )
        
        # Second split: separate train and holdout sets
        holdout_size = self.holdout_ratio / (self.train_ratio + self.holdout_ratio)
        
        if self.stratify:
            train_samples, holdout_samples, train_labels, holdout_labels = train_test_split(
                train_holdout_samples, train_holdout_labels,
                test_size=holdout_size,
                random_state=self.random_seed,
                stratify=train_holdout_labels
            )
        else:
            train_samples, holdout_samples, train_labels, holdout_labels = train_test_split(
                train_holdout_samples, train_holdout_labels,
                test_size=holdout_size,
                random_state=self.random_seed
            )
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save split files
        train_file = os.path.join(output_dir, 'train_split.txt')
        holdout_file = os.path.join(output_dir, 'holdout_split.txt')
        test_file = os.path.join(output_dir, 'test_split.txt')
        
        self._save_split_file(train_file, train_samples, train_labels)
        self._save_split_file(holdout_file, holdout_samples, holdout_labels)
        self._save_split_file(test_file, test_samples, test_labels)
        
        # Log split statistics
        self._log_split_stats(train_samples, train_labels, "Train")
        self._log_split_stats(holdout_samples, holdout_labels, "Holdout")
        self._log_split_stats(test_samples, test_labels, "Test")
        
        return train_file, holdout_file, test_file
    
    def _save_split_file(self, filename: str, samples: np.ndarray, labels: np.ndarray):
        """Save split file with sample paths and labels."""
        with open(filename, 'w') as f:
            for sample, label in zip(samples, labels):
                f.write(f"{sample}\t{label}\n")
        
        logger.info(f"Saved {len(samples)} samples to {filename}")
    
    def _log_split_stats(self, samples: np.ndarray, labels: np.ndarray, split_name: str):
        """Log statistics for a data split."""
        unique_labels, counts = np.unique(labels, return_counts=True)
        total = len(samples)
        
        logger.info(f"{split_name} split statistics:")
        logger.info(f"  Total samples: {total}")
        for label, count in zip(unique_labels, counts):
            percentage = (count / total) * 100
            logger.info(f"  Class {label}: {count} samples ({percentage:.1f}%)")
    
    def load_split_files(self, splits_dir: str) -> Dict[str, Tuple[List[str], List[int]]]:
        """
        Load existing split files.
        
        Args:
            splits_dir: Directory containing split files
            
        Returns:
            Dictionary with split data
        """
        splits = {}
        
        for split_name in ['train', 'holdout', 'test']:
            split_file = os.path.join(splits_dir, f'{split_name}_split.txt')
            
            if os.path.exists(split_file):
                samples, labels = self._load_split_file(split_file)
                splits[split_name] = (samples, labels)
                logger.info(f"Loaded {len(samples)} samples from {split_file}")
            else:
                logger.warning(f"Split file not found: {split_file}")
        
        return splits
    
    def _load_split_file(self, filename: str) -> Tuple[List[str], List[int]]:
        """Load split file and return samples and labels."""
        samples = []
        labels = []
        
        with open(filename, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    sample_path, label = parts
                    samples.append(sample_path)
                    labels.append(int(label))
        
        return samples, labels
    
    def validate_splits(self, splits_dir: str) -> bool:
        """
        Validate that split files exist and contain data.
        
        Args:
            splits_dir: Directory containing split files
            
        Returns:
            True if all splits are valid, False otherwise
        """
        required_files = ['train_split.txt', 'holdout_split.txt', 'test_split.txt']
        
        for filename in required_files:
            filepath = os.path.join(splits_dir, filename)
            
            if not os.path.exists(filepath):
                logger.error(f"Missing split file: {filepath}")
                return False
            
            # Check if file has content
            with open(filepath, 'r') as f:
                lines = f.readlines()
                if len(lines) == 0:
                    logger.error(f"Empty split file: {filepath}")
                    return False
        
        logger.info("All split files are valid")
        return True


def create_balanced_splits(
    data_dir: str,
    categories: List[str],
    output_dir: str,
    train_ratio: float = 0.6,
    holdout_ratio: float = 0.2,
    test_ratio: float = 0.2,
    random_seed: int = 42
) -> None:
    """
    Create balanced splits for deepfake detection datasets.
    
    Args:
        data_dir: Root directory containing processed data
        categories: List of data categories
        output_dir: Directory to save split files
        train_ratio: Training set ratio
        holdout_ratio: Holdout set ratio
        test_ratio: Test set ratio
        random_seed: Random seed for reproducibility
    """
    splitter = DataSplitter(
        train_ratio=train_ratio,
        holdout_ratio=holdout_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed,
        stratify=True
    )
    
    # Collect all samples and labels
    all_samples = []
    all_labels = []
    
    for category in categories:
        category_dir = os.path.join(data_dir, category)
        if not os.path.exists(category_dir):
            logger.warning(f"Category directory not found: {category_dir}")
            continue
        
        # Determine label (assuming Original/Real is 0, others are 1)
        label = 0 if category in ["Original", "Real"] else 1
        
        # Collect all image files
        for root, dirs, files in os.walk(category_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(root, file)
                    all_samples.append(image_path)
                    all_labels.append(label)
    
    # Create splits
    splitter.split_dataset(all_samples, all_labels, output_dir)
    logger.info(f"Created balanced splits in {output_dir}")
