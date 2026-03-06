#!/usr/bin/env python3
"""
Create data splits for deepfake detection research.

This script implements the 3-way splitting strategy:
- Training set (60%): For training base models
- Hold-out set (20%): For training meta-learner
- Test set (20%): For final evaluation

Usage:
    python scripts/data_preparation/create_splits.py --config config.yaml
    python scripts/data_preparation/create_splits.py --data-dir data/processed --output-dir data/splits
"""

import os
import sys
import argparse
import yaml
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from deepfake_detection.data.data_splitter import DataSplitter, create_balanced_splits

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def collect_dataset_samples(data_dir: str, dataset_type: str = 'faceforensics') -> tuple:
    """
    Collect all samples and labels from processed dataset.
    
    Args:
        data_dir: Directory containing processed dataset
        dataset_type: Type of dataset ('faceforensics' or 'celebdf')
        
    Returns:
        Tuple of (samples, labels)
    """
    samples = []
    labels = []
    
    if dataset_type == 'faceforensics':
        categories = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "Original"]
    elif dataset_type == 'celebdf':
        categories = ["Real", "Fake"]
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    for category in categories:
        category_dir = os.path.join(data_dir, category)
        if not os.path.exists(category_dir):
            logger.warning(f"Category directory not found: {category_dir}")
            continue
        
        # Determine label (Original/Real is 0, others are 1)
        label = 0 if category in ["Original", "Real"] else 1
        
        # Collect all image files
        for root, dirs, files in os.walk(category_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(root, file)
                    samples.append(image_path)
                    labels.append(label)
        
        logger.info(f"Found {sum(1 for l in labels if l == label)} samples in {category}")
    
    logger.info(f"Total samples collected: {len(samples)}")
    return samples, labels


def create_splits_for_dataset(
    data_dir: str,
    output_dir: str,
    dataset_type: str,
    config: dict
) -> None:
    """
    Create splits for a specific dataset.
    
    Args:
        data_dir: Directory containing processed dataset
        output_dir: Directory to save split files
        dataset_type: Type of dataset
        config: Configuration dictionary
    """
    logger.info(f"Creating splits for {dataset_type} dataset")
    
    # Collect samples and labels
    samples, labels = collect_dataset_samples(data_dir, dataset_type)
    
    if len(samples) == 0:
        logger.error(f"No samples found in {data_dir}")
        return
    
    # Create data splitter
    data_config = config['data']
    splitter = DataSplitter(
        train_ratio=data_config['splits']['train_ratio'],
        holdout_ratio=data_config['splits']['holdout_ratio'],
        test_ratio=data_config['splits']['test_ratio'],
        random_seed=data_config['splits']['random_seed'],
        stratify=True
    )
    
    # Create output directory
    dataset_output_dir = os.path.join(output_dir, dataset_type)
    os.makedirs(dataset_output_dir, exist_ok=True)
    
    # Create splits
    train_file, holdout_file, test_file = splitter.split_dataset(
        samples, labels, dataset_output_dir
    )
    
    logger.info(f"Created splits for {dataset_type}:")
    logger.info(f"  Train: {train_file}")
    logger.info(f"  Holdout: {holdout_file}")
    logger.info(f"  Test: {test_file}")
    
    # Validate splits
    if splitter.validate_splits(dataset_output_dir):
        logger.info(f"All splits for {dataset_type} are valid")
    else:
        logger.error(f"Invalid splits for {dataset_type}")


def merge_dataset_splits(
    splits_dirs: list,
    output_dir: str,
    dataset_names: list
) -> None:
    """
    Merge splits from multiple datasets into combined splits.
    
    Args:
        splits_dirs: List of directories containing dataset splits
        output_dir: Directory to save merged splits
        dataset_names: Names of datasets being merged
    """
    logger.info(f"Merging splits from datasets: {dataset_names}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    for split_name in ['train', 'holdout', 'test']:
        merged_samples = []
        merged_labels = []
        
        for splits_dir, dataset_name in zip(splits_dirs, dataset_names):
            split_file = os.path.join(splits_dir, f'{split_name}_split.txt')
            
            if os.path.exists(split_file):
                with open(split_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) == 2:
                            sample_path, label = parts
                            merged_samples.append(sample_path)
                            merged_labels.append(int(label))
                
                logger.info(f"Added {len(merged_samples)} samples from {dataset_name} {split_name} split")
        
        # Save merged split
        merged_file = os.path.join(output_dir, f'{split_name}_split.txt')
        with open(merged_file, 'w') as f:
            for sample, label in zip(merged_samples, merged_labels):
                f.write(f"{sample}\t{label}\n")
        
        logger.info(f"Saved merged {split_name} split with {len(merged_samples)} samples to {merged_file}")


def main():
    parser = argparse.ArgumentParser(description='Create data splits for deepfake detection')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--data-dir', type=str,
                        help='Directory containing processed datasets')
    parser.add_argument('--output-dir', type=str,
                        help='Directory to save split files')
    parser.add_argument('--dataset-type', type=str, choices=['faceforensics', 'celebdf', 'both'],
                        default='both', help='Type of dataset to process')
    parser.add_argument('--merge-datasets', action='store_true',
                        help='Merge splits from multiple datasets')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set default paths from config if not provided
    if args.data_dir is None:
        args.data_dir = os.path.join(config['paths']['data_dir'], 'processed')
    
    if args.output_dir is None:
        args.output_dir = os.path.join(config['paths']['data_dir'], 'splits')
    
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Process datasets
    if args.dataset_type in ['faceforensics', 'both']:
        faceforensics_dir = os.path.join(args.data_dir, 'faceforensics')
        if os.path.exists(faceforensics_dir):
            create_splits_for_dataset(
                faceforensics_dir,
                args.output_dir,
                'faceforensics',
                config
            )
        else:
            logger.warning(f"FaceForensics++ directory not found: {faceforensics_dir}")
    
    if args.dataset_type in ['celebdf', 'both']:
        celebdf_dir = os.path.join(args.data_dir, 'celebdf')
        if os.path.exists(celebdf_dir):
            create_splits_for_dataset(
                celebdf_dir,
                args.output_dir,
                'celebdf',
                config
            )
        else:
            logger.warning(f"CelebDF directory not found: {celebdf_dir}")
    
    # Merge datasets if requested
    if args.merge_datasets and args.dataset_type == 'both':
        splits_dirs = [
            os.path.join(args.output_dir, 'faceforensics'),
            os.path.join(args.output_dir, 'celebdf')
        ]
        dataset_names = ['faceforensics', 'celebdf']
        
        # Filter existing directories
        existing_dirs = []
        existing_names = []
        for splits_dir, name in zip(splits_dirs, dataset_names):
            if os.path.exists(splits_dir):
                existing_dirs.append(splits_dir)
                existing_names.append(name)
        
        if len(existing_dirs) > 1:
            merged_output_dir = os.path.join(args.output_dir, 'merged')
            merge_dataset_splits(existing_dirs, merged_output_dir, existing_names)
        else:
            logger.warning("Need at least 2 datasets to merge splits")
    
    logger.info("Data splitting completed successfully!")


if __name__ == '__main__':
    main()
