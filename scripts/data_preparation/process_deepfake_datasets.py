#!/usr/bin/env python3
"""
Optimized Processing Script for FaceForensics++ and CelebDF Datasets

This script is specifically optimized for the characteristics of deepfake datasets:
- FaceForensics++: 10-15 second MP4 clips, ~2MB (c23) or ~0.4MB (c40)
- CelebDF: ~13 second MP4 clips, high quality, few MB per video

Usage:
    python scripts/data_preparation/process_deepfake_datasets.py --config config.yaml
"""

import os
import sys
import argparse
import yaml
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import json
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_dataset_config(dataset_name: str, base_config: Dict) -> Dict:
    """Get optimized configuration for specific dataset."""
    face_config = base_config.get('data', {}).get('face_extraction', {})
    
    if dataset_name == 'faceforensics':
        # Optimized for compressed MP4s, 10-15 seconds
        return {
            **face_config,
            'method': 'opencv',  # Fast and reliable for compressed videos
            'min_face_quality': 0.25,  # Lower threshold for compression artifacts
            'faces_per_video': 6,  # Conservative due to compression
            'max_frames_per_video': 12,  # ~1 frame per second for 10-15s clips
            'target_fps': 1.0,
            'confidence_threshold': 0.6  # Lower for compressed videos
        }
    elif dataset_name == 'celebdf':
        # Optimized for high-quality MP4s, ~13 seconds
        return {
            **face_config,
            'method': 'mtcnn',  # Higher accuracy for high-quality videos
            'min_face_quality': 0.4,  # Higher threshold for quality videos
            'faces_per_video': 10,  # More faces from quality source
            'max_frames_per_video': 15,  # ~1 frame per second for 13s clips
            'target_fps': 1.0,
            'confidence_threshold': 0.7
        }
    else:
        return face_config


def estimate_processing_time(video_count: int, dataset_name: str) -> Tuple[int, int]:
    """Estimate processing time based on dataset characteristics."""
    if dataset_name == 'faceforensics':
        # OpenCV processing: ~2-3 seconds per video
        seconds_per_video = 2.5
    elif dataset_name == 'celebdf':
        # MTCNN processing: ~8-12 seconds per video
        seconds_per_video = 10
    else:
        seconds_per_video = 5
    
    total_seconds = int(video_count * seconds_per_video)
    return total_seconds, total_seconds // 60


def process_faceforensics_category(
    category: str,
    input_dir: str,
    output_dir: str,
    config: Dict,
    dry_run: bool = False
) -> bool:
    """Process a single FaceForensics++ category."""
    logger.info(f"Processing FaceForensics++ category: {category}")
    
    category_input = Path(input_dir) / category
    category_output = Path(output_dir) / category
    
    if not category_input.exists():
        logger.warning(f"Category directory not found: {category_input}")
        return False
    
    # Count videos
    video_files = list(category_input.glob('*.mp4'))
    logger.info(f"Found {len(video_files)} MP4 files in {category}")
    
    if len(video_files) == 0:
        logger.warning(f"No MP4 files found in {category_input}")
        return False
    
    # Estimate processing time
    total_seconds, total_minutes = estimate_processing_time(len(video_files), 'faceforensics')
    logger.info(f"Estimated processing time: {total_minutes} minutes ({total_seconds} seconds)")
    
    if dry_run:
        logger.info(f"[DRY RUN] Would process {len(video_files)} videos")
        return True
    
    # Run face extraction
    extract_script = project_root / 'scripts' / 'data_preparation' / 'extract_faces_from_videos.py'
    
    command = [
        sys.executable, str(extract_script),
        '--input-dir', str(category_input),
        '--output-dir', str(category_output),
        '--method', config['method'],
        '--config', 'config.yaml'
    ]
    
    try:
        start_time = time.time()
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        end_time = time.time()
        
        actual_time = int(end_time - start_time)
        logger.info(f"✅ {category} completed in {actual_time // 60}m {actual_time % 60}s")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Error processing {category}: {e}")
        logger.error(f"Error output: {e.stderr}")
        return False


def process_celebdf_category(
    category: str,
    input_dir: str,
    output_dir: str,
    config: Dict,
    dry_run: bool = False
) -> bool:
    """Process a CelebDF category (Real or Fake)."""
    logger.info(f"Processing CelebDF category: {category}")
    
    category_input = Path(input_dir) / category
    category_output = Path(output_dir) / category
    
    if not category_input.exists():
        logger.warning(f"Category directory not found: {category_input}")
        return False
    
    # Count videos
    video_files = list(category_input.glob('*.mp4'))
    logger.info(f"Found {len(video_files)} MP4 files in {category}")
    
    if len(video_files) == 0:
        logger.warning(f"No MP4 files found in {category_input}")
        return False
    
    # Estimate processing time
    total_seconds, total_minutes = estimate_processing_time(len(video_files), 'celebdf')
    logger.info(f"Estimated processing time: {total_minutes} minutes ({total_seconds} seconds)")
    
    if dry_run:
        logger.info(f"[DRY RUN] Would process {len(video_files)} videos")
        return True
    
    # Run face extraction
    extract_script = project_root / 'scripts' / 'data_preparation' / 'extract_faces_from_videos.py'
    
    command = [
        sys.executable, str(extract_script),
        '--input-dir', str(category_input),
        '--output-dir', str(category_output),
        '--method', config['method'],
        '--config', 'config.yaml'
    ]
    
    try:
        start_time = time.time()
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        end_time = time.time()
        
        actual_time = int(end_time - start_time)
        logger.info(f"✅ {category} completed in {actual_time // 60}m {actual_time % 60}s")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Error processing {category}: {e}")
        logger.error(f"Error output: {e.stderr}")
        return False


def create_processing_summary(results: Dict, output_dir: str):
    """Create a summary of the processing results."""
    summary = {
        'datasets_processed': list(results.keys()),
        'total_categories': sum(len(categories) for categories in results.values()),
        'successful_categories': sum(
            sum(1 for success in categories.values() if success)
            for categories in results.values()
        ),
        'processing_results': results,
        'timestamp': time.time()
    }
    
    summary_file = Path(output_dir) / 'processing_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Processing summary saved to: {summary_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("DATASET PROCESSING SUMMARY")
    print("="*60)
    
    for dataset, categories in results.items():
        print(f"\n{dataset.upper()}:")
        for category, success in categories.items():
            status = "✅ SUCCESS" if success else "❌ FAILED"
            print(f"  {category:15}: {status}")
    
    total_success = summary['successful_categories']
    total_categories = summary['total_categories']
    print(f"\nOverall: {total_success}/{total_categories} categories processed successfully")


def main():
    parser = argparse.ArgumentParser(
        description='Process FaceForensics++ and CelebDF datasets for face extraction'
    )
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Configuration file')
    parser.add_argument('--dataset', type=str, choices=['faceforensics', 'celebdf', 'both'],
                        default='both', help='Which dataset to process')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be done without executing')
    parser.add_argument('--categories', type=str, nargs='+',
                        help='Specific categories to process (optional)')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    data_dir = config['paths']['data_dir']
    
    print("="*60)
    print("DEEPFAKE DATASET PROCESSING")
    print("="*60)
    print(f"Configuration: {args.config}")
    print(f"Dataset(s): {args.dataset}")
    if args.dry_run:
        print("🔍 DRY RUN MODE - No actual processing")
    print("="*60)
    
    results = {}
    
    # Process FaceForensics++
    if args.dataset in ['faceforensics', 'both']:
        logger.info("Processing FaceForensics++ dataset...")
        
        ff_config = get_dataset_config('faceforensics', config)
        ff_input = os.path.join(data_dir, 'raw', 'faceforensics')
        ff_output = os.path.join(data_dir, 'processed', 'faceforensics')
        
        ff_categories = ['original', 'Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
        if args.categories:
            ff_categories = [cat for cat in ff_categories if cat in args.categories]
        
        results['faceforensics'] = {}
        
        for category in ff_categories:
            success = process_faceforensics_category(
                category, ff_input, ff_output, ff_config, args.dry_run
            )
            results['faceforensics'][category] = success
    
    # Process CelebDF
    if args.dataset in ['celebdf', 'both']:
        logger.info("Processing CelebDF dataset...")
        
        celebdf_config = get_dataset_config('celebdf', config)
        celebdf_input = os.path.join(data_dir, 'raw', 'celebdf')
        celebdf_output = os.path.join(data_dir, 'processed', 'celebdf')
        
        celebdf_categories = ['Real', 'Fake']
        if args.categories:
            celebdf_categories = [cat for cat in celebdf_categories if cat in args.categories]
        
        results['celebdf'] = {}
        
        for category in celebdf_categories:
            success = process_celebdf_category(
                category, celebdf_input, celebdf_output, celebdf_config, args.dry_run
            )
            results['celebdf'][category] = success
    
    # Create summary
    if not args.dry_run:
        output_dir = os.path.join(data_dir, 'processed')
        create_processing_summary(results, output_dir)
    
    # Final status
    all_success = all(
        all(categories.values()) for categories in results.values()
    )
    
    if all_success:
        print("\n🎉 All datasets processed successfully!")
        print("Next step: Create data splits and train models")
    else:
        print("\n⚠️  Some categories failed. Check logs above.")
    
    return 0 if all_success else 1


if __name__ == "__main__":
    exit(main())
