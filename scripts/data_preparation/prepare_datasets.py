#!/usr/bin/env python3
"""
Complete Dataset Preparation Script

This script coordinates the preparation of both FaceForensics++ and CelebDF datasets
for the deepfake detection research project.

Usage:
    python scripts/data_preparation/prepare_datasets.py --config config.yaml
    python scripts/data_preparation/prepare_datasets.py --celebdf-path /path/to/celebdf
"""

import os
import sys
import argparse
import yaml
import subprocess
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare datasets for deepfake detection research.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--celebdf-path', type=str, required=True,
                        help='Path to your existing CelebDF dataset')
    parser.add_argument('--skip-faceforensics', action='store_true',
                        help='Skip FaceForensics++ download (if already downloaded)')
    parser.add_argument('--skip-celebdf', action='store_true',
                        help='Skip CelebDF setup (if already organized)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be done without executing')
    
    return parser.parse_args()


def load_config(config_path):
    """Load configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_command(command, description, dry_run=False):
    """Run a command with logging."""
    logger.info(f"{'[DRY RUN] ' if dry_run else ''}{description}")
    logger.info(f"Command: {' '.join(command)}")
    
    if dry_run:
        return True
    
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        logger.info(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} failed:")
        logger.error(f"Return code: {e.returncode}")
        logger.error(f"Error output: {e.stderr}")
        return False


def check_dataset_status(data_dir):
    """Check the current status of datasets."""
    data_dir = Path(data_dir)
    
    status = {
        'faceforensics': {
            'path': data_dir / 'raw' / 'faceforensics',
            'exists': False,
            'categories': {},
            'total_videos': 0
        },
        'celebdf': {
            'path': data_dir / 'raw' / 'celebdf',
            'exists': False,
            'categories': {},
            'total_videos': 0
        }
    }
    
    # Check FaceForensics++
    ff_path = status['faceforensics']['path']
    if ff_path.exists():
        status['faceforensics']['exists'] = True
        categories = ['original', 'Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
        for category in categories:
            category_path = ff_path / category
            if category_path.exists():
                video_count = len(list(category_path.glob('*.mp4')))
                status['faceforensics']['categories'][category] = video_count
                status['faceforensics']['total_videos'] += video_count
    
    # Check CelebDF
    celebdf_path = status['celebdf']['path']
    if celebdf_path.exists():
        status['celebdf']['exists'] = True
        categories = ['Real', 'Fake']
        for category in categories:
            category_path = celebdf_path / category
            if category_path.exists():
                video_count = len(list(category_path.glob('*.mp4')))
                status['celebdf']['categories'][category] = video_count
                status['celebdf']['total_videos'] += video_count
    
    return status


def print_dataset_status(status):
    """Print current dataset status."""
    print("\n" + "="*60)
    print("CURRENT DATASET STATUS")
    print("="*60)
    
    # FaceForensics++
    print("FaceForensics++:")
    if status['faceforensics']['exists']:
        print(f"  ✅ Found at: {status['faceforensics']['path']}")
        for category, count in status['faceforensics']['categories'].items():
            print(f"    {category:15}: {count:3d} videos")
        print(f"    {'Total':15}: {status['faceforensics']['total_videos']:3d} videos")
    else:
        print(f"  ❌ Not found at: {status['faceforensics']['path']}")
    
    print()
    
    # CelebDF
    print("CelebDF:")
    if status['celebdf']['exists']:
        print(f"  ✅ Found at: {status['celebdf']['path']}")
        for category, count in status['celebdf']['categories'].items():
            print(f"    {category:15}: {count:3d} videos")
        print(f"    {'Total':15}: {status['celebdf']['total_videos']:3d} videos")
    else:
        print(f"  ❌ Not found at: {status['celebdf']['path']}")


def prepare_faceforensics(config, dry_run=False):
    """Download and prepare FaceForensics++ dataset."""
    logger.info("Preparing FaceForensics++ dataset...")
    
    data_dir = config['paths']['data_dir']
    output_path = os.path.join(data_dir, 'raw', 'faceforensics')
    
    # Download FaceForensics++
    download_script = project_root / 'scripts' / 'data_preparation' / 'download_faceforensics.py'
    command = [sys.executable, str(download_script), output_path]
    
    return run_command(command, "FaceForensics++ download", dry_run)


def prepare_celebdf(celebdf_path, config, dry_run=False):
    """Setup CelebDF dataset."""
    logger.info("Preparing CelebDF dataset...")
    
    data_dir = config['paths']['data_dir']
    output_path = os.path.join(data_dir, 'raw', 'celebdf')
    
    # Setup CelebDF
    setup_script = project_root / 'scripts' / 'data_preparation' / 'setup_celebdf.py'
    command = [sys.executable, str(setup_script), celebdf_path, output_path]
    
    return run_command(command, "CelebDF setup", dry_run)


def main():
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    data_dir = config['paths']['data_dir']
    
    print("="*60)
    print("DEEPFAKE DETECTION DATASET PREPARATION")
    print("="*60)
    print(f"Configuration: {args.config}")
    print(f"Data directory: {data_dir}")
    print(f"CelebDF source: {args.celebdf_path}")
    if args.dry_run:
        print("🔍 DRY RUN MODE - No actual changes will be made")
    print("="*60)
    
    # Check current status
    status = check_dataset_status(data_dir)
    print_dataset_status(status)
    
    # Prepare datasets
    success = True
    
    # 1. FaceForensics++
    if not args.skip_faceforensics:
        if status['faceforensics']['exists'] and status['faceforensics']['total_videos'] > 0:
            logger.info("FaceForensics++ already exists. Use --skip-faceforensics to skip.")
            response = input("Download FaceForensics++ anyway? (y/N): ")
            if response.lower() not in ['y', 'yes']:
                logger.info("Skipping FaceForensics++ download.")
            else:
                success &= prepare_faceforensics(config, args.dry_run)
        else:
            success &= prepare_faceforensics(config, args.dry_run)
    else:
        logger.info("Skipping FaceForensics++ download (--skip-faceforensics)")
    
    # 2. CelebDF
    if not args.skip_celebdf:
        if not os.path.exists(args.celebdf_path):
            logger.error(f"CelebDF source path does not exist: {args.celebdf_path}")
            success = False
        elif status['celebdf']['exists'] and status['celebdf']['total_videos'] > 0:
            logger.info("CelebDF already organized. Use --skip-celebdf to skip.")
            response = input("Setup CelebDF anyway? (y/N): ")
            if response.lower() not in ['y', 'yes']:
                logger.info("Skipping CelebDF setup.")
            else:
                success &= prepare_celebdf(args.celebdf_path, config, args.dry_run)
        else:
            success &= prepare_celebdf(args.celebdf_path, config, args.dry_run)
    else:
        logger.info("Skipping CelebDF setup (--skip-celebdf)")
    
    # Final status check
    if not args.dry_run:
        print("\n" + "="*60)
        print("FINAL DATASET STATUS")
        print("="*60)
        final_status = check_dataset_status(data_dir)
        print_dataset_status(final_status)
    
    # Summary
    print("\n" + "="*60)
    if success:
        print("🎉 DATASET PREPARATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("Next steps:")
        print("1. Extract frames: python scripts/data_preparation/extract_frames.py")
        print("2. Detect faces: python scripts/data_preparation/face_detection.py")
        print("3. Create splits: python scripts/data_preparation/create_splits.py")
    else:
        print("❌ DATASET PREPARATION FAILED!")
        print("="*60)
        print("Please check the error messages above and try again.")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
