#!/usr/bin/env python3
"""
CelebDF Dataset Setup Script

This script organizes your existing CelebDF dataset for the deepfake detection research.
It creates the proper directory structure and prepares the data for integration
with FaceForensics++.

Usage:
    python scripts/data_preparation/setup_celebdf.py /path/to/your/celebdf data/raw/celebdf
"""

import os
import shutil
import argparse
import json
import logging
from pathlib import Path
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Setup CelebDF dataset for deepfake detection research.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('source_path', type=str, 
                        help='Path to your existing CelebDF dataset')
    parser.add_argument('output_path', type=str, 
                        help='Output directory (e.g., data/raw/celebdf)')
    parser.add_argument('--copy', action='store_true',
                        help='Copy files instead of creating symbolic links')
    parser.add_argument('--max-videos', type=int, default=None,
                        help='Maximum number of videos per category (for testing)')
    
    return parser.parse_args()


def find_celebdf_structure(source_path):
    """
    Analyze the CelebDF dataset structure to understand the organization.
    CelebDF typically has different possible structures.
    """
    source_path = Path(source_path)
    
    if not source_path.exists():
        raise ValueError(f"Source path does not exist: {source_path}")
    
    logger.info(f"Analyzing CelebDF structure in: {source_path}")
    
    # Common CelebDF directory patterns
    possible_structures = [
        # Structure 1: Celeb-real / Celeb-synthesis
        {
            'real': source_path / 'Celeb-real',
            'fake': source_path / 'Celeb-synthesis'
        },
        # Structure 2: YouTube-real / Celeb-DF
        {
            'real': source_path / 'YouTube-real',
            'fake': source_path / 'Celeb-DF'
        },
        # Structure 3: real / fake
        {
            'real': source_path / 'real',
            'fake': source_path / 'fake'
        },
        # Structure 4: videos directly in subdirectories
        {
            'real': source_path / 'real',
            'fake': source_path / 'fake'
        }
    ]
    
    # Check which structure exists
    for i, structure in enumerate(possible_structures):
        if structure['real'].exists() and structure['fake'].exists():
            real_videos = list(structure['real'].glob('**/*.mp4'))
            fake_videos = list(structure['fake'].glob('**/*.mp4'))
            
            if real_videos and fake_videos:
                logger.info(f"Found CelebDF structure {i+1}:")
                logger.info(f"  Real videos: {len(real_videos)} in {structure['real']}")
                logger.info(f"  Fake videos: {len(fake_videos)} in {structure['fake']}")
                return structure, real_videos, fake_videos
    
    # If no standard structure found, try to find all videos
    all_videos = list(source_path.glob('**/*.mp4'))
    if all_videos:
        logger.warning("No standard CelebDF structure found. Found all videos:")
        logger.warning(f"  Total videos: {len(all_videos)}")
        logger.warning("Please manually organize into real/fake categories.")
        return None, [], all_videos
    
    raise ValueError("No video files found in the source directory")


def organize_videos(videos, output_dir, category, max_videos=None, copy_files=False):
    """Organize videos into the target directory structure."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if max_videos:
        videos = videos[:max_videos]
    
    logger.info(f"Organizing {len(videos)} {category} videos...")
    
    organized_count = 0
    for video_path in tqdm(videos, desc=f"Organizing {category} videos"):
        try:
            # Create output filename
            output_file = output_dir / video_path.name
            
            # Skip if already exists
            if output_file.exists():
                logger.debug(f"Skipping existing file: {output_file}")
                organized_count += 1
                continue
            
            # Copy or link the file
            if copy_files:
                shutil.copy2(video_path, output_file)
                logger.debug(f"Copied: {video_path} -> {output_file}")
            else:
                # Create symbolic link (Windows may require admin privileges)
                try:
                    output_file.symlink_to(video_path.absolute())
                    logger.debug(f"Linked: {video_path} -> {output_file}")
                except OSError:
                    # Fallback to copying if symlink fails
                    shutil.copy2(video_path, output_file)
                    logger.debug(f"Copied (symlink failed): {video_path} -> {output_file}")
            
            organized_count += 1
            
        except Exception as e:
            logger.error(f"Error organizing {video_path}: {e}")
            continue
    
    logger.info(f"Successfully organized {organized_count} {category} videos")
    return organized_count


def create_celebdf_summary(output_path, real_count, fake_count):
    """Create a summary of the CelebDF dataset organization."""
    summary = {
        'dataset': 'CelebDF',
        'categories': {
            'Real': real_count,
            'Fake': fake_count
        },
        'total_videos': real_count + fake_count,
        'structure': {
            'Real': 'Real/',
            'Fake': 'Fake/'
        }
    }
    
    summary_file = Path(output_path) / 'celebdf_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"CelebDF summary saved to: {summary_file}")
    
    # Print summary
    print("\n" + "="*50)
    print("CELEBDF DATASET SUMMARY")
    print("="*50)
    print(f"Real videos:  {real_count:4d}")
    print(f"Fake videos:  {fake_count:4d}")
    print("-"*25)
    print(f"Total:        {real_count + fake_count:4d}")
    print(f"Output directory: {output_path}")


def main():
    args = parse_args()
    
    print("="*60)
    print("CELEBDF DATASET SETUP")
    print("="*60)
    print(f"Source: {args.source_path}")
    print(f"Output: {args.output_path}")
    print(f"Method: {'Copy' if args.copy else 'Symbolic link'}")
    if args.max_videos:
        print(f"Max videos per category: {args.max_videos}")
    print("="*60)
    
    try:
        # Analyze source structure
        structure, real_videos, fake_videos = find_celebdf_structure(args.source_path)
        
        if structure is None:
            logger.error("Could not determine CelebDF structure. Please organize manually.")
            return
        
        # Create output directories
        output_path = Path(args.output_path)
        real_output = output_path / 'Real'
        fake_output = output_path / 'Fake'
        
        # Organize videos
        real_count = organize_videos(
            real_videos, real_output, 'Real', 
            args.max_videos, args.copy
        )
        
        fake_count = organize_videos(
            fake_videos, fake_output, 'Fake', 
            args.max_videos, args.copy
        )
        
        # Create summary
        create_celebdf_summary(args.output_path, real_count, fake_count)
        
        print('\n🎉 CelebDF dataset setup completed!')
        print(f'📁 Data organized in: {args.output_path}')
        print('📋 Next step: Run frame extraction script')
        
    except Exception as e:
        logger.error(f"Error setting up CelebDF dataset: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
