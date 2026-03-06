#!/usr/bin/env python3
"""
FaceForensics++ Download Script for Deepfake Detection Research

Modified version of the original FaceForensics++ download script to download
exactly 100 videos from each category for our deepfake detection research.

Categories:
- Original (Real videos)
- Deepfakes 
- Face2Face
- FaceSwap
- NeuralTextures

Usage:
    python scripts/data_preparation/download_faceforensics.py data/raw/faceforensics
"""

import argparse
import os
import urllib
import urllib.request
import tempfile
import time
import sys
import json
import random
from tqdm import tqdm
from os.path import join
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# URLs and filenames
FILELIST_URL = 'misc/filelist.json'
DEEPFEAKES_DETECTION_URL = 'misc/deepfake_detection_filenames.json'

# Parameters for our research
RESEARCH_DATASETS = {
    'original': 'original_sequences/youtube',
    'Deepfakes': 'manipulated_sequences/Deepfakes',
    'Face2Face': 'manipulated_sequences/Face2Face',
    'FaceSwap': 'manipulated_sequences/FaceSwap',
    'NeuralTextures': 'manipulated_sequences/NeuralTextures'
}

COMPRESSION = 'c23'  # Good balance between quality and size
TYPE = 'videos'
VIDEOS_PER_CATEGORY = 100  # Exactly 100 videos per category
SERVERS = ['EU', 'EU2', 'CA']


def parse_args():
    parser = argparse.ArgumentParser(
        description='Downloads FaceForensics++ data for deepfake detection research.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('output_path', type=str, help='Output directory (e.g., data/raw/faceforensics)')
    parser.add_argument('--server', type=str, default='EU',
                        help='Server to download from. Change if slow download.',
                        choices=SERVERS)
    parser.add_argument('--compression', type=str, default='c23',
                        help='Compression level for videos.',
                        choices=['raw', 'c23', 'c40'])
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible video selection.')
    
    args = parser.parse_args()

    # Set server URL
    server = args.server
    if server == 'EU':
        server_url = 'http://canis.vc.in.tum.de:8100/'
    elif server == 'EU2':
        server_url = 'http://kaldir.vc.in.tum.de/faceforensics/'
    elif server == 'CA':
        server_url = 'http://falas.cmpt.sfu.ca:8100/'
    else:
        raise Exception('Wrong server name. Choices: {}'.format(str(SERVERS)))
    
    args.tos_url = server_url + 'webpage/FaceForensics_TOS.pdf'
    args.base_url = server_url + 'v3/'
    
    return args


def download_file(url, out_file, report_progress=True):
    """Download a single file with progress reporting."""
    out_dir = os.path.dirname(out_file)
    os.makedirs(out_dir, exist_ok=True)
    
    if os.path.isfile(out_file):
        logger.info(f'File already exists: {out_file}')
        return
    
    fh, out_file_tmp = tempfile.mkstemp(dir=out_dir)
    f = os.fdopen(fh, 'w')
    f.close()
    
    try:
        if report_progress:
            def reporthook(count, block_size, total_size):
                if count == 0:
                    return
                duration = time.time() - start_time
                progress_size = int(count * block_size)
                speed = int(progress_size / (1024 * duration)) if duration > 0 else 0
                percent = min(int(count * block_size * 100 / total_size), 100)
                sys.stdout.write(f"\rProgress: {percent}%, {progress_size // (1024 * 1024)} MB, {speed} KB/s")
                sys.stdout.flush()
            
            start_time = time.time()
            urllib.request.urlretrieve(url, out_file_tmp, reporthook=reporthook)
            print()  # New line after progress
        else:
            urllib.request.urlretrieve(url, out_file_tmp)
        
        os.rename(out_file_tmp, out_file)
        logger.info(f'Downloaded: {out_file}')
        
    except Exception as e:
        logger.error(f'Error downloading {url}: {e}')
        if os.path.exists(out_file_tmp):
            os.remove(out_file_tmp)
        raise


def get_video_filelist(base_url, dataset_path):
    """Get the list of available videos for a dataset."""
    if 'original' in dataset_path:
        # Load filelist from server
        file_pairs = json.loads(urllib.request.urlopen(base_url + '/' + FILELIST_URL).read().decode("utf-8"))
        filelist = []
        for pair in file_pairs:
            filelist += pair
    else:
        # Load filelist from server for manipulated videos
        file_pairs = json.loads(urllib.request.urlopen(base_url + '/' + FILELIST_URL).read().decode("utf-8"))
        filelist = []
        for pair in file_pairs:
            filelist.append('_'.join(pair))
            filelist.append('_'.join(pair[::-1]))  # Both directions for manipulated videos
    
    return filelist


def select_videos(filelist, num_videos, seed=42):
    """Randomly select a subset of videos for reproducible results."""
    random.seed(seed)
    if len(filelist) <= num_videos:
        logger.warning(f'Only {len(filelist)} videos available, using all.')
        return filelist
    
    selected = random.sample(filelist, num_videos)
    logger.info(f'Selected {len(selected)} videos from {len(filelist)} available.')
    return selected


def download_category(args, category, dataset_path):
    """Download videos for a specific category."""
    logger.info(f'Downloading {category} videos...')
    
    # Get available videos
    try:
        filelist = get_video_filelist(args.base_url, dataset_path)
    except Exception as e:
        logger.error(f'Error getting filelist for {category}: {e}')
        return False
    
    # Select subset of videos
    selected_videos = select_videos(filelist, VIDEOS_PER_CATEGORY, args.seed)
    
    # Prepare URLs and paths
    dataset_videos_url = args.base_url + f'{dataset_path}/{args.compression}/{TYPE}/'
    dataset_output_path = join(args.output_path, category)
    
    logger.info(f'Output path: {dataset_output_path}')
    logger.info(f'Downloading {len(selected_videos)} videos for {category}')
    
    # Download videos
    success_count = 0
    for i, filename in enumerate(tqdm(selected_videos, desc=f'Downloading {category}')):
        video_filename = filename + '.mp4'
        video_url = dataset_videos_url + video_filename
        output_file = join(dataset_output_path, video_filename)
        
        try:
            download_file(video_url, output_file, report_progress=False)
            success_count += 1
        except Exception as e:
            logger.error(f'Failed to download {video_filename}: {e}')
            continue
    
    logger.info(f'Successfully downloaded {success_count}/{len(selected_videos)} videos for {category}')
    return success_count > 0


def create_download_summary(output_path, download_stats):
    """Create a summary of downloaded data."""
    summary = {
        'total_categories': len(download_stats),
        'videos_per_category': VIDEOS_PER_CATEGORY,
        'compression': COMPRESSION,
        'categories': download_stats,
        'total_videos': sum(download_stats.values())
    }
    
    summary_file = join(output_path, 'download_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f'Download summary saved to: {summary_file}')
    
    # Print summary
    print("\n" + "="*60)
    print("FACEFORENSICS++ DOWNLOAD SUMMARY")
    print("="*60)
    for category, count in download_stats.items():
        print(f"{category:15}: {count:3d} videos")
    print("-"*30)
    print(f"{'Total':15}: {sum(download_stats.values()):3d} videos")
    print(f"Compression: {COMPRESSION}")
    print(f"Output directory: {output_path}")


def main():
    args = parse_args()
    
    # Terms of Service agreement
    print('='*60)
    print('FACEFORENSICS++ TERMS OF SERVICE')
    print('='*60)
    print('By continuing, you confirm that you have agreed to the FaceForensics++ terms of use:')
    print(args.tos_url)
    print('\nThis script will download 100 videos from each of the following categories:')
    for category in RESEARCH_DATASETS.keys():
        print(f'  - {category}')
    print(f'\nTotal: {len(RESEARCH_DATASETS) * VIDEOS_PER_CATEGORY} videos')
    print(f'Estimated size: ~10-15 GB')
    print('='*60)
    
    response = input('Press ENTER to continue, or CTRL-C to exit: ')
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    # Download each category
    download_stats = {}
    
    for category, dataset_path in RESEARCH_DATASETS.items():
        try:
            success = download_category(args, category, dataset_path)
            if success:
                # Count actual downloaded files
                category_path = join(args.output_path, category)
                if os.path.exists(category_path):
                    video_count = len([f for f in os.listdir(category_path) if f.endswith('.mp4')])
                    download_stats[category] = video_count
                else:
                    download_stats[category] = 0
            else:
                download_stats[category] = 0
        except KeyboardInterrupt:
            logger.info('Download interrupted by user.')
            break
        except Exception as e:
            logger.error(f'Error downloading {category}: {e}')
            download_stats[category] = 0
    
    # Create summary
    create_download_summary(args.output_path, download_stats)
    
    print('\n🎉 FaceForensics++ download completed!')
    print(f'📁 Data saved to: {args.output_path}')
    print('📋 Next step: Run frame extraction script')


if __name__ == "__main__":
    main()
