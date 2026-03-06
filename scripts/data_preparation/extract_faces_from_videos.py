#!/usr/bin/env python3
"""
Advanced Face Extraction from Videos/GIFs for Deepfake Detection

This script handles face extraction from short videos and GIFs with:
- Multiple face detection backends (MTCNN, RetinaFace, MediaPipe)
- Quality assessment and filtering
- Temporal consistency for video sequences
- Robust handling of GIFs and short clips

Usage:
    python scripts/data_preparation/extract_faces_from_videos.py --config config.yaml
"""

import os
import sys
import cv2
import numpy as np
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm
import logging
from typing import List, Tuple, Optional, Dict, Any
import json
import torch

# Face detection imports
try:
    from facenet_pytorch import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FaceExtractor:
    """Advanced face extraction with multiple backends and quality assessment."""
    
    def __init__(
        self,
        method: str = 'opencv',
        min_face_size: int = 64,
        confidence_threshold: float = 0.7,
        device: str = 'auto'
    ):
        """
        Args:
            method: Face detection method ('opencv', 'mtcnn', 'mediapipe')
            min_face_size: Minimum face size in pixels
            confidence_threshold: Minimum confidence for face detection
            device: Device for computation ('cpu', 'cuda', 'auto')
        """
        self.method = method
        self.min_face_size = min_face_size
        self.confidence_threshold = confidence_threshold
        
        # Set device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Initialize face detector
        self.detector = self._init_detector()
        
        logger.info(f"Initialized {method} face detector on {self.device}")
    
    def _init_detector(self):
        """Initialize the face detection backend."""
        if self.method == 'opencv':
            # OpenCV Haar Cascade (most compatible)
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            detector = cv2.CascadeClassifier(cascade_path)
            if detector.empty():
                raise ValueError("Could not load OpenCV face cascade")
            return detector
            
        elif self.method == 'mtcnn' and MTCNN_AVAILABLE:
            # MTCNN (more accurate)
            device = torch.device(self.device)
            return MTCNN(
                image_size=224,
                margin=20,
                min_face_size=self.min_face_size,
                thresholds=[0.6, 0.7, 0.7],
                factor=0.709,
                post_process=False,
                device=device
            )
            
        elif self.method == 'mediapipe' and MEDIAPIPE_AVAILABLE:
            # MediaPipe (fast and robust)
            mp_face_detection = mp.solutions.face_detection
            return mp_face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=self.confidence_threshold
            )
            
        else:
            logger.warning(f"Method {self.method} not available, falling back to OpenCV")
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            return cv2.CascadeClassifier(cascade_path)
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect faces in an image.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of (x, y, w, h, confidence) tuples
        """
        if self.method == 'opencv':
            return self._detect_opencv(image)
        elif self.method == 'mtcnn' and MTCNN_AVAILABLE:
            return self._detect_mtcnn(image)
        elif self.method == 'mediapipe' and MEDIAPIPE_AVAILABLE:
            return self._detect_mediapipe(image)
        else:
            return self._detect_opencv(image)
    
    def _detect_opencv(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """OpenCV face detection."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(self.min_face_size, self.min_face_size)
        )
        
        # Convert to (x, y, w, h, confidence) format
        return [(x, y, w, h, 1.0) for x, y, w, h in faces]
    
    def _detect_mtcnn(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """MTCNN face detection."""
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        boxes, probs = self.detector.detect(rgb_image)
        
        faces = []
        if boxes is not None:
            for box, prob in zip(boxes, probs):
                if prob >= self.confidence_threshold:
                    x, y, x2, y2 = box.astype(int)
                    w, h = x2 - x, y2 - y
                    if w >= self.min_face_size and h >= self.min_face_size:
                        faces.append((x, y, w, h, prob))
        
        return faces
    
    def _detect_mediapipe(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """MediaPipe face detection."""
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = self.detector.process(rgb_image)
        
        faces = []
        if results.detections:
            h, w, _ = image.shape
            for detection in results.detections:
                confidence = detection.score[0]
                if confidence >= self.confidence_threshold:
                    bbox = detection.location_data.relative_bounding_box
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    face_w = int(bbox.width * w)
                    face_h = int(bbox.height * h)
                    
                    if face_w >= self.min_face_size and face_h >= self.min_face_size:
                        faces.append((x, y, face_w, face_h, confidence))
        
        return faces
    
    def extract_face_crop(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        target_size: int = 224,
        margin: float = 0.2
    ) -> Optional[np.ndarray]:
        """
        Extract and crop face from image.
        
        Args:
            image: Input image
            bbox: Bounding box (x, y, w, h)
            target_size: Target size for output
            margin: Margin around face (as fraction of face size)
            
        Returns:
            Cropped face image or None if extraction fails
        """
        x, y, w, h = bbox
        
        # Add margin
        margin_x = int(w * margin)
        margin_y = int(h * margin)
        
        # Calculate crop coordinates
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(image.shape[1], x + w + margin_x)
        y2 = min(image.shape[0], y + h + margin_y)
        
        # Extract face
        face_crop = image[y1:y2, x1:x2]
        
        if face_crop.size == 0:
            return None
        
        # Resize to target size
        face_resized = cv2.resize(face_crop, (target_size, target_size))
        
        return face_resized
    
    def assess_face_quality(self, face_crop: np.ndarray) -> float:
        """
        Assess the quality of a face crop.
        
        Args:
            face_crop: Cropped face image
            
        Returns:
            Quality score (0-1, higher is better)
        """
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        
        # Calculate sharpness (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(laplacian_var / 1000.0, 1.0)  # Normalize
        
        # Calculate brightness (avoid too dark/bright)
        brightness = np.mean(gray) / 255.0
        brightness_score = 1.0 - abs(brightness - 0.5) * 2  # Optimal around 0.5
        
        # Calculate contrast
        contrast = np.std(gray) / 255.0
        contrast_score = min(contrast * 4, 1.0)  # Normalize
        
        # Combined quality score
        quality = (sharpness_score * 0.5 + brightness_score * 0.3 + contrast_score * 0.2)
        
        return quality


def extract_frames_from_video(
    video_path: str,
    max_frames: int = 15,  # Optimized for 10-15 second clips
    frame_interval: Optional[int] = None,
    target_fps: Optional[float] = 1.0  # Extract 1 frame per second
) -> List[np.ndarray]:
    """
    Extract frames from video optimized for FaceForensics++/CelebDF.

    Args:
        video_path: Path to video file (MP4 format)
        max_frames: Maximum number of frames to extract (15 for 10-15s clips)
        frame_interval: Interval between frames (None for uniform sampling)
        target_fps: Target FPS for extraction (1.0 = 1 frame per second)

    Returns:
        List of frame images
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return []
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    logger.debug(f"Video: {total_frames} frames, {fps:.2f} FPS")
    
    frames = []

    if frame_interval is None:
        # Optimized sampling for 10-15 second clips
        if target_fps and fps > 0:
            # Extract frames at target FPS (e.g., 1 FPS for 10-15 frames from 10-15s video)
            frame_step = max(1, int(fps / target_fps))
            frame_indices = list(range(0, min(total_frames, max_frames * frame_step), frame_step))
        elif total_frames <= max_frames:
            # Use all frames if video is very short
            frame_indices = list(range(total_frames))
        else:
            # Uniform sampling across the video
            frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
    else:
        # Fixed interval sampling
        frame_indices = list(range(0, total_frames, frame_interval))[:max_frames]
    
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            frames.append(frame)
        else:
            logger.warning(f"Could not read frame {frame_idx} from {video_path}")
    
    cap.release()
    
    logger.debug(f"Extracted {len(frames)} frames from {video_path}")
    return frames


def process_video_file(
    video_path: str,
    output_dir: str,
    face_extractor: FaceExtractor,
    config: Dict[str, Any]
) -> int:
    """
    Process a single video file and extract faces.
    
    Args:
        video_path: Path to video file
        output_dir: Output directory for face crops
        face_extractor: Face extraction instance
        config: Configuration dictionary
        
    Returns:
        Number of faces extracted
    """
    video_name = Path(video_path).stem
    video_output_dir = Path(output_dir) / video_name
    video_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract frames
    frames = extract_frames_from_video(
        video_path,
        max_frames=config.get('max_frames_per_video', 10),
        frame_interval=config.get('frame_interval', None)
    )
    
    if not frames:
        logger.warning(f"No frames extracted from {video_path}")
        return 0
    
    faces_extracted = 0
    min_quality = config.get('min_face_quality', 0.3)
    
    for frame_idx, frame in enumerate(frames):
        # Detect faces
        face_detections = face_extractor.detect_faces(frame)
        
        for face_idx, (x, y, w, h, confidence) in enumerate(face_detections):
            # Extract face crop
            face_crop = face_extractor.extract_face_crop(
                frame, (x, y, w, h),
                target_size=config.get('face_size', 224)
            )
            
            if face_crop is None:
                continue
            
            # Assess quality
            quality = face_extractor.assess_face_quality(face_crop)
            
            if quality >= min_quality:
                # Save face crop
                face_filename = f"frame_{frame_idx:03d}_face_{face_idx:02d}_q{quality:.2f}.jpg"
                face_path = video_output_dir / face_filename
                
                cv2.imwrite(str(face_path), face_crop)
                faces_extracted += 1
                
                logger.debug(f"Saved face: {face_path} (quality: {quality:.2f})")
    
    return faces_extracted


def main():
    parser = argparse.ArgumentParser(description='Extract faces from videos/GIFs')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Configuration file')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Input directory containing videos')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for face crops')
    parser.add_argument('--method', type=str, default='opencv',
                        choices=['opencv', 'mtcnn', 'mediapipe'],
                        help='Face detection method')
    parser.add_argument('--max-videos', type=int, default=None,
                        help='Maximum number of videos to process (for testing)')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize face extractor
    face_extractor = FaceExtractor(
        method=args.method,
        min_face_size=config.get('face_extraction', {}).get('min_face_size', 64),
        confidence_threshold=config.get('face_extraction', {}).get('confidence_threshold', 0.7)
    )
    
    # Find video files
    input_dir = Path(args.input_dir)
    video_extensions = ['.mp4', '.avi', '.mov', '.gif', '.webm', '.mkv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(input_dir.glob(f'**/*{ext}'))
        video_files.extend(input_dir.glob(f'**/*{ext.upper()}'))
    
    if args.max_videos:
        video_files = video_files[:args.max_videos]
    
    logger.info(f"Found {len(video_files)} video files to process")
    
    # Process videos
    total_faces = 0
    processed_videos = 0
    
    for video_path in tqdm(video_files, desc="Processing videos"):
        try:
            faces_count = process_video_file(
                str(video_path), args.output_dir, face_extractor, config
            )
            total_faces += faces_count
            processed_videos += 1
            
            if faces_count == 0:
                logger.warning(f"No faces extracted from {video_path}")
                
        except Exception as e:
            logger.error(f"Error processing {video_path}: {e}")
            continue
    
    # Summary
    logger.info(f"Processing completed:")
    logger.info(f"  Videos processed: {processed_videos}/{len(video_files)}")
    logger.info(f"  Total faces extracted: {total_faces}")
    logger.info(f"  Average faces per video: {total_faces/processed_videos:.1f}")
    
    # Save processing summary
    summary = {
        'total_videos': len(video_files),
        'processed_videos': processed_videos,
        'total_faces': total_faces,
        'average_faces_per_video': total_faces / processed_videos if processed_videos > 0 else 0,
        'method': args.method,
        'config': config.get('face_extraction', {})
    }
    
    summary_path = Path(args.output_dir) / 'extraction_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
