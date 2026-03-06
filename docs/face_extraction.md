# Face Extraction Pipeline

Technical reference for the video-to-face-crop extraction pipeline.

---

## Overview

Both FaceForensics++ and CelebDF consist of short video clips (10--15 seconds), not individual face images. The extraction pipeline converts raw video into quality-filtered face crops suitable for Vision Transformer training at 224x224 resolution.

---

## Detection Backends

The pipeline supports three face detection backends. Select via the `method` field in `config.yaml`.

| Backend | Speed | Accuracy | Dependencies |
|---------|-------|----------|--------------|
| OpenCV Haar Cascades | Fast | Moderate | Included with `opencv-python` |
| MTCNN | Moderate | High | Requires `facenet-pytorch` |
| MediaPipe | Fast | High | Requires `mediapipe` |

**Recommendation:** Use OpenCV for FaceForensics++ (compressed, needs speed). Use MTCNN for CelebDF (high quality, needs accuracy).

---

## Frame Sampling Strategy

For 10--15 second clips at 24--30 FPS (240--450 total frames), the pipeline samples approximately one frame per second:

```
max_frames_per_video: 12    --> extracts ~12 uniformly spaced frames
faces_per_video: 5          --> retains the 5 highest-quality face crops
```

This provides temporal diversity (varied poses and expressions) without redundant near-identical frames.

---

## Quality Assessment

Each extracted face is assigned a quality score in the range [0, 1] based on three weighted components:

| Component | Weight | Measurement |
|-----------|--------|-------------|
| Sharpness | 0.5 | Laplacian variance |
| Brightness | 0.3 | Mean pixel intensity, optimal near 0.5 |
| Contrast | 0.2 | Standard deviation of pixel intensities |

Faces below `min_face_quality` are discarded. The top `faces_per_video` are retained per video.

---

## Configuration

```yaml
face_extraction:
  method: 'opencv'
  min_face_size: 64
  confidence_threshold: 0.7
  min_face_quality: 0.3
  max_frames_per_video: 10
  faces_per_video: 5
  margin: 0.2                # 20% padding around detected face box
```

---

## Running the Pipeline

```bash
# Extract faces from all processed videos
python scripts/data_preparation/extract_faces_from_videos.py \
    --config config.yaml
```

### Output Directory Structure

```
data/processed/
|-- faceforensics/
|   |-- original/
|   |   |-- video_001/
|   |   |   |-- frame_000_face_00_q0.85.jpg
|   |   |   |-- frame_003_face_00_q0.72.jpg
|   |   |   +-- frame_007_face_00_q0.91.jpg
|   |   +-- video_002/
|   |-- Deepfakes/
|   |-- Face2Face/
|   |-- FaceSwap/
|   +-- NeuralTextures/
+-- celebdf/
    |-- Real/
    +-- Fake/
```

---

## Supported Video Formats

MP4, AVI, MOV, GIF, WEBM, MKV. Format detection is automatic via OpenCV.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No faces detected | Lower `confidence_threshold` (e.g., 0.5) and `min_face_size` (e.g., 32) |
| Too many low-quality faces | Increase `min_face_quality` (e.g., 0.5) and reduce `faces_per_video` |
| Processing too slow | Reduce `max_frames_per_video`; switch to OpenCV backend |
| Memory issues | Process in smaller batches with `--max-videos` flag |

---

## Dataset Quality Checklist

### FaceForensics++ (c23 compression)
- [ ] 6--12 faces per video extracted
- [ ] Quality scores >= 0.25 for retained faces
- [ ] Processing time <= 3 seconds per video

### CelebDF (high quality)
- [ ] 8--15 faces per video extracted
- [ ] Quality scores >= 0.4 for retained faces
- [ ] Consistent 224x224 output dimensions
