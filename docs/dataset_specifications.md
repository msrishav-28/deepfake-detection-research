# Dataset Specifications

Processing parameters and resource estimates for FaceForensics++ and CelebDF datasets.

---

## FaceForensics++

### Source Characteristics

| Property | Value |
|----------|-------|
| Format | MP4 (H.264) |
| Duration | 10--15 seconds per clip |
| c23 file size | ~2 MB per video |
| c40 file size | ~0.4 MB per video |
| Total videos | 5,000 (1,000 real + 4,000 fake) |
| Categories | Original, Deepfakes, Face2Face, FaceSwap, NeuralTextures |

### Recommended Processing Settings (c23 compression)

```yaml
faceforensics:
  method: 'opencv'
  min_face_quality: 0.25
  faces_per_video: 6
  max_frames_per_video: 12        # ~1 frame/second for 10-15s clips
  confidence_threshold: 0.6
```

### Resource Estimates

| Metric | Value |
|--------|-------|
| Processing method | OpenCV |
| Time per video | 2--3 seconds |
| Total processing time | 20--25 minutes (500 videos) |
| Expected face crops | ~3,000 |
| Storage requirement | ~500 MB |

---

## CelebDF

### Source Characteristics

| Property | Value |
|----------|-------|
| Format | MP4 (high quality) |
| Duration | ~13 seconds per clip |
| File size | Several MB per video |
| Quality | High resolution |

### Recommended Processing Settings

```yaml
celebdf:
  method: 'mtcnn'
  min_face_quality: 0.4
  faces_per_video: 10
  max_frames_per_video: 15
  confidence_threshold: 0.7
```

### Resource Estimates

| Metric | Value |
|--------|-------|
| Processing method | MTCNN |
| Time per video | 8--12 seconds |
| Expected faces per video | 8--10 |
| Storage requirement | 1--2 GB |

---

## Processing Commands

```bash
# Download FaceForensics++ (c23 compression recommended)
python scripts/data_preparation/download_faceforensics.py \
    data/raw/faceforensics --compression c23

# Extract faces from both datasets
python scripts/data_preparation/extract_faces_from_videos.py \
    --config config.yaml
```

---

## Expected Output

### FaceForensics++ (100 videos per category)

```
Input:  500 MP4 videos (10-15 seconds each)
Output: 3,000-6,000 face crops

Breakdown:
  Original:       100 videos --> 600-1,200 face crops
  Deepfakes:      100 videos --> 600-1,200 face crops
  Face2Face:      100 videos --> 600-1,200 face crops
  FaceSwap:       100 videos --> 600-1,200 face crops
  NeuralTextures: 100 videos --> 600-1,200 face crops

Quality distribution (c23 compression):
  High quality   (0.50-1.00): ~40%
  Medium quality (0.25-0.50): ~45%
  Low quality    (<0.25):     ~15% (discarded)
```

### CelebDF

```
Input:  Variable number of MP4 videos (~13 seconds each)
Output: 8-10 face crops per video

Quality distribution (high-quality source):
  High quality   (0.70-1.00): ~60%
  Medium quality (0.40-0.70): ~35%
  Low quality    (<0.40):     ~5% (discarded)
```

---

## Compression-Aware Thresholds

| Compression Level | `min_face_quality` | `confidence_threshold` |
|-------------------|--------------------|------------------------|
| Raw / high quality | 0.40 | 0.70 |
| c23 (medium) | 0.25 | 0.60 |
| c40 (low) | 0.15 | 0.50 |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Compression artifacts reduce face detection | Lower `min_face_quality` and `confidence_threshold` per table above |
| Variable video durations | Framework automatically caps at `max_frames_per_video` |
| Memory usage with large videos | Reduce batch size or process a subset with `--max-videos` |
