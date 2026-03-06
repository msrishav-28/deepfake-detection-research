# Data Directory

This directory contains the datasets used for deepfake detection research.

## Structure

```
data/
├── raw/                    # Raw downloaded datasets
│   ├── faceforensics/     # FaceForensics++ dataset
│   └── celebdf/           # CelebDF dataset
├── processed/             # Processed and split datasets
│   ├── train/             # Training set for base models
│   ├── holdout/           # Hold-out set for meta-learner
│   └── test/              # Test set for final evaluation
└── splits/                # Data split information
    ├── train_split.txt
    ├── holdout_split.txt
    └── test_split.txt
```

## Dataset Information

### FaceForensics++
- **Categories**: Deepfakes, Face2Face, FaceSwap, NeuralTextures, Original
- **Usage**: 100 videos from each category (500 total)
- **Format**: Video files with extracted frames

### CelebDF
- **Categories**: Real, Fake
- **Usage**: Additional dataset for robustness testing
- **Format**: Video files with extracted frames

## Data Preparation

1. Download datasets using provided scripts
2. Extract frames from videos at 1 FPS
3. Apply face detection and cropping
4. Split into train/holdout/test sets (60%/20%/20%)
5. Apply data augmentation during training
