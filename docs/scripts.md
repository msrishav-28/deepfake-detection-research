# Scripts Directory

This directory contains utility scripts for data preparation, training, and evaluation.

## Structure

```
scripts/
├── data_preparation/      # Data preprocessing scripts
│   ├── download_datasets.py
│   ├── extract_frames.py
│   ├── face_detection.py
│   └── create_splits.py
├── training/             # Training scripts
│   ├── train_base_models.py
│   ├── train_ensemble.py
│   └── hyperparameter_search.py
├── evaluation/           # Evaluation scripts
│   ├── evaluate_models.py
│   ├── generate_gradcam.py
│   └── benchmark_performance.py
└── utils/               # Utility scripts
    ├── model_converter.py
    ├── data_validator.py
    └── results_analyzer.py
```

## Script Descriptions

### Data Preparation
- **download_datasets.py**: Download FaceForensics++ and CelebDF datasets
- **extract_frames.py**: Extract frames from video files
- **face_detection.py**: Detect and crop faces from frames
- **create_splits.py**: Create train/holdout/test splits

### Training
- **train_base_models.py**: Train individual ViT, DeiT, and Swin models
- **train_ensemble.py**: Train the meta-learner for ensemble
- **hyperparameter_search.py**: Optimize hyperparameters

### Evaluation
- **evaluate_models.py**: Comprehensive model evaluation
- **generate_gradcam.py**: Generate Grad-CAM visualizations
- **benchmark_performance.py**: Performance benchmarking

### Utilities
- **model_converter.py**: Convert between model formats
- **data_validator.py**: Validate dataset integrity
- **results_analyzer.py**: Analyze and summarize results
