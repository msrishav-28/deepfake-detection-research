# Scripts Reference

Reference for all executable scripts in the `scripts/` directory.

## Directory Structure

```
scripts/
|-- check_environment.py                   # Environment verification
|-- data_preparation/
|   |-- download_faceforensics.py          # FaceForensics++ downloader
|   |-- setup_celebdf.py                   # CelebDF directory organizer
|   |-- prepare_datasets.py               # Combined dataset preparation
|   |-- extract_faces_from_videos.py       # Video-to-face-crop extraction
|   +-- create_splits.py                   # Train/val/holdout/test splitting
|-- training/
|   |-- train_base_models.py               # ViT, DeiT, Swin fine-tuning
|   +-- train_ensemble.py                  # Meta-learner training
+-- evaluation/
    |-- comprehensive_evaluation.py        # Full evaluation with Grad-CAM
    |-- benchmark_deepfake_models.py       # Benchmarking and CSV export
    +-- inference_pipeline.py              # Single-image and batch inference
```

## Script Descriptions

### Environment

| Script | Description |
|--------|-------------|
| `check_environment.py` | Validates Python version, GPU availability, installed packages, and directory structure. Run this first after installation. |

### Data Preparation

| Script | Description |
|--------|-------------|
| `download_faceforensics.py` | Downloads FaceForensics++ videos (100 per category, c23 compression). Requires access credentials from the dataset authors. |
| `setup_celebdf.py` | Organizes an existing CelebDF download into `Real/` and `Fake/` subdirectories under `data/raw/celebdf/`. |
| `prepare_datasets.py` | Combined entry point: downloads FaceForensics++ and organizes CelebDF in a single command. Accepts `--celebdf-path` to point to your local CelebDF copy. |
| `extract_faces_from_videos.py` | Extracts face crops from video files using OpenCV, MTCNN, or MediaPipe backends. Performs quality assessment and retains the top-N faces per video. |
| `create_splits.py` | Generates stratified 4-way splits (train 50% / val 10% / holdout 20% / test 20%) and writes split files to `data/splits/`. |

### Training

| Script | Description |
|--------|-------------|
| `train_base_models.py` | Fine-tunes ViT-Base, DeiT-Base, and Swin-Base using LLRD, warmup + cosine scheduler, label smoothing, gradient clipping, and MixUp augmentation. Validates against the `val` split; the `holdout` split is reserved for the meta-learner. |
| `train_ensemble.py` | Loads trained base models, generates meta-features on the holdout set, trains a logistic regression meta-learner with cross-validation, and saves the stacked ensemble. |

### Evaluation

| Script | Description |
|--------|-------------|
| `comprehensive_evaluation.py` | Runs full evaluation: per-model metrics, ensemble comparison, McNemar's significance test, and optional Grad-CAM explainability visualizations. |
| `benchmark_deepfake_models.py` | Exports benchmark results in timm-style CSV format with accuracy, AUC, F1, inference time, and throughput. |
| `inference_pipeline.py` | Runs inference on individual images or directories. Outputs predictions as JSON with optional batch analysis. |
