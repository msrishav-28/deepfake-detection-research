# Quick Start Guide

Step-by-step instructions to go from a fresh clone to trained models and evaluation results.

---

## Prerequisites

| Requirement | Minimum |
|-------------|---------|
| Python | 3.8+ |
| PyTorch | >= 1.13 |
| GPU VRAM | 8 GB (recommended) |
| RAM | 16 GB |
| Disk Space | 50 GB |

## Installation

```bash
git clone https://github.com/msrishav-28/Deepfake-Detection-Research.git
cd Deepfake-Detection-Research
pip install -r requirements.txt
python scripts/check_environment.py
```

---

## Pipeline Execution

### Step 1: Prepare Datasets

```bash
python scripts/data_preparation/prepare_datasets.py \
    --config config.yaml \
    --celebdf-path "C:/path/to/your/celebdf/dataset"
```

This command downloads FaceForensics++ (100 videos per category, c23 compression) and organizes your existing CelebDF into `Real/Fake` categories.

**Expected time:** 1--2 hours depending on network speed.

### Step 2: Extract Faces from Videos

```bash
python scripts/data_preparation/extract_faces_from_videos.py \
    --config config.yaml
```

Extracts face crops from both datasets using OpenCV (FaceForensics++) or MTCNN (CelebDF). Performs quality filtering and retains the top faces per video.

**Expected time:** 30--45 minutes. **Output:** 3,000--6,000 face crops.

### Step 3: Create Data Splits

```bash
python scripts/data_preparation/create_splits.py --config config.yaml
```

Generates stratified splits: train (50%), val (10%), holdout (20%), test (20%). The val split is used for base model early stopping; the holdout split is reserved exclusively for meta-learner training.

**Expected time:** Under 1 minute.

### Step 4: Train Base Models

```bash
python scripts/training/train_base_models.py --config config.yaml
```

Fine-tunes ViT-Base, DeiT-Base, and Swin-Base with LLRD, warmup + cosine scheduling, label smoothing, gradient clipping, and MixUp augmentation.

**Expected time:** 8--12 hours total on a modern GPU.

### Step 5: Train Ensemble Meta-Learner

```bash
python scripts/training/train_ensemble.py --config config.yaml
```

Generates meta-features from holdout predictions and trains a logistic regression meta-learner with cross-validation.

**Expected time:** 30--60 minutes.

### Step 6: Comprehensive Evaluation

```bash
python scripts/evaluation/comprehensive_evaluation.py \
    --config config.yaml \
    --explainability \
    --output-dir results/evaluation
```

Produces benchmark CSV, performance visualizations, statistical significance tests, and Grad-CAM attention heatmaps.

**Expected time:** 1--2 hours.

### Step 7: Interactive Analysis

```bash
jupyter notebook notebooks/analysis.ipynb
```

---

## Expected Results

| Model | Expected Accuracy | Training Time |
|-------|-------------------|---------------|
| ViT-Base | 85--90% | 4--6 hours |
| DeiT-Base | 84--89% | 4--6 hours |
| Swin-Base | 86--91% | 5--8 hours |
| **Stacked Ensemble** | **88--93%** | **30--60 min** |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce `batch_size` in `config.yaml` (e.g., 32 to 16) |
| FaceForensics++ download fails | Re-run with `--server EU2` flag |
| CelebDF path not found | Use forward slashes on Windows, verify directory exists |
| Slow training | Train models individually with `--model vit`, `--model deit`, `--model swin` |

---

## Checklist

- [ ] Environment verified (`scripts/check_environment.py`)
- [ ] Datasets prepared
- [ ] Face crops extracted
- [ ] Data splits created
- [ ] Base models trained
- [ ] Ensemble meta-learner trained
- [ ] Evaluation completed
- [ ] Jupyter analysis reviewed
