# Usage Guide

Detailed instructions for configuring and running the deepfake detection research pipeline.

---

## Prerequisites

### System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.8+ | 3.10+ |
| PyTorch | >= 1.13 | >= 2.0 |
| GPU VRAM | 4 GB | 8 GB+ |
| RAM | 16 GB | 32 GB |
| Disk Space | 50 GB | 100 GB |

### Installation

```bash
git clone https://github.com/msrishav-28/Deepfake-Detection-Research.git
cd Deepfake-Detection-Research
pip install -r requirements.txt
python scripts/check_environment.py
```

---

## Complete Pipeline

### Step 1: Data Preparation

**Automated preparation (recommended):**

```bash
python scripts/data_preparation/prepare_datasets.py \
    --config config.yaml \
    --celebdf-path /path/to/your/celebdf
```

This script downloads FaceForensics++ (100 videos per category, c23 compression) and organizes CelebDF into `Real/Fake` directories.

**Manual preparation (alternative):**

```bash
# Download FaceForensics++ (requires access credentials)
python scripts/data_preparation/download_faceforensics.py data/raw/faceforensics

# Organize CelebDF
python scripts/data_preparation/setup_celebdf.py /path/to/your/celebdf data/raw/celebdf

# Extract face crops from videos
python scripts/data_preparation/extract_faces_from_videos.py --config config.yaml

# Create stratified data splits
python scripts/data_preparation/create_splits.py --config config.yaml
```

### Step 2: Train Base Models

```bash
# Train all three models
python scripts/training/train_base_models.py --config config.yaml

# Or train individually
python scripts/training/train_base_models.py --model vit --config config.yaml
python scripts/training/train_base_models.py --model deit --config config.yaml
python scripts/training/train_base_models.py --model swin --config config.yaml
```

**Expected training time:** 4--8 hours per model on a modern GPU.

### Step 3: Train Ensemble Meta-Learner

```bash
python scripts/training/train_ensemble.py --config config.yaml
```

**Expected training time:** 30--60 minutes.

### Step 4: Evaluation

```bash
python scripts/evaluation/comprehensive_evaluation.py \
    --config config.yaml \
    --explainability \
    --output-dir results/evaluation
```

**Expected evaluation time:** 1--2 hours.

### Step 5: Analysis

```bash
jupyter notebook notebooks/analysis.ipynb
```

---

## Inference on New Data

### Single Image

```bash
python scripts/evaluation/inference_pipeline.py \
    --config config.yaml \
    --input path/to/image.jpg \
    --output results.json
```

### Batch Prediction

```bash
python scripts/evaluation/inference_pipeline.py \
    --config config.yaml \
    --input path/to/image/directory \
    --output batch_results.json \
    --batch-size 32 \
    --analyze
```

---

## Configuration Reference

The main configuration file is `config.yaml`. Key sections:

### Model Configuration

```yaml
models:
  base_models:
    vit:
      name: "vit_base_patch16_224"
      pretrained: true
      num_classes: 2
    deit:
      name: "deit_base_distilled_patch16_224"
      pretrained: true
      num_classes: 2
    swin:
      name: "swin_base_patch4_window7_224"
      pretrained: true
      num_classes: 2
```

### Training Configuration

```yaml
training:
  base_models:
    epochs: 50
    batch_size: 32
    learning_rate: 1e-4
    weight_decay: 1e-5
    label_smoothing: 0.1        # Cross-entropy label smoothing
    layer_decay: 0.75           # LLRD decay factor
    warmup_epochs: 5            # Linear warmup before cosine decay
    min_lr: 1e-6                # Cosine annealing floor
```

### Data Split Configuration

```yaml
data:
  splits:
    train_ratio: 0.5            # Base model training
    val_ratio: 0.1              # Base model early stopping
    holdout_ratio: 0.2          # Meta-learner training
    test_ratio: 0.2             # Final evaluation
    random_seed: 42
```

---

## File Structure After Completion

```
Deepfake-Detection-Research/
|-- data/
|   |-- raw/                   # Original video datasets
|   |-- processed/             # Extracted face crops
|   +-- splits/                # Split definition files
|-- models/
|   |-- base_models/           # Trained model weights
|   |   |-- vit.pth
|   |   |-- deit.pth
|   |   +-- swin.pth
|   +-- ensemble/              # Meta-learner and ensemble config
|       |-- meta_learner.pkl
|       +-- ensemble_config.pkl
|-- results/
|   +-- evaluation/            # Evaluation outputs
|       |-- model_comparison.csv
|       |-- detailed_results.json
|       +-- explainability/    # Grad-CAM heatmaps
+-- notebooks/
    +-- analysis.ipynb         # Research analysis
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce `batch_size` in `config.yaml`; enable mixed precision |
| Dataset not found | Verify `data/` directory structure; check that splits exist |
| Model loading errors | Confirm checkpoint paths; verify architecture compatibility |
| Slow training | Use larger batch size if memory allows; reduce image resolution |
| Memory pressure | Enable gradient checkpointing; reduce number of data loader workers |

---

## Citation

```bibtex
@misc{deepfake_detection_ensemble_2025,
  title   = {Deepfake Detection using Stacked Ensemble of Vision Transformers},
  author  = {msrishav-28},
  year    = {2025},
  howpublished = {\url{https://github.com/msrishav-28/Deepfake-Detection-Research}}
}
```

## Acknowledgments

- [PyTorch Image Models (timm)](https://github.com/huggingface/pytorch-image-models) -- Ross Wightman
- [FaceForensics++](https://github.com/ondyari/FaceForensics) -- Rossler et al.
- [CelebDF](https://github.com/yuezunli/celeb-deepfakeforensics) -- Li et al.
