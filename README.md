# Deepfake Detection Research Framework

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A51.13-EE4C2C?logo=pytorch&logoColor=white)
![timm](https://img.shields.io/badge/timm-%E2%89%A50.9.2-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%E2%89%A51.0-F7931E?logo=scikit-learn&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-%E2%89%A54.5-5C3EE8?logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

A production-ready framework for deepfake detection research using stacked ensemble learning with Vision Transformers. The system combines three architectures (ViT, DeiT, Swin) through meta-learning and incorporates Grad-CAM explainability for transparent decision analysis.

---

## Research Contributions

### Methodological Innovation
- **Stacked Ensemble Architecture** -- Meta-learning applied to deepfake detection with a logistic regression meta-learner trained on holdout predictions
- **Vision Transformer Integration** -- Systematic comparison of ViT, DeiT, and Swin variants for facial manipulation detection
- **Explainable AI Framework** -- Dynamically resolved Grad-CAM attention visualization using post-attention normalization layers

### Empirical Validation
- **Comprehensive Benchmarking** -- Evaluation on FaceForensics++ and CelebDF datasets
- **Statistical Analysis** -- McNemar's test (Dietterich 1998) for pairwise model comparison
- **Cross-Dataset Generalization** -- Assessment across different manipulation techniques

---

## System Architecture

```
Raw Videos --> Face Extraction --> Quality Assessment --> Dataset Creation
                                                              |
                              +-------------------------------+
                              v
              Base Models (ViT, DeiT, Swin) --> Individual Training
                              |
                              v
                    Holdout Predictions --> Meta-Learner Training
                              |
                              v
              Model Assessment --> Statistical Analysis --> Grad-CAM Explainability
```

### Data Split Strategy

The framework uses a strict **4-way split** to prevent holdout contamination:

| Split | Ratio | Purpose |
|-------|-------|---------|
| Train | 50% | Base model training |
| Val | 10% | Base model early stopping and checkpoint selection |
| Holdout | 20% | Meta-learner training (never seen by base models) |
| Test | 20% | Final evaluation |

### Training Features

- **Layer-Wise Learning Rate Decay (LLRD)** for Vision Transformer fine-tuning (BEiT, Bao et al., ICLR 2022)
- **Warmup + Cosine Annealing** scheduler via `SequentialLR`
- **Label Smoothing** (default 0.1) in cross-entropy loss
- **Gradient Clipping** (`max_norm=1.0`) for training stability
- **MixUp Augmentation** with lambda clamping and local RNG for reproducibility
- **Real JPEG Compression Simulation** via OpenCV encode/decode round-trip

---

## Performance Benchmarks

### Expected Results (FaceForensics++ Dataset)

| Model | Accuracy | AUC | F1-Score | Inference Time |
|-------|----------|-----|----------|----------------|
| ViT-Base | 87.5% | 0.923 | 0.875 | 12.5ms |
| DeiT-Base | 86.9% | 0.919 | 0.869 | 8.3ms |
| Swin-Base | 88.2% | 0.931 | 0.882 | 15.7ms |
| **Stacked Ensemble** | **89.3%** | **0.946** | **0.893** | **18.2ms** |

---

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch >= 1.13 (required for `SequentialLR`, `LinearLR`, `label_smoothing`)
- CUDA-capable GPU (8 GB+ VRAM recommended)
- 16 GB+ RAM

### Installation

```bash
git clone https://github.com/msrishav-28/Deepfake-Detection-Research.git
cd Deepfake-Detection-Research
pip install -r requirements.txt
```

### Verify Environment

```bash
python scripts/check_environment.py
```

### Research Workflow

```bash
# 1. Prepare datasets (1-2 hours)
python scripts/data_preparation/prepare_datasets.py \
    --config config.yaml \
    --celebdf-path /path/to/celebdf

# 2. Extract faces from videos (30-45 minutes)
python scripts/data_preparation/extract_faces_from_videos.py \
    --config config.yaml

# 3. Create data splits
python scripts/data_preparation/create_splits.py --config config.yaml

# 4. Train base models (8-12 hours)
python scripts/training/train_base_models.py --config config.yaml

# 5. Train ensemble meta-learner (30-60 minutes)
python scripts/training/train_ensemble.py --config config.yaml

# 6. Comprehensive evaluation (1-2 hours)
python scripts/evaluation/comprehensive_evaluation.py \
    --config config.yaml \
    --explainability \
    --output-dir results/evaluation

# 7. Interactive analysis
jupyter notebook notebooks/analysis.ipynb
```

---

## Project Structure

```
Deepfake-Detection-Research/
|-- deepfake_detection/            # Core package
|   |-- data/                      # Data loading, augmentation, splitting
|   |-- evaluation/                # Metrics, explainability
|   |-- models/                    # Model factory, ensemble
|   +-- utils/                     # Training utilities, seeding, LLRD
|-- scripts/
|   |-- data_preparation/          # Dataset download, face extraction, splits
|   |-- training/                  # Base model and ensemble training
|   |-- evaluation/                # Benchmarking, evaluation, inference
|   +-- check_environment.py       # Environment verification
|-- notebooks/
|   +-- analysis.ipynb             # Research analysis notebook
|-- docs/                          # Extended documentation
|-- data/                          # Runtime data (gitignored)
|-- models/                        # Saved checkpoints (gitignored)
+-- results/                       # Evaluation outputs (gitignored)
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [Quick Start](docs/quick_start.md) | Step-by-step execution guide |
| [Usage Guide](docs/usage_guide.md) | Detailed pipeline instructions and configuration reference |
| [Benchmarking System](docs/benchmarking_system.md) | Evaluation framework and publication-ready outputs |
| [Dataset Specifications](docs/dataset_specifications.md) | FaceForensics++ and CelebDF processing parameters |
| [Face Extraction](docs/face_extraction.md) | Video-to-face-crop pipeline details |
| [Scripts Reference](docs/scripts.md) | Script inventory and descriptions |
| [Notebooks Reference](docs/notebooks.md) | Jupyter notebook contents |

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

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [PyTorch Image Models (timm)](https://github.com/huggingface/pytorch-image-models) -- Ross Wightman
- [FaceForensics++](https://github.com/ondyari/FaceForensics) -- Rossler et al.
- [CelebDF](https://github.com/yuezunli/celeb-deepfakeforensics) -- Li et al.
- [PyTorch](https://pytorch.org/) and [Hugging Face](https://huggingface.co/) communities
