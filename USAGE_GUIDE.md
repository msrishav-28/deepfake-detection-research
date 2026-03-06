# Deepfake Detection Research - Usage Guide

This guide provides step-by-step instructions for running the complete deepfake detection research pipeline.

## Prerequisites

### System Requirements
- Python 3.8+
- CUDA-capable GPU (recommended, 8GB+ VRAM)
- 32GB+ RAM (recommended)
- 100GB+ free disk space

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-username/deepfake-detection-research.git
cd deepfake-detection-research
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Verify installation**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

## Complete Pipeline Execution

### Step 1: Data Preparation

#### **Automated Dataset Preparation (Recommended)**

1. **Get FaceForensics++ Access**
   - Fill out the form at: https://github.com/ondyari/FaceForensics
   - You'll receive download credentials

2. **Prepare Both Datasets**
```bash
# Replace /path/to/your/celebdf with your actual CelebDF path
python scripts/data_preparation/prepare_datasets.py \
    --config config.yaml \
    --celebdf-path /path/to/your/celebdf
```

This script will:
- Download 100 videos from each FaceForensics++ category (Original, Deepfakes, Face2Face, FaceSwap, NeuralTextures)
- Organize your existing CelebDF dataset into Real/Fake categories
- Create proper directory structure

#### **Manual Dataset Preparation (Alternative)**

1. **Download FaceForensics++ (100 videos per category)**
```bash
python scripts/data_preparation/download_faceforensics.py data/raw/faceforensics
```

2. **Setup CelebDF Dataset**
```bash
python scripts/data_preparation/setup_celebdf.py /path/to/your/celebdf data/raw/celebdf
```

3. **Extract frames from videos**
```bash
python scripts/data_preparation/extract_frames.py --config config.yaml
```

4. **Detect and crop faces**
```bash
python scripts/data_preparation/face_detection.py --config config.yaml
```

5. **Create data splits**
```bash
python scripts/data_preparation/create_splits.py --config config.yaml
```

### Step 2: Train Base Models

Train all three Vision Transformer models:

```bash
# Train all models (recommended)
python scripts/training/train_base_models.py --config config.yaml

# Or train individually
python scripts/training/train_base_models.py --model vit --config config.yaml
python scripts/training/train_base_models.py --model deit --config config.yaml
python scripts/training/train_base_models.py --model swin --config config.yaml
```

**Expected training time:** 4-8 hours per model on modern GPU

### Step 3: Train Ensemble Meta-Learner

```bash
python scripts/training/train_ensemble.py --config config.yaml
```

**Expected training time:** 30-60 minutes

### Step 4: Comprehensive Evaluation

```bash
python scripts/evaluation/evaluate_models.py --config config.yaml --explainability
```

**Expected evaluation time:** 1-2 hours

### Step 5: Analysis and Visualization

Open and run the Jupyter notebook:

```bash
jupyter notebook notebooks/analysis.ipynb
```

## Quick Start (Using Pre-trained Models)

If you have pre-trained model weights:

1. **Place model weights in the correct directories:**
   - `models/base_models/vit.pth`
   - `models/base_models/deit.pth`
   - `models/base_models/swin.pth`

2. **Train ensemble meta-learner:**
```bash
python scripts/training/train_ensemble.py --config config.yaml
```

3. **Run evaluation:**
```bash
python scripts/evaluation/evaluate_models.py --config config.yaml --explainability
```

## Inference on New Data

### Single Image Prediction

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

## Configuration

The main configuration file is `config.yaml`. Key settings:

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
```

### Data Configuration
```yaml
data:
  splits:
    train_ratio: 0.6
    holdout_ratio: 0.2
    test_ratio: 0.2
    random_seed: 42
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in `config.yaml`
   - Use gradient accumulation
   - Enable mixed precision training

2. **Dataset Not Found**
   - Verify data directory structure
   - Check file permissions
   - Ensure data splits are created

3. **Model Loading Errors**
   - Check model checkpoint paths
   - Verify model architecture compatibility
   - Ensure all dependencies are installed

### Performance Optimization

1. **Training Speed**
   - Use larger batch sizes if memory allows
   - Enable mixed precision training
   - Use multiple GPUs with DataParallel

2. **Memory Usage**
   - Reduce image resolution
   - Use gradient checkpointing
   - Optimize data loading with more workers

## Expected Results

### Performance Benchmarks

Based on FaceForensics++ dataset:

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| ViT   | 0.85-0.90| 0.84-0.89 | 0.86-0.91| 0.85-0.90|
| DeiT  | 0.84-0.89| 0.83-0.88 | 0.85-0.90| 0.84-0.89|
| Swin  | 0.86-0.91| 0.85-0.90 | 0.87-0.92| 0.86-0.91|
| **Ensemble** | **0.88-0.93**| **0.87-0.92** | **0.89-0.94**| **0.88-0.93**|

*Note: Actual results may vary based on dataset quality and training conditions.*

### File Structure After Completion

```
deepfake-detection-research/
├── data/
│   ├── processed/          # Processed images
│   ├── splits/            # Data split files
│   └── raw/               # Original datasets
├── models/
│   ├── base_models/       # Trained model weights
│   │   ├── vit.pth
│   │   ├── deit.pth
│   │   └── swin.pth
│   └── ensemble/          # Ensemble components
│       ├── meta_learner.pkl
│       └── ensemble_config.pkl
├── results/
│   └── evaluation/        # Evaluation results
│       ├── model_comparison.csv
│       ├── detailed_results.json
│       └── explainability/
└── notebooks/
    └── analysis.ipynb     # Research analysis
```

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{deepfake_detection_ensemble_2025,
  title={Deepfake Detection using Stacked Ensemble of Vision Transformers},
   author={msrishav-28},
  year={2025},
   howpublished={\url{https://github.com/msrishav-28/Deepfake-Detection-Research}}
}
```

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the configuration settings
3. Open an issue on GitHub with detailed error messages
4. Include system specifications and dataset information

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PyTorch Image Models (timm) library
- FaceForensics++ dataset creators
- CelebDF dataset creators
- PyTorch and Hugging Face communities
