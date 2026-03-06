# Quick Start Guide - Deepfake Detection Research

## Your Current Setup

✅ **FaceForensics++ Download Script**: Ready to download 100 videos per category  
✅ **CelebDF Dataset**: You have this downloaded on your system  
✅ **Project Structure**: Clean and research-focused  
✅ **Scripts**: All data preparation scripts ready  

---

## 🎯 **Step-by-Step Execution**

### **Step 1: Prepare Your Datasets**

Since you have CelebDF already downloaded, run this single command:

```bash
# Replace with your actual CelebDF path
python scripts/data_preparation/prepare_datasets.py \
    --config config.yaml \
    --celebdf-path "C:/path/to/your/celebdf/dataset"
```

**What this does:**
- ✅ Downloads FaceForensics++ c23 compression (100 videos × 5 categories = 500 MP4s)
- ✅ Organizes your CelebDF into Real/Fake categories
- ✅ Creates proper directory structure
- ✅ Generates dataset summaries

**Expected time:** 1-2 hours (c23 compression is smaller than raw)
**Dataset sizes:** ~1GB FaceForensics++ + your CelebDF size

### **Step 2: Extract Faces from Videos (OPTIMIZED!)**

```bash
# Single command to extract faces from both datasets with optimized settings
python scripts/data_preparation/process_deepfake_datasets.py \
    --dataset both --config config.yaml
```

**What this does:**
- ✅ **FaceForensics++**: OpenCV detection optimized for c23 compression (fast)
- ✅ **CelebDF**: MTCNN detection optimized for high-quality videos (accurate)
- ✅ Extracts ~1 frame per second from 10-15 second clips
- ✅ Quality assessment and filtering (keeps only good faces)
- ✅ Generates 6-10 best faces per video

**Expected time:** 30-45 minutes total
**Output:** ~3,000-6,000 high-quality face crops ready for training

### **Step 3: Create Data Splits**

```bash
# Create train/holdout/test splits (60%/20%/20%)
python scripts/data_preparation/create_splits.py --config config.yaml
```

**Expected time:** 5-10 minutes

### **Step 4: Train Models**

```bash
# Train all three base models (ViT, DeiT, Swin)
python scripts/training/train_base_models.py --config config.yaml

# Train ensemble meta-learner
python scripts/training/train_ensemble.py --config config.yaml
```

**Expected time:** 8-12 hours (optimized dataset size reduces training time)

### **Step 5: Comprehensive Evaluation**

```bash
# Professional evaluation framework with explainability analysis
python scripts/evaluation/comprehensive_evaluation.py \
    --config config.yaml \
    --explainability \
    --output-dir results/evaluation

# Interactive research analysis
jupyter notebook notebooks/analysis.ipynb
```

**Expected time:** 1-2 hours
**Outputs:** Benchmark CSV, performance visualizations, Grad-CAM explainability

---

## 📊 **Expected Dataset Structure After Step 1**

```
data/
├── raw/
│   ├── faceforensics/          # 500 videos total
│   │   ├── original/           # 100 real videos
│   │   ├── Deepfakes/          # 100 deepfake videos
│   │   ├── Face2Face/          # 100 face2face videos
│   │   ├── FaceSwap/           # 100 faceswap videos
│   │   └── NeuralTextures/     # 100 neural texture videos
│   └── celebdf/                # Your CelebDF organized
│       ├── Real/               # Real celebrity videos
│       └── Fake/               # Fake celebrity videos
├── processed/                  # After frame extraction
└── splits/                     # After data splitting
```

---

## 🔧 **Troubleshooting Common Issues**

### **Issue 1: FaceForensics++ Download Fails**
```bash
# Try different server
python scripts/data_preparation/download_faceforensics.py \
    data/raw/faceforensics --server EU2
```

### **Issue 2: CelebDF Path Not Found**
```bash
# Check your CelebDF path
ls "C:/path/to/your/celebdf"

# Use forward slashes even on Windows
python scripts/data_preparation/prepare_datasets.py \
    --celebdf-path "C:/Users/YourName/Downloads/CelebDF"
```

### **Issue 3: CUDA Out of Memory**
Edit `config.yaml`:
```yaml
training:
  base_models:
    batch_size: 16  # Reduce from 32
```

### **Issue 4: Slow Training**
```bash
# Train one model at a time
python scripts/training/train_base_models.py --model vit --config config.yaml
python scripts/training/train_base_models.py --model deit --config config.yaml
python scripts/training/train_base_models.py --model swin --config config.yaml
```

---

## 📈 **Expected Results**

### **Dataset Size**
- **FaceForensics++**: ~500 videos (100 per category)
- **CelebDF**: Variable (depends on your download)
- **Total frames**: ~50,000-100,000 after extraction
- **Disk space**: ~15-25 GB

### **Model Performance**
| Model | Expected Accuracy | Training Time |
|-------|------------------|---------------|
| ViT   | 85-90%          | 4-6 hours     |
| DeiT  | 84-89%          | 4-6 hours     |
| Swin  | 86-91%          | 5-8 hours     |
| **Ensemble** | **88-93%** | **1 hour**    |

### **Research Outputs**
- ✅ Trained model weights
- ✅ Performance comparison tables
- ✅ Grad-CAM visualizations
- ✅ Statistical analysis
- ✅ Interactive Jupyter notebook
- ✅ Publication-ready results

---

## 🎯 **One-Command Full Pipeline (Advanced)**

If you want to run everything automatically:

```bash
# Full pipeline (will take 20-30 hours total)
python scripts/data_preparation/prepare_datasets.py --celebdf-path "C:/path/to/celebdf" && \
python scripts/data_preparation/extract_frames.py --config config.yaml && \
python scripts/data_preparation/face_detection.py --config config.yaml && \
python scripts/data_preparation/create_splits.py --config config.yaml && \
python scripts/training/train_base_models.py --config config.yaml && \
python scripts/training/train_ensemble.py --config config.yaml && \
python scripts/evaluation/evaluate_models.py --config config.yaml --explainability
```

---

## 📋 **Checklist**

- [ ] **Step 1**: Dataset preparation completed
- [ ] **Step 2**: Frame extraction and face detection completed  
- [ ] **Step 3**: Data splits created
- [ ] **Step 4**: Base models trained
- [ ] **Step 5**: Ensemble trained
- [ ] **Step 6**: Evaluation completed
- [ ] **Step 7**: Jupyter notebook analysis completed

---

## 🎉 **You're Ready to Start!**

Your deepfake detection research project is fully set up and ready for execution. The scripts will handle all the complex details while you focus on the research insights.

**Start with:**
```bash
python scripts/data_preparation/prepare_datasets.py \
    --config config.yaml \
    --celebdf-path "YOUR_CELEBDF_PATH_HERE"
```

**Professional deepfake detection research framework ready for execution.**
