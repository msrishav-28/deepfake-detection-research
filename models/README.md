# Models Directory

This directory contains trained model weights and configurations.

## Structure

```
models/
├── base_models/           # Individual model weights
│   ├── vit.pth           # ViT base model weights
│   ├── deit.pth          # DeiT base model weights
│   └── swin.pth          # Swin base model weights
├── ensemble/             # Ensemble model components
│   ├── meta_learner.pkl  # Trained meta-learner
│   └── ensemble.pth      # Complete ensemble weights
├── checkpoints/          # Training checkpoints
│   ├── vit_checkpoints/
│   ├── deit_checkpoints/
│   └── swin_checkpoints/
└── configs/              # Model configurations
    ├── vit_config.yaml
    ├── deit_config.yaml
    └── swin_config.yaml
```

## Model Information

### Base Models
1. **ViT (Vision Transformer)**
   - Model: `vit_base_patch16_224`
   - Input Size: 224x224
   - Parameters: ~86M

2. **DeiT (Data-efficient Image Transformer)**
   - Model: `deit_base_distilled_patch16_224`
   - Input Size: 224x224
   - Parameters: ~86M

3. **Swin (Swin Transformer)**
   - Model: `swin_base_patch4_window7_224`
   - Input Size: 224x224
   - Parameters: ~88M

### Ensemble
- **Meta-Learner**: LogisticRegression
- **Input Features**: Prediction probabilities from 3 base models
- **Output**: Final binary classification (Real/Fake)
