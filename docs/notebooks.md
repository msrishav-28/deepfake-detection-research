# Notebooks Reference

Reference for Jupyter notebooks in the `notebooks/` directory.

## Directory Structure

```
notebooks/
+-- analysis.ipynb    # Primary research analysis notebook
```

## Notebook Description

### analysis.ipynb

The main research notebook for interactive exploration of training results, model performance, and explainability outputs. Contents include:

- **Methodology Documentation** -- Overview of the stacked ensemble architecture and training protocol
- **Training Curve Analysis** -- Loss and accuracy progression across epochs for each base model
- **Performance Comparison** -- Side-by-side metrics (accuracy, AUC, F1, precision, recall) for ViT, DeiT, Swin, and the stacked ensemble
- **Statistical Analysis** -- McNemar's test results for pairwise model significance testing
- **Grad-CAM Visualizations** -- Attention heatmaps showing where each architecture focuses when classifying real vs. fake faces
- **Research Conclusions** -- Summary of findings and directions for future work

### Usage

```bash
jupyter notebook notebooks/analysis.ipynb
```

> **Note:** Notebook outputs are automatically stripped on commit via `nbstripout`. Re-run all cells after cloning to regenerate outputs.
