# Benchmarking System

Overview of the evaluation framework, benchmark reporting, and publication-ready output generation.

---

## Evaluation Framework

### Running a Full Evaluation

```bash
python scripts/evaluation/comprehensive_evaluation.py \
    --config config.yaml \
    --explainability \
    --output-dir results/evaluation
```

### Metrics Computed

| Category | Metrics |
|----------|---------|
| Classification | Accuracy, Precision, Recall, F1-Score |
| Ranking | AUC-ROC, AUC-PR |
| Efficiency | Inference time (ms), throughput (samples/sec) |
| Statistical | McNemar's test (chi-squared, p-value), confusion matrix |

---

## Benchmark Results Format

### CSV Output (timm-style)

```csv
model,dataset,accuracy,precision,recall,f1_score,auc,avg_inference_time_ms,throughput_samples_per_sec,total_samples
Vision Transformer (ViT-Base),faceforensics,0.8750,0.8654,0.8846,0.8749,0.9234,12.5,80.0,2500
Data-efficient Image Transformer (DeiT-Base),faceforensics,0.8692,0.8598,0.8788,0.8692,0.9187,8.3,120.5,2500
Swin Transformer (Swin-Base),faceforensics,0.8821,0.8734,0.8909,0.8821,0.9312,15.7,63.7,2500
Stacked Ensemble,faceforensics,0.8934,0.8856,0.9012,0.8933,0.9456,18.2,54.9,2500
```

### Performance Summary

| Model | Dataset | Accuracy | AUC | F1-Score | Inference (ms) |
|-------|---------|----------|-----|----------|----------------|
| ViT-Base | FaceForensics++ | 87.50% | 0.9234 | 0.8749 | 12.5 |
| DeiT-Base | FaceForensics++ | 86.92% | 0.9187 | 0.8692 | 8.3 |
| Swin-Base | FaceForensics++ | 88.21% | 0.9312 | 0.8821 | 15.7 |
| **Stacked Ensemble** | **FaceForensics++** | **89.34%** | **0.9456** | **0.8933** | **18.2** |

---

## Statistical Significance Testing

Model comparisons use McNemar's test with continuity correction (Dietterich, 1998). The test builds a 2x2 contingency table from paired predictions and computes:

```
chi2 = (|n01 - n10| - 1)^2 / (n01 + n10)
```

Where `n01` = count of samples classified correctly by model A but incorrectly by model B, and vice versa. A p-value below 0.05 indicates statistically significant performance difference.

### Ensemble Analysis

```json
{
  "ensemble_analysis": {
    "ensemble_accuracy": 0.8934,
    "average_individual_accuracy": 0.8754,
    "best_individual_accuracy": 0.8821,
    "improvement_over_average": 0.0180,
    "improvement_over_best": 0.0113
  }
}
```

---

## Explainability Analysis

### Grad-CAM Visualization

The framework generates attention heatmaps for each Vision Transformer model. Target layers are resolved dynamically based on the actual model architecture:

- **ViT/DeiT** -- Uses `blocks.<last>.norm2` (post-attention normalization, per Chefer et al., CVPR 2021)
- **Swin** -- Uses `layers.<last>.blocks.<last>.norm2`

This ensures correct visualization regardless of model depth (ViT-Small, ViT-Base, ViT-Large, etc.).

### Generated Outputs

| File | Description |
|------|-------------|
| `explainability_vit.png` | ViT attention heatmap |
| `explainability_deit.png` | DeiT attention heatmap |
| `explainability_swin.png` | Swin attention heatmap |

### Interpretation

- **Red/Yellow regions** -- High attention: features that most influence the classification decision
- **Blue/Green regions** -- Low attention: minimal influence on output

---

## Publication-Ready Outputs

### LaTeX Table

```latex
\begin{table}[h]
\centering
\caption{Deepfake Detection Performance Comparison}
\begin{tabular}{lcccc}
\hline
Model & Accuracy & Precision & Recall & AUC \\
\hline
ViT-Base & 87.50\% & 86.54\% & 88.46\% & 0.9234 \\
DeiT-Base & 86.92\% & 85.98\% & 87.88\% & 0.9187 \\
Swin-Base & 88.21\% & 87.34\% & 89.09\% & 0.9312 \\
\textbf{Stacked Ensemble} & \textbf{89.34\%} & \textbf{88.56\%} & \textbf{90.12\%} & \textbf{0.9456} \\
\hline
\end{tabular}
\end{table}
```

---

## Code Organization

```
deepfake_detection/evaluation/
|-- metrics.py                     # EvaluationMetrics, ModelComparator, McNemar's test
+-- explainability.py              # Grad-CAM visualization with dynamic layer resolution

scripts/evaluation/
|-- comprehensive_evaluation.py    # Full evaluation entry point
|-- benchmark_deepfake_models.py   # Benchmarking and CSV export
+-- inference_pipeline.py          # Single/batch inference

notebooks/
+-- analysis.ipynb                 # Interactive analysis
```

---

## Reproducibility

- **Fixed random seeds** via `set_seed()` across Python, NumPy, PyTorch, and CUDA
- **Deterministic cuDNN** -- `torch.backends.cudnn.deterministic = True`
- **Stratified splits** -- Class balance maintained across all four splits
- **Local RNG** in augmentations -- `np.random.default_rng(seed)` instead of global state
- **Version-controlled configuration** via `config.yaml`

---

## Expected Timeline

| Phase | Duration |
|-------|----------|
| Model training | 8--12 hours |
| Evaluation | 1--2 hours |
| Explainability analysis | 30 minutes |
| Report generation | 15 minutes |
| **Total pipeline** | **10--15 hours** |
