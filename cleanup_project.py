#!/usr/bin/env python3
"""
Project Cleanup Script for Deepfake Detection Research

This script removes unnecessary files from the original timm repository
that are not needed for the deepfake detection project.
"""

import os
import shutil
import sys
from pathlib import Path

def remove_file_or_dir(path):
    """Safely remove a file or directory."""
    if os.path.exists(path):
        if os.path.isfile(path):
            os.remove(path)
            print(f"Removed file: {path}")
        elif os.path.isdir(path):
            shutil.rmtree(path)
            print(f"Removed directory: {path}")
    else:
        print(f"Not found (skipping): {path}")

def cleanup_project():
    """Remove unnecessary files and directories."""
    
    print("🧹 Starting project cleanup...")
    print("Removing files not needed for deepfake detection research...")
    
    # Files to remove - original timm specific files
    files_to_remove = [
        # Benchmark and validation scripts (original timm)
        "benchmark.py",
        "validate.py", 
        "train.py",  # We have our own training scripts
        "inference.py",  # We have our own inference pipeline
        
        # Conversion utilities
        "convert",
        
        # Original timm utilities
        "avg_checkpoints.py",
        "bulk_runner.py", 
        "clean_checkpoint.py",
        "onnx_export.py",
        "onnx_validate.py",
        "hubconf.py",
        
        # Development and testing files
        "tests",
        "requirements-dev.txt",
        
        # Documentation that's not relevant
        "hfdocs",
        "CONTRIBUTING.md",
        "CODE_OF_CONDUCT.md",
        "UPGRADING.md",
        "CITATION.cff",
        "MANIFEST.in",
        "setup.cfg",
        "pyproject.toml",
        
        # Original benchmark results
        "results/benchmark-infer-amp-nchw-pt113-cu117-rtx3090.csv",
        "results/benchmark-infer-amp-nchw-pt210-cu121-rtx3090.csv", 
        "results/benchmark-infer-amp-nchw-pt240-cu124-rtx3090.csv",
        "results/benchmark-infer-amp-nchw-pt240-cu124-rtx4090-dynamo.csv",
        "results/benchmark-infer-amp-nchw-pt240-cu124-rtx4090.csv",
        "results/benchmark-infer-amp-nhwc-pt113-cu117-rtx3090.csv",
        "results/benchmark-infer-amp-nhwc-pt210-cu121-rtx3090.csv",
        "results/benchmark-infer-amp-nhwc-pt240-cu124-rtx3090.csv",
        "results/benchmark-infer-amp-nhwc-pt240-cu124-rtx4090.csv",
        "results/benchmark-infer-fp32-nchw-pt221-cpu-i9_10940x-dynamo.csv",
        "results/benchmark-infer-fp32-nchw-pt240-cpu-i7_12700h-dynamo.csv",
        "results/benchmark-infer-fp32-nchw-pt240-cpu-i9_10940x-dynamo.csv",
        "results/benchmark-train-amp-nchw-pt112-cu113-rtx3090.csv",
        "results/benchmark-train-amp-nhwc-pt112-cu113-rtx3090.csv",
        "results/generate_csv_results.py",
        "results/model_metadata-in1k.csv",
        "results/results-imagenet-a-clean.csv",
        "results/results-imagenet-a.csv",
        "results/results-imagenet-r-clean.csv", 
        "results/results-imagenet-r.csv",
        "results/results-imagenet-real.csv",
        "results/results-imagenet.csv",
        "results/results-imagenetv2-matched-frequency.csv",
        "results/results-sketch.csv",
        
        # Shell scripts for distributed training (we have our own)
        "distributed_train.sh",
    ]
    
    # Remove files
    for file_path in files_to_remove:
        remove_file_or_dir(file_path)
    
    # Clean up results directory but keep the README
    results_readme_content = None
    results_readme_path = "results/README.md"
    if os.path.exists(results_readme_path):
        with open(results_readme_path, 'r') as f:
            results_readme_content = f.read()
    
    # Remove all files in results except README
    if os.path.exists("results"):
        for item in os.listdir("results"):
            item_path = os.path.join("results", item)
            if item != "README.md":
                remove_file_or_dir(item_path)
    
    # Restore results README with updated content
    if results_readme_content:
        with open(results_readme_path, 'w') as f:
            f.write("# Results Directory\n\n")
            f.write("This directory contains evaluation results from the deepfake detection research.\n\n")
            f.write("## Structure\n\n")
            f.write("```\n")
            f.write("results/\n")
            f.write("├── evaluation/           # Model evaluation results\n")
            f.write("│   ├── model_comparison.csv\n")
            f.write("│   ├── detailed_results.json\n")
            f.write("│   └── explainability/  # Grad-CAM visualizations\n")
            f.write("└── training/            # Training logs and metrics\n")
            f.write("```\n")
    
    print("\n✅ Cleanup completed!")
    print("\n📁 Remaining project structure:")
    print("├── deepfake_detection/     # Main package")
    print("├── scripts/               # Training and evaluation scripts") 
    print("├── notebooks/             # Research analysis")
    print("├── timm/                  # Core timm library (kept)")
    print("├── data/                  # Dataset storage")
    print("├── models/                # Model weights")
    print("├── results/               # Evaluation results")
    print("├── config.yaml            # Project configuration")
    print("├── USAGE_GUIDE.md         # Usage instructions")
    print("├── requirements.txt       # Dependencies")
    print("└── LICENSE                # License file")
    
    print(f"\n💾 Estimated space saved: ~50-100MB")
    print("🎯 Project is now focused on deepfake detection research!")

if __name__ == "__main__":
    # Confirm before cleanup
    response = input("⚠️  This will remove original timm files not needed for deepfake detection. Continue? (y/N): ")
    if response.lower() in ['y', 'yes']:
        cleanup_project()
    else:
        print("Cleanup cancelled.")
