#!/usr/bin/env python3
"""
Test Setup Script for Deepfake Detection Research

This script verifies that all dependencies and components are properly installed
and configured for the deepfake detection research project.

Usage:
    python test_setup.py
"""

import sys
import os
import importlib
import torch
from pathlib import Path

def test_python_version():
    """Test Python version compatibility."""
    print("🐍 Testing Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} (Compatible)")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} (Requires Python 3.8+)")
        return False

def test_pytorch():
    """Test PyTorch installation and CUDA availability."""
    print("🔥 Testing PyTorch...")
    try:
        print(f"   ✅ PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"   ✅ CUDA available: {gpu_count} GPU(s)")
            print(f"   ✅ GPU: {gpu_name}")
        else:
            print("   ⚠️  CUDA not available (will use CPU - slower training)")
        return True
    except Exception as e:
        print(f"   ❌ PyTorch error: {e}")
        return False

def test_dependencies():
    """Test required dependencies."""
    print("📦 Testing dependencies...")
    
    required_packages = [
        'timm',
        'sklearn',
        'numpy',
        'pandas', 
        'matplotlib',
        'seaborn',
        'tqdm',
        'opencv-cv2',
        'PIL',
        'yaml',
        'pytorch_grad_cam'
    ]
    
    success = True
    for package in required_packages:
        try:
            if package == 'opencv-cv2':
                import cv2
                print(f"   ✅ opencv-python {cv2.__version__}")
            elif package == 'PIL':
                from PIL import Image
                print(f"   ✅ Pillow (PIL)")
            elif package == 'yaml':
                import yaml
                print(f"   ✅ PyYAML")
            elif package == 'pytorch_grad_cam':
                import pytorch_grad_cam
                print(f"   ✅ pytorch-grad-cam")
            else:
                module = importlib.import_module(package)
                version = getattr(module, '__version__', 'unknown')
                print(f"   ✅ {package} {version}")
        except ImportError:
            print(f"   ❌ {package} not found")
            success = False
        except Exception as e:
            print(f"   ⚠️  {package} error: {e}")
    
    return success

def test_project_structure():
    """Test project directory structure."""
    print("📁 Testing project structure...")
    
    required_dirs = [
        'deepfake_detection',
        'scripts',
        'scripts/data_preparation',
        'scripts/training', 
        'scripts/evaluation',
        'notebooks',
        'timm'
    ]
    
    required_files = [
        'config.yaml',
        'requirements.txt',
        'USAGE_GUIDE.md',
        'QUICK_START_GUIDE.md',
        'notebooks/analysis.ipynb'
    ]
    
    success = True
    
    # Check directories
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"   ✅ {dir_path}/")
        else:
            print(f"   ❌ {dir_path}/ not found")
            success = False
    
    # Check files
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} not found")
            success = False
    
    return success

def test_timm_models():
    """Test timm model creation."""
    print("🤖 Testing timm model creation...")
    
    models_to_test = [
        'vit_base_patch16_224',
        'deit_base_distilled_patch16_224', 
        'swin_base_patch4_window7_224'
    ]
    
    success = True
    try:
        import timm
        
        for model_name in models_to_test:
            try:
                model = timm.create_model(model_name, pretrained=False, num_classes=2)
                param_count = sum(p.numel() for p in model.parameters())
                print(f"   ✅ {model_name} ({param_count/1e6:.1f}M params)")
            except Exception as e:
                print(f"   ❌ {model_name} failed: {e}")
                success = False
                
    except Exception as e:
        print(f"   ❌ timm import failed: {e}")
        success = False
    
    return success

def test_data_directories():
    """Test data directory structure."""
    print("💾 Testing data directories...")
    
    data_dirs = ['data', 'models', 'results']
    
    for dir_name in data_dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
            print(f"   ✅ Created {dir_name}/")
        else:
            print(f"   ✅ {dir_name}/ exists")
    
    # Create subdirectories
    subdirs = [
        'data/raw',
        'data/processed', 
        'data/splits',
        'models/base_models',
        'models/ensemble',
        'results/evaluation',
        'results/training'
    ]
    
    for subdir in subdirs:
        if not os.path.exists(subdir):
            os.makedirs(subdir, exist_ok=True)
            print(f"   ✅ Created {subdir}/")
        else:
            print(f"   ✅ {subdir}/ exists")
    
    return True

def test_config_file():
    """Test configuration file."""
    print("⚙️  Testing configuration...")
    
    try:
        import yaml
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        required_sections = ['paths', 'models', 'training', 'data']
        
        for section in required_sections:
            if section in config:
                print(f"   ✅ config.{section}")
            else:
                print(f"   ❌ config.{section} missing")
                return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ Config file error: {e}")
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("🧪 DEEPFAKE DETECTION RESEARCH - SETUP TEST")
    print("="*60)
    
    tests = [
        ("Python Version", test_python_version),
        ("PyTorch & CUDA", test_pytorch),
        ("Dependencies", test_dependencies),
        ("Project Structure", test_project_structure),
        ("timm Models", test_timm_models),
        ("Data Directories", test_data_directories),
        ("Configuration", test_config_file)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"   ❌ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("📋 TEST SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20}: {status}")
        if result:
            passed += 1
    
    print("-"*40)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("Your setup is ready for deepfake detection research!")
        print("\nNext step: Run dataset preparation")
        print("python scripts/data_preparation/prepare_datasets.py --celebdf-path YOUR_PATH")
    else:
        print(f"\n⚠️  {total - passed} TESTS FAILED!")
        print("Please fix the issues above before proceeding.")
        print("Check USAGE_GUIDE.md for installation instructions.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
