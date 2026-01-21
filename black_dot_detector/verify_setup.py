#!/usr/bin/env python3
"""
Project Verification Script
Checks that all components are properly installed and configured.
"""

import sys
from pathlib import Path
import importlib.util


def check_file(path, description):
    """Check if a file exists."""
    if Path(path).exists():
        print(f"✅ {description}: {path}")
        return True
    else:
        print(f"❌ MISSING {description}: {path}")
        return False


def check_directory(path, description):
    """Check if a directory exists."""
    if Path(path).is_dir():
        print(f"✅ {description}: {path}")
        return True
    else:
        print(f"❌ MISSING {description}: {path}")
        return False


def check_import(module_name):
    """Check if a Python module can be imported."""
    try:
        __import__(module_name)
        print(f"✅ Python module: {module_name}")
        return True
    except ImportError:
        print(f"❌ MISSING Python module: {module_name}")
        return False


def main():
    """Run verification checks."""
    print("=" * 60)
    print("Black Dot Detection System - Verification")
    print("=" * 60)
    print()
    
    all_checks_passed = True
    
    # Check core Python files
    print("\n📄 Core Python Files")
    print("-" * 60)
    files = [
        ("annotator.py", "Annotation tool"),
        ("train_model.py", "Training pipeline"),
        ("detect_dot.py", "Detection application"),
        ("camera_utils.py", "Camera utilities"),
        ("utils.py", "General utilities"),
        ("quick_start.py", "Quick start guide"),
    ]
    
    for file, desc in files:
        if not check_file(file, desc):
            all_checks_passed = False
    
    # Check configuration and documentation
    print("\n📋 Configuration & Documentation")
    print("-" * 60)
    config_files = [
        ("config.yaml", "Configuration file"),
        ("requirements.txt", "Requirements file"),
        ("README.md", "Documentation"),
        ("PROJECT_SUMMARY.md", "Project summary"),
    ]
    
    for file, desc in config_files:
        if not check_file(file, desc):
            all_checks_passed = False
    
    # Check installation scripts
    print("\n🔧 Installation Scripts")
    print("-" * 60)
    install_files = [
        ("install_deps.py", "Python installer"),
        ("install_deps.sh", "Bash installer"),
    ]
    
    for file, desc in install_files:
        if not check_file(file, desc):
            all_checks_passed = False
    
    # Check directories
    print("\n📁 Directory Structure")
    print("-" * 60)
    directories = [
        ("data", "Data root"),
        ("data/raw_images", "Raw images"),
        ("data/camera_captures", "Camera captures"),
        ("data/dataset", "Training dataset"),
        ("models", "Models root"),
        ("models/trained_models", "Trained models"),
        ("results", "Results root"),
        ("results/detections", "Detection results"),
        ("results/training_metrics", "Training metrics"),
    ]
    
    for dir_path, desc in directories:
        if not check_directory(dir_path, desc):
            all_checks_passed = False
    
    # Check Python environment
    print("\n🐍 Python Environment")
    print("-" * 60)
    
    py_version = sys.version_info
    print(f"✅ Python version: {py_version.major}.{py_version.minor}.{py_version.micro}")
    
    if py_version.major < 3 or (py_version.major == 3 and py_version.minor < 8):
        print("⚠️  Warning: Python 3.8+ recommended")
        all_checks_passed = False
    
    # Check if virtual environment exists
    venv_dir = Path("venv_blackdot")
    if venv_dir.exists():
        print(f"✅ Virtual environment: {venv_dir}")
        
        # Try to check dependencies
        print("\n📦 Python Dependencies (if venv activated)")
        print("-" * 60)
        
        dependencies = [
            "cv2",
            "numpy",
            "torch",
            "yaml",
            "PIL",
        ]
        
        for dep in dependencies:
            check_import(dep)
        
        # Special check for pyrealsense2 and ultralytics
        try:
            import pyrealsense2
            print(f"✅ Python module: pyrealsense2")
        except ImportError:
            print(f"⚠️  Optional module: pyrealsense2 (needed for camera)")
        
        try:
            import ultralytics
            print(f"✅ Python module: ultralytics")
        except ImportError:
            print(f"⚠️  Optional module: ultralytics (needed for training)")
    else:
        print(f"⚠️  Virtual environment not found: {venv_dir}")
        print("   Run install_deps.py to create it")
        all_checks_passed = False
    
    # Check file sizes (sanity check)
    print("\n📊 File Size Verification")
    print("-" * 60)
    
    file_sizes = {
        "annotator.py": 10000,      # At least 10KB
        "train_model.py": 10000,
        "detect_dot.py": 10000,
        "camera_utils.py": 10000,
        "utils.py": 10000,
        "README.md": 10000,
    }
    
    for file, min_size in file_sizes.items():
        path = Path(file)
        if path.exists():
            size = path.stat().st_size
            if size >= min_size:
                print(f"✅ {file}: {size:,} bytes")
            else:
                print(f"⚠️  {file}: {size:,} bytes (expected >{min_size:,})")
        else:
            print(f"❌ {file}: Not found")
            all_checks_passed = False
    
    # Final summary
    print("\n" + "=" * 60)
    if all_checks_passed:
        print("✅ ALL CHECKS PASSED!")
        print("=" * 60)
        print("\nYour Black Dot Detection System is properly set up!")
        print("\nNext steps:")
        print("  1. Activate virtual environment: source venv_blackdot/bin/activate")
        print("  2. Run quick start: python quick_start.py")
        print("  3. Or see README.md for manual usage")
    else:
        print("⚠️  SOME CHECKS FAILED")
        print("=" * 60)
        print("\nPlease review the errors above and:")
        print("  1. Make sure you're in the black_dot_detector directory")
        print("  2. Run install_deps.py if environment not set up")
        print("  3. Check file permissions")
    
    print()
    
    return 0 if all_checks_passed else 1


if __name__ == "__main__":
    sys.exit(main())
