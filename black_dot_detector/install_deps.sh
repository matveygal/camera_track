#!/bin/bash

# Black Dot Detector - Dependency Installation Script
# This script sets up a fresh virtual environment and installs all dependencies

echo "========================================="
echo "Black Dot Detector - Setup Script"
echo "========================================="
echo ""

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Get Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python version: $PYTHON_VERSION"
echo ""

# Create virtual environment
VENV_DIR="venv_blackdot"
echo "Creating virtual environment in ./$VENV_DIR..."
python3 -m venv $VENV_DIR

# Activate virtual environment
echo "Activating virtual environment..."
source $VENV_DIR/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (with CUDA support if available, otherwise CPU)
echo ""
echo "Installing PyTorch..."
# For macOS or CPU-only Linux:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
echo ""
echo "Installing other dependencies..."
pip install -r requirements.txt

# Verify critical installations
echo ""
echo "Verifying installations..."
python3 -c "import cv2; print(f'✓ OpenCV {cv2.__version__}')"
python3 -c "import torch; print(f'✓ PyTorch {torch.__version__}')"
python3 -c "import pyrealsense2 as rs; print(f'✓ pyrealsense2 installed')" 2>/dev/null || echo "⚠ pyrealsense2 not detected (may need manual installation)"
python3 -c "import ultralytics; print(f'✓ Ultralytics installed')"

echo ""
echo "========================================="
echo "Installation complete!"
echo "========================================="
echo ""
echo "To activate the environment, run:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "To deactivate, run:"
echo "  deactivate"
echo ""
