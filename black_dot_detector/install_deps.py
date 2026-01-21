#!/usr/bin/env python3
"""
Black Dot Detector - Python Installation Script
Cross-platform dependency installer with virtual environment setup
"""

import sys
import subprocess
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{description}...")
    try:
        subprocess.check_call(cmd, shell=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return False

def main():
    print("=" * 50)
    print("Black Dot Detector - Setup Script")
    print("=" * 50)
    print()
    
    # Check Python version
    py_version = sys.version_info
    if py_version.major < 3 or (py_version.major == 3 and py_version.minor < 8):
        print(f"Error: Python 3.8+ required. Found {py_version.major}.{py_version.minor}")
        sys.exit(1)
    
    print(f"✓ Python {py_version.major}.{py_version.minor}.{py_version.micro}")
    
    # Virtual environment directory
    venv_dir = Path("venv_blackdot")
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Create virtual environment
    if not venv_dir.exists():
        print(f"\nCreating virtual environment in {venv_dir}...")
        if not run_command(f"{sys.executable} -m venv {venv_dir}", 
                          "Setting up virtual environment"):
            sys.exit(1)
    else:
        print(f"\n✓ Virtual environment already exists at {venv_dir}")
    
    # Determine pip executable
    if sys.platform == "win32":
        pip_exe = venv_dir / "Scripts" / "pip.exe"
        python_exe = venv_dir / "Scripts" / "python.exe"
    else:
        pip_exe = venv_dir / "bin" / "pip"
        python_exe = venv_dir / "bin" / "python"
    
    # Upgrade pip
    run_command(f"{python_exe} -m pip install --upgrade pip", 
                "Upgrading pip")
    
    # Install PyTorch
    print("\n" + "=" * 50)
    print("Installing PyTorch...")
    print("=" * 50)
    
    if sys.platform == "darwin":  # macOS
        pytorch_cmd = f"{pip_exe} install torch torchvision --index-url https://download.pytorch.org/whl/cpu"
    elif sys.platform == "win32":  # Windows
        pytorch_cmd = f"{pip_exe} install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
    else:  # Linux - try CUDA, fallback to CPU
        pytorch_cmd = f"{pip_exe} install torch torchvision"
    
    run_command(pytorch_cmd, "Installing PyTorch and torchvision")
    
    # Install requirements
    print("\n" + "=" * 50)
    print("Installing other dependencies...")
    print("=" * 50)
    
    requirements_file = script_dir / "requirements.txt"
    if requirements_file.exists():
        run_command(f"{pip_exe} install -r {requirements_file}", 
                   "Installing from requirements.txt")
    else:
        print("Warning: requirements.txt not found!")
    
    # Verify installations
    print("\n" + "=" * 50)
    print("Verifying installations...")
    print("=" * 50)
    
    packages_to_check = [
        ("cv2", "OpenCV"),
        ("torch", "PyTorch"),
        ("pyrealsense2", "RealSense SDK"),
        ("ultralytics", "Ultralytics"),
        ("yaml", "PyYAML"),
        ("numpy", "NumPy"),
        ("PIL", "Pillow"),
    ]
    
    for module, name in packages_to_check:
        try:
            result = subprocess.run(
                [str(python_exe), "-c", f"import {module}; print({module}.__version__ if hasattr({module}, '__version__') else 'OK')"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                version = result.stdout.strip()
                print(f"✓ {name}: {version}")
            else:
                print(f"⚠ {name}: Import error")
        except Exception as e:
            print(f"⚠ {name}: Could not verify ({e})")
    
    print("\n" + "=" * 50)
    print("Installation Complete!")
    print("=" * 50)
    print()
    print("To activate the virtual environment:")
    if sys.platform == "win32":
        print(f"  {venv_dir}\\Scripts\\activate")
    else:
        print(f"  source {venv_dir}/bin/activate")
    print()
    print("To deactivate:")
    print("  deactivate")
    print()
    print("Quick Start:")
    print("  1. Activate the virtual environment (see above)")
    print("  2. Run the annotator: python annotator.py")
    print("  3. Capture/annotate training data")
    print("  4. Train the model: python train_model.py")
    print("  5. Run detection: python detect_dot.py")
    print()

if __name__ == "__main__":
    main()
