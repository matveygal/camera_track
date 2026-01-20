#!/usr/bin/env python3
"""
Install all required dependencies from requirements.txt
"""
import subprocess
import sys
from pathlib import Path

def main():
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    requirements_file = script_dir / "requirements.txt"
    
    if not requirements_file.exists():
        print(f"Error: requirements.txt not found at {requirements_file}")
        sys.exit(1)
    
    print(f"Installing dependencies from {requirements_file}...")
    print("-" * 50)
    
    try:
        # Use pip to install requirements
        subprocess.check_call([
            sys.executable, 
            "-m", 
            "pip", 
            "install", 
            "-r", 
            str(requirements_file)
        ])
        print("-" * 50)
        print("✓ All dependencies installed successfully!")
        
    except subprocess.CalledProcessError as e:
        print("-" * 50)
        print(f"✗ Error installing dependencies: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
