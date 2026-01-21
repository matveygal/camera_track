#!/usr/bin/env python3
"""
Quick Start Script for Black Dot Detection System
Provides an interactive menu to guide users through the workflow.
"""

import sys
import subprocess
from pathlib import Path
import time


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n")


def print_menu(title, options):
    """Print a menu with options."""
    print(f"\n{title}")
    print("-" * 40)
    for key, value in options.items():
        print(f"  [{key}] {value}")
    print("-" * 40)


def run_command(cmd, description):
    """Run a command with description."""
    print(f"\n🚀 {description}...")
    print(f"   Command: {cmd}")
    print()
    
    try:
        result = subprocess.run(cmd, shell=True, check=True)
        print(f"\n✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error: {description} failed!")
        print(f"   Exit code: {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        return False


def check_environment():
    """Check if environment is set up."""
    venv_dir = Path("venv_blackdot")
    
    if not venv_dir.exists():
        print("\n⚠️  Virtual environment not found!")
        print("   Please run install_deps.py first")
        return False
    
    return True


def main():
    """Main menu loop."""
    print_header("Black Dot Detection System - Quick Start")
    
    print("Welcome to the Black Dot Detection System!")
    print("This interactive guide will help you get started.\n")
    
    # Check if we're in the right directory
    if not Path("config.yaml").exists():
        print("❌ Error: config.yaml not found!")
        print("   Please run this script from the black_dot_detector directory")
        return
    
    while True:
        print_menu(
            "MAIN MENU",
            {
                "1": "Setup - Install Dependencies",
                "2": "Test Camera Connection",
                "3": "Annotate Images (File Mode)",
                "4": "Annotate Images (Camera Mode)",
                "5": "Train Model",
                "6": "Test Detection (Single Image)",
                "7": "Test Detection (Camera Capture)",
                "8": "Run Live Detection",
                "9": "Batch Process Images",
                "h": "Show Help & Documentation",
                "q": "Quit"
            }
        )
        
        choice = input("\nSelect an option: ").strip().lower()
        
        if choice == 'q':
            print("\n👋 Goodbye!")
            break
        
        elif choice == '1':
            print_header("Setup - Installing Dependencies")
            print("This will create a virtual environment and install all required packages.")
            confirm = input("Continue? (y/n): ").strip().lower()
            
            if confirm == 'y':
                run_command("python3 install_deps.py", "Installing dependencies")
        
        elif choice == '2':
            print_header("Test Camera Connection")
            
            if not check_environment():
                continue
            
            print("This will list available cameras and show a live preview.")
            print("Press 'q' in the camera window to quit.\n")
            
            run_command("venv_blackdot/bin/python camera_utils.py", 
                       "Testing camera")
        
        elif choice == '3':
            print_header("Annotate Images - File Mode")
            
            if not check_environment():
                continue
            
            raw_images_dir = Path("data/raw_images")
            image_files = list(raw_images_dir.glob("*.jpg")) + list(raw_images_dir.glob("*.png"))
            
            if not image_files:
                print("⚠️  No images found in data/raw_images/")
                print("   Please add some images to annotate first.")
                continue
            
            print(f"Found {len(image_files)} images in data/raw_images/")
            print("\nControls:")
            print("  - Click on dot position")
            print("  - Press 's' to save annotation")
            print("  - Press 'n' for next image")
            print("  - Press 'p' for previous image")
            print("  - Press 'q' to quit")
            
            input("\nPress Enter to start...")
            
            run_command("venv_blackdot/bin/python annotator.py --mode file",
                       "Running annotator")
        
        elif choice == '4':
            print_header("Annotate Images - Camera Mode")
            
            if not check_environment():
                continue
            
            print("This will capture images from the camera and annotate them.")
            print("\nControls:")
            print("  - Press 'c' to capture image")
            print("  - Click on dot position")
            print("  - Press 's' to save annotation")
            print("  - Press 'q' to quit")
            
            input("\nPress Enter to start...")
            
            run_command("venv_blackdot/bin/python annotator.py --mode camera",
                       "Running camera annotator")
        
        elif choice == '5':
            print_header("Train Model")
            
            if not check_environment():
                continue
            
            # Check for annotations
            annotations_file = Path("data/annotations.json")
            if not annotations_file.exists():
                print("⚠️  No annotations found!")
                print("   Please annotate some images first (option 3 or 4)")
                continue
            
            print("This will:")
            print("  1. Prepare the dataset with train/validation split")
            print("  2. Apply data augmentation")
            print("  3. Train YOLOv8 model")
            print("  4. Save the best model")
            print("\nNote: Training may take 10-60 minutes depending on your hardware.")
            
            confirm = input("\nContinue? (y/n): ").strip().lower()
            
            if confirm == 'y':
                run_command("venv_blackdot/bin/python train_model.py",
                           "Training model")
        
        elif choice == '6':
            print_header("Test Detection - Single Image")
            
            if not check_environment():
                continue
            
            # Check for model
            model_path = Path("models/trained_models/best_model.pt")
            if not model_path.exists():
                print("⚠️  No trained model found!")
                print("   Please train a model first (option 5)")
                continue
            
            image_path = input("Enter path to image file: ").strip()
            
            if not Path(image_path).exists():
                print(f"❌ Image not found: {image_path}")
                continue
            
            run_command(f"venv_blackdot/bin/python detect_dot.py image --input '{image_path}'",
                       "Running detection")
        
        elif choice == '7':
            print_header("Test Detection - Camera Capture")
            
            if not check_environment():
                continue
            
            # Check for model
            model_path = Path("models/trained_models/best_model.pt")
            if not model_path.exists():
                print("⚠️  No trained model found!")
                print("   Please train a model first (option 5)")
                continue
            
            print("This will capture a single frame and detect the dot.")
            input("\nPress Enter to start...")
            
            run_command("venv_blackdot/bin/python detect_dot.py camera",
                       "Running camera detection")
        
        elif choice == '8':
            print_header("Run Live Detection")
            
            if not check_environment():
                continue
            
            # Check for model
            model_path = Path("models/trained_models/best_model.pt")
            if not model_path.exists():
                print("⚠️  No trained model found!")
                print("   Please train a model first (option 5)")
                continue
            
            print("This will run real-time detection on the camera feed.")
            print("\nControls:")
            print("  - Press 's' to save current detection")
            print("  - Press 'q' to quit")
            
            input("\nPress Enter to start...")
            
            run_command("venv_blackdot/bin/python detect_dot.py live",
                       "Running live detection")
        
        elif choice == '9':
            print_header("Batch Process Images")
            
            if not check_environment():
                continue
            
            # Check for model
            model_path = Path("models/trained_models/best_model.pt")
            if not model_path.exists():
                print("⚠️  No trained model found!")
                print("   Please train a model first (option 5)")
                continue
            
            images_dir = input("Enter path to images directory: ").strip()
            
            if not Path(images_dir).exists():
                print(f"❌ Directory not found: {images_dir}")
                continue
            
            run_command(f"venv_blackdot/bin/python detect_dot.py batch --input '{images_dir}'",
                       "Running batch detection")
        
        elif choice == 'h':
            print_header("Help & Documentation")
            
            print("""
WORKFLOW OVERVIEW:
==================

1. Setup (Option 1)
   - Install all dependencies in a virtual environment
   - Only needs to be done once

2. Annotate Data (Options 3 or 4)
   - Collect 20-50+ images with black dots
   - Mark the dot position on each image
   - More data = better model

3. Train Model (Option 5)
   - Trains AI model on your annotated data
   - Takes 10-60 minutes depending on hardware
   - Creates best_model.pt in models/trained_models/

4. Run Detection (Options 6, 7, 8, or 9)
   - Use the trained model to detect dots
   - Multiple modes available for different use cases

TIPS:
=====

- Collect diverse training data (different angles, lighting)
- Annotate at least 30-50 images for good results
- Test camera connection before starting
- Monitor training progress - stop if loss plateaus
- Adjust confidence threshold in config.yaml if needed

FILES:
======

- config.yaml: Main configuration file
- data/annotations.json: Stores annotations
- models/trained_models/best_model.pt: Trained model
- results/: Detection results and logs

For detailed documentation, see README.md
            """)
            
            input("\nPress Enter to continue...")
        
        else:
            print("\n❌ Invalid option! Please try again.")
        
        time.sleep(0.5)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Goodbye!")
        sys.exit(0)
