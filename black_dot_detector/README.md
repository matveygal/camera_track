# Black Dot Detection System

A complete computer vision system for detecting black dots on rubber heart mock-ups using Intel RealSense D435i camera and deep learning.

## 🎯 Overview

This system provides three main components:
1. **Annotation Tool** - GUI for marking dots on images with camera capture
2. **Training Pipeline** - Train YOLOv8 models on annotated data
3. **Detection Application** - Real-time and batch detection modes

## 📋 Requirements

- Python 3.8+
- Intel RealSense D435i camera
- CUDA-capable GPU (optional, for faster training)
- 4GB+ RAM
- ~2GB disk space for dependencies

## 🚀 Quick Start

### 1. Installation

Run the installation script to set up a fresh virtual environment with all dependencies:

```bash
# Using Python script (cross-platform)
python3 install_deps.py

# Or using bash script (Linux/macOS)
chmod +x install_deps.sh
./install_deps.sh
```

### 2. Activate Virtual Environment

```bash
# Linux/macOS
source venv_blackdot/bin/activate

# Windows
venv_blackdot\Scripts\activate
```

### 3. Test Camera Connection

```bash
python camera_utils.py
```

This will list available cameras and show a live preview.

## 📸 Workflow

### Step 1: Collect and Annotate Data

**Option A: Annotate Existing Images**

1. Place your images in `data/raw_images/`
2. Run the annotator:
   ```bash
   python annotator.py --mode file
   ```
3. Click on the dot position in each image
4. Press `s` to save, `n` for next image

**Option B: Capture from Camera**

1. Run annotator in camera mode:
   ```bash
   python annotator.py --mode camera
   ```
2. Press `c` to capture image
3. Click on the dot position
4. Press `s` to save annotation
5. Repeat for multiple angles/positions

**Keyboard Controls:**
- `c` - Capture image (camera mode)
- `s` - Save annotation
- `n` - Next image
- `p` - Previous image
- `k` - Skip image
- `q` - Quit

### Step 2: Train the Model

After collecting at least 20-30 annotated images:

```bash
python train_model.py
```

This will:
- Prepare the dataset with train/validation split
- Apply data augmentation
- Train YOLOv8 model
- Save best model to `models/trained_models/best_model.pt`
- Generate training metrics

**Training Options:**

```bash
# Prepare dataset only
python train_model.py --prepare-only

# Train only (skip dataset prep)
python train_model.py --train-only

# Evaluate existing model
python train_model.py --evaluate models/trained_models/best_model.pt
```

### Step 3: Run Detection

**Single Image Detection:**
```bash
python detect_dot.py image --input path/to/image.jpg
```

**Batch Processing:**
```bash
python detect_dot.py batch --input path/to/images/directory
```

**Camera Capture + Detection:**
```bash
python detect_dot.py camera
```

**Live Camera Stream:**
```bash
python detect_dot.py live
```

**Additional Options:**
```bash
# Use specific model
python detect_dot.py live --model models/trained_models/custom_model.pt

# No display (save only)
python detect_dot.py batch --input data/raw_images --no-display

# No save (display only)
python detect_dot.py image --input test.jpg --no-save
```

## 📁 Project Structure

```
black_dot_detector/
├── annotator.py              # Annotation tool with camera capture
├── train_model.py            # Training pipeline
├── detect_dot.py             # Detection application
├── camera_utils.py           # RealSense camera interface
├── utils.py                  # General utilities
├── config.yaml               # Configuration file
├── requirements.txt          # Python dependencies
├── install_deps.py           # Installation script (Python)
├── install_deps.sh           # Installation script (Bash)
├── README.md                 # This file
│
├── data/
│   ├── raw_images/          # Place existing images here
│   ├── camera_captures/     # Camera-captured images
│   ├── annotations.json     # Annotation data
│   └── dataset/             # Prepared training dataset
│       ├── images/
│       │   ├── train/
│       │   └── val/
│       └── labels/
│           ├── train/
│           └── val/
│
├── models/
│   └── trained_models/      # Saved models
│       └── best_model.pt
│
└── results/
    ├── training_metrics/    # Training logs and metrics
    ├── detections/          # Detection results
    ├── feedback.json        # Detection feedback log
    └── system.log           # System logs
```

## ⚙️ Configuration

Edit `config.yaml` to customize:

### Camera Settings
```yaml
camera:
  rgb_resolution: [1280, 720]  # Camera resolution
  framerate: 30                 # FPS
  enable_depth: true            # Enable depth stream
  auto_exposure: true           # Auto-exposure
```

### Training Settings
```yaml
training:
  model_architecture: "yolov8"  # Model type
  model_size: "n"               # n, s, m, l, x
  batch_size: 16                # Batch size
  epochs: 100                   # Training epochs
  learning_rate: 0.001          # Learning rate
  train_val_split: 0.8          # Train/val split ratio
```

### Detection Settings
```yaml
detection:
  confidence_threshold: 0.5     # Min confidence
  max_detections: 1             # Max dots per image
  use_depth_validation: false   # Use depth filtering
```

### Data Augmentation
```yaml
augmentation:
  enabled: true
  rotation_range: 15            # Rotation degrees
  brightness_range: [0.7, 1.3]  # Brightness adjustment
  horizontal_flip: 0.5          # Flip probability
```

## 🔧 Advanced Usage

### Custom Model Training

```python
from train_model import DotDetectorTrainer

config = load_config('config.yaml')
trainer = DotDetectorTrainer(config)
trainer.train_yolo()
```

### Programmatic Detection

```python
from detect_dot import DotDetector

config = load_config('config.yaml')
detector = DotDetector(config)

# Single image
result = detector.detect_single_image('image.jpg')
print(f"Dot at: ({result['x']}, {result['y']})")

# Batch processing
results = detector.detect_batch('images_directory')
```

### Camera Integration

```python
from camera_utils import RealSenseCamera

# Initialize camera
camera = RealSenseCamera(
    rgb_resolution=(1280, 720),
    framerate=30
)

if camera.initialize():
    # Capture single frame
    frame_data = camera.capture_frame()
    color_image = frame_data['color']
    
    # Stream with callback
    def process_frame(frame_data):
        # Process frame here
        return True  # Continue streaming
    
    camera.stream_frames(process_frame)
    camera.cleanup()
```

## 📊 Performance Tips

1. **Training:**
   - Collect 50+ images from various angles for best results
   - Use data augmentation to increase dataset size
   - Train for at least 50-100 epochs
   - Monitor validation loss to prevent overfitting

2. **Detection:**
   - Adjust `confidence_threshold` based on your needs (0.3-0.7 typical range)
   - Lower resolution = faster inference
   - Use GPU for real-time detection

3. **Camera:**
   - Ensure good lighting conditions
   - Keep camera stable during capture
   - Use auto-exposure for varying lighting

## 🐛 Troubleshooting

### Camera Not Detected

```bash
# List available cameras
python camera_utils.py

# Check RealSense drivers
rs-enumerate-devices
```

### Training Issues

- **Out of Memory:** Reduce `batch_size` in config.yaml
- **Slow Training:** Enable GPU or reduce `image_size`
- **Poor Accuracy:** Collect more diverse training data

### Detection Issues

- **False Positives:** Increase `confidence_threshold`
- **Missed Detections:** Lower `confidence_threshold`, retrain with more data
- **Slow Detection:** Reduce image resolution or use smaller model (yolov8n)

### Import Errors

```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Check installations
python -c "import cv2, torch, pyrealsense2; print('All imports OK')"
```

## 📈 Model Performance

Expected performance with 50+ training images:
- **Precision:** 85-95%
- **Recall:** 80-90%
- **Inference Speed:** 
  - CPU: ~50-100ms per image
  - GPU: ~10-20ms per image
  - Live feed: 15-30 FPS

## 🔬 Technical Details

### Detection Method
- Uses YOLOv8 object detection with keypoint/bounding box approach
- Trains on small bounding boxes around dot positions
- Extracts center point from detected boxes

### Data Augmentation
- Rotation, brightness/contrast adjustment
- Horizontal/vertical flips
- Gaussian noise, blur
- Maintains keypoint annotation through transforms

### Camera Integration
- Direct pyrealsense2 integration
- Supports RGB + optional depth streams
- Configurable resolution and framerate
- Automatic exposure and error recovery

## 🤝 Contributing

To extend the system:

1. **Add New Model Architecture:**
   - Modify `train_model.py`
   - Add configuration in `config.yaml`

2. **Custom Augmentation:**
   - Edit `_apply_augmentation()` in `train_model.py`

3. **Additional Detection Modes:**
   - Extend `DotDetector` class in `detect_dot.py`

## 📝 License

This project is provided as-is for educational and research purposes.

## 🙏 Acknowledgments

- YOLOv8 by Ultralytics
- Intel RealSense SDK
- OpenCV community
- PyTorch team

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review configuration settings
3. Test camera connection
4. Verify training data quality

## 🎓 Tips for Best Results

1. **Data Collection:**
   - Capture from multiple angles (top, side, angled)
   - Include different lighting conditions
   - Vary distance from camera
   - Include some difficult cases (glare, shadows)

2. **Annotation:**
   - Be consistent with dot center marking
   - Double-check annotations before training
   - Skip blurry or unclear images

3. **Training:**
   - Start with small model (yolov8n) for testing
   - Gradually increase to larger models if needed
   - Monitor validation metrics during training
   - Save checkpoints frequently

4. **Deployment:**
   - Test thoroughly before production use
   - Maintain consistent lighting in deployment environment
   - Collect feedback for continuous improvement
   - Retrain periodically with new data

## 📚 References

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Intel RealSense SDK](https://github.com/IntelRealSense/librealsense)
- [OpenCV Documentation](https://docs.opencv.org/)
- [PyTorch Documentation](https://pytorch.org/docs/)

---

**Version:** 1.0  
**Last Updated:** January 2026  
**Status:** Production Ready
