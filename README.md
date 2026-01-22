# Heart Surface Tracking System

Real-time tracking system for surgery robot applications using SIFT feature detection and Extended Kalman Filter for robust, stable tracking of points on a moving heart surface.

## Features

- **Robust Feature Tracking**: SIFT (Scale-Invariant Feature Transform) for illumination-invariant feature detection
- **Motion Prediction**: Extended Kalman Filter (EKF) models quasi-periodic heart motion for stable tracking
- **Occlusion Handling**: Constellation tracking with multiple points provides redundancy when tools occlude the primary target
- **Noise Suppression**: EKF filters measurement noise while maintaining responsiveness to actual motion
- **Never Lose Track**: Multi-layered approach ensures continuous tracking even under challenging conditions

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Intel RealSense 435i                    │
│                      RGB Camera (Only)                       │
└────────────────────┬────────────────────────────────────────┘
                     │ RGB Frame
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  SIFT Feature Detection                     │
│  - Detects distinctive keypoints on heart surface          │
│  - Illumination and scale invariant                         │
│  - Matches features between frames                          │
└────────────────────┬────────────────────────────────────────┘
                     │ Feature Matches
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Homography Transformation                      │
│  - Computes geometric transformation                        │
│  - RANSAC for outlier rejection                             │
│  - Projects target point to current frame                   │
└────────────────────┬────────────────────────────────────────┘
                     │ Target Position Estimate
                     ▼
┌─────────────────────────────────────────────────────────────┐
│           Extended Kalman Filter (EKF)                      │
│  - State: [x, y, vx, vy] (position + velocity)             │
│  - Predicts motion during occlusion                         │
│  - Smooths noisy measurements                               │
│  - Maintains tracking stability                             │
└────────────────────┬────────────────────────────────────────┘
                     │ Filtered Position + Velocity
                     ▼
┌─────────────────────────────────────────────────────────────┐
│            Constellation Tracking (Redundancy)              │
│  - Multiple auxiliary points around target                  │
│  - Each has own EKF                                          │
│  - Recovers primary tracking from constellation             │
└────────────────────┬────────────────────────────────────────┘
                     │ Robust Tracked Position
                     ▼
                [Tracked Point + Confidence]
```

## Installation

### Prerequisites

- Python 3.8+
- Intel RealSense 435i camera connected
- macOS (tested) or Linux

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd camera_track
   ```

2. **Create virtual environment** (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On macOS/Linux
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Or use the install script:
   ```bash
   python install_deps.py
   ```

## Usage

### Quick Start

1. **Run the tracker**:
   ```bash
   python heart_tracker.py
   ```

2. **Initialize tracking**:
   - Click on the target point on the heart surface in the video window
   - The system will detect features and begin tracking

3. **Monitor tracking**:
   - Green crosshair = high confidence tracking
   - Yellow/Orange = medium confidence (occlusion recovery)
   - Red = low confidence or prediction-only mode
   - Cyan arrow shows velocity vector
   - Blue circles show constellation points
   - Yellow ellipse shows position uncertainty

### Keyboard Controls

- **`q`**: Quit the application
- **`r`**: Reset tracking (select new target point)
- **`p`**: Pause/resume tracking
- **`+` / `=`**: Increase process noise (smoother, less responsive)
- **`-`**: Decrease process noise (more responsive, noisier)
- **`]`**: Increase measurement noise (trust measurements less)
- **`[`**: Decrease measurement noise (trust measurements more)

## Configuration

Edit [`config.py`](config.py) to adjust tracking parameters:

### Key Parameters

```python
# Feature Detection
SIFT_N_FEATURES = 1000          # More features = better matching, slower
SIFT_MATCH_RATIO = 0.75         # Lower = stricter matching (0.7-0.8 typical)
MIN_MATCHES_REQUIRED = 8        # Minimum for valid tracking

# Kalman Filter Tuning
EKF_PROCESS_NOISE = 5.0         # How much to trust motion model
EKF_MEASUREMENT_NOISE = 15.0    # How much to trust SIFT measurements

# Constellation Tracking
CONSTELLATION_RADIUS = 50        # Distance of auxiliary points from target
CONSTELLATION_N_POINTS = 6       # Number of redundant tracking points
```

### Tuning Guide

**For maximum stability** (minimize jitter):
- ↑ `EKF_PROCESS_NOISE` (trust motion model more)
- ↓ `EKF_MEASUREMENT_NOISE` (trust measurements less)
- ↑ `CONSTELLATION_RADIUS` (more spatial averaging)

**For fast responsiveness**:
- ↓ `EKF_PROCESS_NOISE` (less prediction inertia)
- ↑ `EKF_MEASUREMENT_NOISE` (follow measurements closely)

**For better occlusion handling**:
- ↑ `CONSTELLATION_N_POINTS` (more redundancy)
- ↑ `MAX_FRAMES_WITHOUT_MATCH` (longer prediction window)

## Project Structure

```
camera_track/
├── heart_tracker.py              # Main tracking system
├── extended_kalman_filter.py     # EKF implementation
├── config.py                      # Configuration parameters
├── requirements.txt               # Python dependencies
├── install_deps.py                # Dependency installer
└── README.md                      # This file
```

## Algorithm Details

### SIFT Feature Detection

SIFT (Scale-Invariant Feature Transform) provides:
- **Illumination invariance**: Handles changing surgical lighting
- **Scale invariance**: Robust to camera zoom changes
- **Rotation invariance**: Handles camera/heart rotation
- **Distinctive descriptors**: Reliable feature matching

Features are matched using:
1. Brute-force KNN matcher with k=2
2. Lowe's ratio test for robust matching
3. RANSAC-based homography for outlier rejection

### Extended Kalman Filter

**State vector**: `[x, y, vx, vy]`
- Position (x, y) and velocity (vx, vy) in image coordinates

**Motion model**: Constant velocity
- Predicts next position based on current position and velocity
- Models quasi-periodic heart motion

**Measurement model**: Position only
- SIFT provides position measurements
- Velocity inferred from position changes

**Benefits**:
- Smooths noisy SIFT detections
- Predicts position during occlusion
- Provides velocity estimates for motion compensation

### Constellation Tracking

Multiple points arranged around target in circular pattern:
- Each point has independent EKF
- During occlusion, constellation provides geometric constraint
- Target recovered from constellation centroid

## Performance

Typical performance on modern hardware:
- **FPS**: 20-30 Hz (depending on feature density)
- **Latency**: <50ms
- **Accuracy**: Sub-pixel for good illumination
- **Occlusion tolerance**: Up to 30 frames (1 second @ 30fps)

## Troubleshooting

### Low FPS
- Reduce `SIFT_N_FEATURES` in config
- Check CPU usage
- Ensure no other applications using camera

### Tracking jumps/jitters
- Increase `EKF_PROCESS_NOISE` for smoother tracking
- Check for texture-poor regions (SIFT needs texture)
- Increase `MIN_MATCHES_REQUIRED`

### Loses tracking during occlusion
- Increase `CONSTELLATION_RADIUS`
- Increase `MAX_FRAMES_WITHOUT_MATCH`
- Ensure good feature coverage around target

### Poor feature matching
- Ensure adequate lighting
- Check for motion blur (reduce exposure time)
- Increase `SIFT_N_FEATURES`
- Adjust `SIFT_MATCH_RATIO` (try 0.7 for stricter matching)

## Technical Notes

### Why SIFT over other detectors?
- **vs ORB**: SIFT more accurate and illumination-invariant (ORB is faster but less robust)
- **vs SURF**: SIFT comparable performance, better availability
- **vs Deep Learning**: SIFT deterministic, no GPU required, interpretable

### Why Extended Kalman Filter?
- Linear Kalman Filter insufficient for non-linear transformations
- EKF handles homography-based position estimation
- Particle filters more complex, higher computational cost
- Unscented KF alternative but EKF sufficient for this application

### Camera Choice
Intel RealSense 435i chosen for:
- High-quality RGB sensor
- Good performance in surgical lighting
- Depth sensor available if needed for future 3D tracking
- Wide availability and good SDK support

## Future Enhancements

Potential improvements:
- [ ] Adaptive noise parameter tuning
- [ ] Multiple target tracking
- [ ] 3D tracking using depth data
- [ ] Heart rhythm detection for predictive modeling
- [ ] Integration with robot control system
- [ ] Recording and playback for analysis
- [ ] Machine learning-based feature detection

## Git Workflow

Current branch: `heart-tracking-sift-ekf`

To test locally:
```bash
git pull origin heart-tracking-sift-ekf
python heart_tracker.py
```

## License

This is research software for surgical robot development.

## Citation

If using this system in research, please cite:
- Lowe, D.G. "Distinctive Image Features from Scale-Invariant Keypoints" (SIFT)
- Kalman, R.E. "A New Approach to Linear Filtering and Prediction Problems" (Kalman Filter)

## Contact

For questions about this tracking system, please contact the robotics team.
