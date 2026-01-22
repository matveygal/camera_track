"""
Configuration file for heart tracking system.
Tuned for SIFT + Optical Flow + Kalman smoothing approach.
"""

# Camera Configuration
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# SIFT Feature Detection Parameters (for initial detection only)
SIFT_N_FEATURES = 500           # Number of SIFT features to detect initially
MIN_TRACKED_FEATURES = 15       # Minimum features to maintain with optical flow
OPTICAL_FLOW_REDETECT = 10      # Re-detect SIFT when features drop below this

# Extended Kalman Filter Parameters
# Tuned for temporal smoothing (reduce flicker)
EKF_PROCESS_NOISE = 10.0        # Moderate smoothing (trust motion model)
EKF_MEASUREMENT_NOISE = 20.0    # Reduce trust in individual measurements for stability

# Constellation Tracking Parameters
CONSTELLATION_RADIUS = 150       # Pixels around target to place constellation points

# Performance
FPS_BUFFER_SIZE = 30            # Number of frames to average for FPS calculation

# Tuning Guide for New System:
#
# For SMOOTHER tracking (less flicker):
#   - Increase EKF_PROCESS_NOISE (more temporal averaging)
#   - Increase EKF_MEASUREMENT_NOISE (trust individual frames less)
#   - Increase MIN_TRACKED_FEATURES (maintain more features)
#
# For MORE RESPONSIVE tracking:
#   - Decrease EKF_PROCESS_NOISE (follow measurements closer)
#   - Decrease EKF_MEASUREMENT_NOISE (trust optical flow more)
#
# If losing features too quickly:
#   - Increase SIFT_N_FEATURES (detect more initially)
#   - Increase OPTICAL_FLOW_REDETECT (re-detect earlier)
#   - Decrease MIN_TRACKED_FEATURES (be more lenient)
#
# If tracking drifts:
#   - Decrease OPTICAL_FLOW_REDETECT (re-detect more often)
#   - Increase SIFT_N_FEATURES (better initial detection)
