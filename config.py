"""
Configuration file for heart tracking system.
Adjust these parameters to tune tracking performance for your specific setup.
"""

# Camera Configuration
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# SIFT Feature Detection Parameters
SIFT_N_FEATURES = 1000          # Number of SIFT features to detect
SIFT_MATCH_RATIO = 0.75         # Lowe's ratio test threshold (0.7-0.8 typical)
MIN_MATCHES_REQUIRED = 8        # Minimum matches for valid tracking

# Extended Kalman Filter Parameters
# These control the balance between trusting the motion model vs. measurements
EKF_PROCESS_NOISE = 50.0        # Lower = trust motion model more (smoother, less responsive)
EKF_MEASUREMENT_NOISE = 5.0     # Lower = trust SIFT measurements more (noisier, more responsive)

# Constellation Tracking Parameters
CONSTELLATION_RADIUS = 50        # Pixels around target to place constellation points
CONSTELLATION_N_POINTS = 6       # Number of points in constellation

# Occlusion Handling
MAX_FRAMES_WITHOUT_MATCH = 30    # Maximum frames to predict before declaring tracking lost

# Visualization
DISPLAY_UNCERTAINTY = True       # Show uncertainty ellipse
DISPLAY_VELOCITY = True          # Show velocity vector
DISPLAY_CONSTELLATION = True     # Show constellation points
CROSSHAIR_SIZE = 20             # Size of tracking crosshair

# Performance
FPS_BUFFER_SIZE = 30            # Number of frames to average for FPS calculation

# Tuning Guide:
# 
# For STABLE tracking (minimize jitter):
#   - Increase EKF_PROCESS_NOISE (trust motion model more)
#   - Decrease EKF_MEASUREMENT_NOISE (trust measurements less)
#   - Increase CONSTELLATION_RADIUS (more spatial averaging)
#
# For RESPONSIVE tracking (follow fast movements):
#   - Decrease EKF_PROCESS_NOISE (trust motion model less)
#   - Increase EKF_MEASUREMENT_NOISE (trust measurements more)
#   - Increase MIN_MATCHES_REQUIRED (ensure quality matches)
#
# For OCCLUSION handling:
#   - Increase CONSTELLATION_N_POINTS (more redundancy)
#   - Increase MAX_FRAMES_WITHOUT_MATCH (longer prediction window)
#   - Adjust CONSTELLATION_RADIUS based on typical occlusion size
#
# For ILLUMINATION changes:
#   - Increase SIFT_N_FEATURES (more features to match)
#   - Adjust SIFT_MATCH_RATIO (lower = stricter matching)
