"""
Heart Surface Tracking System using Hybrid SIFT + Optical Flow + Kalman Filter

This system tracks a specific point on the heart surface using:
1. SIFT for initial robust feature detection (once)
2. Lucas-Kanade optical flow for smooth frame-to-frame tracking
3. Extended Kalman Filter for temporal smoothing
4. Intel RealSense 435i RGB camera

Architecture:
- Detect SIFT features ONCE at initialization
- Track those specific features using Lucas-Kanade optical flow
- Use Kalman filter to smooth the tracked position
- Re-detect SIFT only when optical flow loses too many features

This eliminates flickering by maintaining feature correspondence across frames.
"""

import cv2
import numpy as np
import pyrealsense2 as rs
from extended_kalman_filter import ExtendedKalmanFilter
import time
from collections import deque


class HeartTracker:
    """
    Hybrid tracking system: SIFT initialization + Optical Flow tracking + Kalman smoothing.
    """
    
    def __init__(self, 
                 process_noise=10.0,    # Moderate trust in motion model for smoothing
                 measurement_noise=20.0, # Lower trust in individual measurements
                 sift_features=500,
                 min_tracked_features=15,
                 constellation_radius=150,
                 optical_flow_redetect_threshold=10):
        """
        Initialize the hybrid tracker.
        
        Args:
            process_noise: EKF process noise (higher = smoother)
            measurement_noise: EKF measurement noise (higher = less responsive to noise)
            sift_features: Number of SIFT features to detect initially
            min_tracked_features: Minimum features to maintain before re-detection
            constellation_radius: Radius for constellation points
            optical_flow_redetect_threshold: Re-detect SIFT when features drop below this
        """
        # SIFT for initial detection only
        self.sift = cv2.SIFT_create(nfeatures=sift_features)
        self.min_tracked_features = min_tracked_features
        self.optical_flow_redetect_threshold = optical_flow_redetect_threshold
        
        # Lucas-Kanade optical flow parameters
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
        # Tracking state
        self.ekf = None
        self.prev_gray = None
        self.tracked_points = None  # Points we're tracking with optical flow
        self.initial_target = None  # Initial target position
        self.target_feature_idx = None  # Index of the feature closest to target
        self.tracking_initialized = False
        
        # Constellation tracking
        self.constellation_radius = constellation_radius
        self.constellation_initial = []  # Initial constellation positions
        self.constellation_indices = []  # Indices of constellation features
        
        # Performance metrics
        self.fps_buffer = deque(maxlen=30)
        self.last_frame_time = time.time()
        
        # Redetection tracking
        self.frames_since_redetection = 0
        self.redetection_interval = 300  # Force redetection every 300 frames
        
        # Store parameters
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.constellation_radius = constellation_radius
        
    def initialize_tracking(self, frame, target_point):
        """
        Initialize tracking: detect SIFT features once, set up optical flow tracking.
        
        Args:
            frame: Initial RGB frame
            target_point: (x, y) tuple of the target point to track
        """
        self.initial_target = np.array(target_point, dtype=np.float32)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.prev_gray = gray.copy()
        
        # Detect SIFT features ONCE
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        
        if keypoints is None or len(keypoints) < self.min_tracked_features:
            print(f"[ERROR] Not enough features: {len(keypoints) if keypoints else 0}")
            return False
        
        # Convert keypoints to array of points for optical flow
        self.tracked_points = np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 1, 2)
        
        # Find the feature closest to the target point
        distances = np.linalg.norm(self.tracked_points.reshape(-1, 2) - target_point, axis=1)
        self.target_feature_idx = np.argmin(distances)
        
        # Initialize constellation: find features around the target
        self._initialize_constellation(target_point)
        
        # Initialize EKF at target point with smoothing parameters
        self.ekf = ExtendedKalmanFilter(
            initial_position=target_point,
            process_noise=self.process_noise,
            measurement_noise=self.measurement_noise
        )
        
        self.tracking_initialized = True
        self.frames_since_redetection = 0
        
        print(f"[INFO] Tracking initialized")
        print(f"[INFO] Tracking {len(self.tracked_points)} features with optical flow")
        print(f"[INFO] Target feature index: {self.target_feature_idx}")
        print(f"[INFO] Constellation: {len(self.constellation_indices)} points")
        
        return True
    
    def _initialize_constellation(self, target_point):
        """
        Find features in a ring around the target for constellation tracking.
        """
        points = self.tracked_points.reshape(-1, 2)
        distances = np.linalg.norm(points - target_point, axis=1)
        
        # Find features within the constellation radius
        in_radius = (distances < self.constellation_radius) & (distances > 30)
        self.constellation_indices = np.where(in_radius)[0].tolist()
        
        # Keep closest 6-8 constellation points
        if len(self.constellation_indices) > 8:
            constellation_distances = distances[self.constellation_indices]
            sorted_indices = np.argsort(constellation_distances)[:8]
            self.constellation_indices = [self.constellation_indices[i] for i in sorted_indices]
        
        # Store initial constellation positions
        self.constellation_initial = points[self.constellation_indices].copy()
    
    def _redetect_features(self, frame):
        """
        Re-detect SIFT features when optical flow loses too many tracks.
        Maintains correspondence with original target.
        """
        print("[INFO] Re-detecting SIFT features...")
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        
        if keypoints is None or len(keypoints) < self.min_tracked_features:
            print("[WARNING] Re-detection failed, continuing with prediction")
            return False
        
        # Reset tracked points
        self.tracked_points = np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 1, 2)
        self.prev_gray = gray.copy()
        
        # Find new target feature (closest to current EKF estimate)
        current_estimate = self.ekf.get_position()
        distances = np.linalg.norm(self.tracked_points.reshape(-1, 2) - current_estimate, axis=1)
        self.target_feature_idx = np.argmin(distances)
        
        # Re-initialize constellation
        self._initialize_constellation(current_estimate)
        
        self.frames_since_redetection = 0
        print(f"[INFO] Re-detected {len(self.tracked_points)} features")
        
        return True
    
    
    def track(self, frame):
        """
        Track using Lucas-Kanade optical flow + Kalman filter smoothing.
        
        Args:
            frame: Current RGB frame
            
        Returns:
            tracked_position: (x, y) tuple of smoothed tracked point
            confidence: Tracking confidence (0.0 to 1.0)
            status: Tracking status string
        """
        if not self.tracking_initialized:
            return None, 0.0, "Not initialized"
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Track features using Lucas-Kanade optical flow
        new_points, status, error = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.tracked_points, None, **self.lk_params
        )
        
        # Select good points (successfully tracked)
        if new_points is not None and status is not None:
            good_new = new_points[status.flatten() == 1]
            good_old = self.tracked_points[status.flatten() == 1]
            status_bool = status.flatten() == 1
        else:
            good_new = np.array([])
            good_old = np.array([])
            status_bool = np.array([])
        
        num_tracked = len(good_new)
        self.frames_since_redetection += 1
        
        # Check if we need to re-detect features
        if (num_tracked < self.optical_flow_redetect_threshold or 
            self.frames_since_redetection > self.redetection_interval):
            
            if self._redetect_features(frame):
                # Successfully re-detected, continue with prediction for this frame
                tracked_position = self.ekf.predict()
                confidence = 0.5
                status_str = "Re-detecting features"
            else:
                # Re-detection failed, use pure prediction
                tracked_position = self.ekf.predict()
                confidence = 0.3
                status_str = "Prediction only (re-detection failed)"
        
        elif num_tracked >= self.min_tracked_features:
            # Good tracking: update tracked points and get target position
            self.tracked_points = good_new.reshape(-1, 1, 2)
            
            # Update feature indices after filtering
            if self.target_feature_idx < len(status_bool):
                target_still_tracked = status_bool[self.target_feature_idx]
                
                if target_still_tracked:
                    # Calculate new index in filtered array
                    new_target_idx = np.sum(status_bool[:self.target_feature_idx + 1]) - 1
                    target_position = good_new[new_target_idx].flatten()  # Ensure 1D array
                else:
                    # Target feature lost, use constellation centroid
                    target_position = self._estimate_from_constellation(good_new, status_bool)
            else:
                # Target index out of bounds, use constellation
                target_position = self._estimate_from_constellation(good_new, status_bool)
            
            # Update Kalman filter with optical flow measurement
            tracked_position = self.ekf.update(target_position)
            confidence = min(1.0, num_tracked / 50.0)
            status_str = f"Tracking ({num_tracked} features)"
            
            # Update constellation indices
            self._update_constellation_indices(status_bool)
        
        else:
            # Too few features: use EKF prediction
            tracked_position = self.ekf.predict()
            confidence = 0.4
            status_str = f"Predicting ({num_tracked} features remaining)"
        
        # Update previous frame
        self.prev_gray = gray.copy()
        
        # Update FPS
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_frame_time) if (current_time - self.last_frame_time) > 0 else 0
        self.fps_buffer.append(fps)
        self.last_frame_time = current_time
        
        return tracked_position, confidence, status_str
    
    def _estimate_from_constellation(self, good_points, status_bool):
        """
        Estimate target position from constellation when direct tracking fails.
        Returns flattened array [x, y].
        """
        if len(self.constellation_indices) == 0:
            # No constellation, use centroid of all points
            return np.mean(good_points, axis=0).flatten()
        
        # Find which constellation features are still tracked
        tracked_constellation = []
        for idx in self.constellation_indices:
            if idx < len(status_bool) and status_bool[idx]:
                new_idx = np.sum(status_bool[:idx + 1]) - 1
                if new_idx < len(good_points):
                    tracked_constellation.append(new_idx)
        
        if len(tracked_constellation) >= 3:
            # Use constellation centroid
            constellation_points = good_points[tracked_constellation]
            return np.mean(constellation_points, axis=0).flatten()
        else:
            # Fall back to all points centroid
            return np.mean(good_points, axis=0).flatten()
    
    def _update_constellation_indices(self, status_bool):
        """
        Update constellation feature indices after optical flow filtering.
        """
        new_constellation = []
        for idx in self.constellation_indices:
            if idx < len(status_bool) and status_bool[idx]:
                # Calculate new index in filtered array
                new_idx = np.sum(status_bool[:idx + 1]) - 1
                new_constellation.append(new_idx)
        self.constellation_indices = new_constellation
    
    def get_fps(self):
        """Get average FPS over recent frames."""
        return np.mean(self.fps_buffer) if len(self.fps_buffer) > 0 else 0.0
    
    def get_velocity(self):
        """Get current estimated velocity from EKF."""
        if self.ekf is None:
            return np.array([0.0, 0.0])
        return self.ekf.get_velocity()
    
    
    def visualize(self, frame, tracked_position, confidence, status):
        """
        Draw tracking visualization on frame.
        
        Args:
            frame: Frame to draw on
            tracked_position: Current tracked position
            confidence: Tracking confidence
            status: Status string
            
        Returns:
            annotated_frame: Frame with visualization overlays
        """
        vis_frame = frame.copy()
        
        if tracked_position is None:
            return vis_frame
        
        x, y = int(tracked_position[0]), int(tracked_position[1])
        
        # Color based on confidence (red = low, green = high)
        color = (
            int(128 * (1 - confidence)),   # Blue
            int(255 * confidence),          # Green
            int(255 * (1 - confidence))    # Red
        )
        
        # Draw tracked features (optical flow points)
        if self.tracked_points is not None:
            for point in self.tracked_points.reshape(-1, 2):
                cv2.circle(vis_frame, tuple(point.astype(int)), 2, (255, 200, 0), -1)
        
        # Highlight constellation points
        if self.constellation_indices:
            for idx in self.constellation_indices:
                if idx < len(self.tracked_points):
                    pt = self.tracked_points[idx][0]
                    cv2.circle(vis_frame, tuple(pt.astype(int)), 5, (255, 128, 0), 2)
        
        # Draw crosshair at tracked position
        size = 20
        cv2.line(vis_frame, (x - size, y), (x + size, y), color, 3)
        cv2.line(vis_frame, (x, y - size), (x, y + size), color, 3)
        cv2.circle(vis_frame, (x, y), 12, color, 3)
        
        # Draw velocity vector
        velocity = self.get_velocity()
        vel_scale = 5
        vel_end = (int(x + velocity[0] * vel_scale), int(y + velocity[1] * vel_scale))
        cv2.arrowedLine(vis_frame, (x, y), vel_end, (0, 255, 255), 2, tipLength=0.3)
        
        # Draw status information
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_offset = 30
        cv2.putText(vis_frame, f"Status: {status}", (10, y_offset), font, 0.6, (255, 255, 255), 2)
        y_offset += 30
        cv2.putText(vis_frame, f"Confidence: {confidence:.2f}", (10, y_offset), font, 0.6, (255, 255, 255), 2)
        y_offset += 30
        cv2.putText(vis_frame, f"Position: ({x}, {y})", (10, y_offset), font, 0.6, (255, 255, 255), 2)
        y_offset += 30
        cv2.putText(vis_frame, f"Velocity: ({velocity[0]:.1f}, {velocity[1]:.1f})", (10, y_offset), font, 0.6, (255, 255, 255), 2)
        y_offset += 30
        cv2.putText(vis_frame, f"FPS: {self.get_fps():.1f}", (10, y_offset), font, 0.6, (255, 255, 255), 2)
        y_offset += 30
        if self.tracked_points is not None:
            cv2.putText(vis_frame, f"Features: {len(self.tracked_points)}", (10, y_offset), font, 0.6, (255, 255, 255), 2)
        
        return vis_frame


class RealSenseCamera:
    """
    Wrapper for Intel RealSense 435i camera (RGB only).
    """
    
    def __init__(self, width=640, height=480, fps=30):
        """
        Initialize RealSense camera.
        
        Args:
            width: Frame width
            height: Frame height
            fps: Frames per second
        """
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Configure RGB stream only (no depth)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        
        self.width = width
        self.height = height
        self.fps = fps
        self.is_running = False
        
    def start(self):
        """Start the camera stream."""
        try:
            self.pipeline.start(self.config)
            self.is_running = True
            print(f"[INFO] RealSense camera started ({self.width}x{self.height} @ {self.fps} FPS)")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to start camera: {e}")
            return False
    
    def get_frame(self):
        """
        Get a frame from the camera.
        
        Returns:
            frame: BGR frame or None if failed
        """
        if not self.is_running:
            return None
        
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)
            color_frame = frames.get_color_frame()
            
            if not color_frame:
                return None
            
            # Convert to numpy array
            frame = np.asanyarray(color_frame.get_data())
            return frame
            
        except Exception as e:
            print(f"[WARNING] Frame capture failed: {e}")
            return None
    
    def stop(self):
        """Stop the camera stream."""
        if self.is_running:
            self.pipeline.stop()
            self.is_running = False
            print("[INFO] RealSense camera stopped")


def main():
    """
    Main function to run the heart tracking system.
    """
    print("=" * 60)
    print("Heart Surface Tracking System - SIFT + EKF")
    print("=" * 60)
    
    # Initialize camera
    camera = RealSenseCamera(width=640, height=480, fps=30)
    if not camera.start():
        print("[ERROR] Cannot start camera. Exiting.")
        return
    
    # Initialize tracker with smoothing parameters
    tracker = HeartTracker(
        process_noise=10.0,              # Moderate smoothing
        measurement_noise=20.0,          # Trust measurements less for stability
        sift_features=500,               # Detect 500 features initially
        min_tracked_features=15,         # Need 15 features minimum
        constellation_radius=150,        # Wide constellation
        optical_flow_redetect_threshold=10  # Re-detect when <10 features
    )
    
    print("\n[INFO] Click on the target point on the heart to start tracking")
    print("[INFO] Press 'q' to quit, 'r' to reset tracking")
    print("[INFO] Cyan dots = tracked features, Orange circles = constellation")
    print("[INFO] Green crosshair = good tracking, Red = low confidence")
    
    # Mouse callback for selecting target point
    target_selected = False
    clicked_point = None
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal clicked_point, target_selected
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_point = (x, y)
            target_selected = True
    
    cv2.namedWindow('Heart Tracker')
    cv2.setMouseCallback('Heart Tracker', mouse_callback)
    
    try:
        while True:
            # Get frame
            frame = camera.get_frame()
            if frame is None:
                print("[WARNING] Failed to capture frame")
                time.sleep(0.1)
                continue
            
            # Initialize tracking if target selected
            if target_selected and not tracker.tracking_initialized:
                if tracker.initialize_tracking(frame, clicked_point):
                    print("[INFO] Tracking started successfully!")
                else:
                    print("[ERROR] Failed to initialize tracking")
                target_selected = False
            
            # Track if initialized
            if tracker.tracking_initialized:
                tracked_pos, confidence, status = tracker.track(frame)
                vis_frame = tracker.visualize(frame, tracked_pos, confidence, status)
            else:
                vis_frame = frame.copy()
                cv2.putText(vis_frame, "Click to select target point", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Display
            cv2.imshow('Heart Tracker', vis_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n[INFO] Quitting...")
                break
            elif key == ord('r'):
                print("\n[INFO] Resetting tracking...")
                tracker.tracking_initialized = False
                target_selected = False
    
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    
    finally:
        # Cleanup
        camera.stop()
        cv2.destroyAllWindows()
        print("[INFO] Cleanup complete")


if __name__ == "__main__":
    main()
