"""
Heart Surface Tracking System using SIFT + Extended Kalman Filter

This system tracks a specific point on the heart surface using:
1. SIFT feature detection for robust feature matching under illumination changes
2. Extended Kalman Filter for motion prediction and noise filtering
3. Constellation tracking for occlusion handling
4. Intel RealSense 435i RGB camera

Features:
- Illumination invariance via SIFT
- Occlusion handling through motion prediction
- Noise suppression via Kalman filtering
- Multi-point constellation tracking for redundancy
"""

import cv2
import numpy as np
import pyrealsense2 as rs
from extended_kalman_filter import ExtendedKalmanFilter
import time
from collections import deque


class HeartTracker:
    """
    Main tracking system for heart surface point tracking.
    """
    
    def __init__(self, 
                 process_noise=200.0, 
                 measurement_noise=1.0,
                 sift_features=1000,
                 match_ratio=0.75,
                 min_matches=8,
                 constellation_radius=50):
        """
        Initialize the heart tracker.
        
        Args:
            process_noise: EKF process noise (trust in motion model)
            measurement_noise: EKF measurement noise (trust in SIFT detections)
            sift_features: Number of SIFT features to detect
            match_ratio: Lowe's ratio test threshold for SIFT matching
            match_ratio: Lowe's ratio test threshold for SIFT matching
            min_matches: Minimum number of matches to consider tracking valid
            constellation_radius: Radius around target point for constellation tracking
        """
        # SIFT detector configuration
        self.sift = cv2.SIFT_create(nfeatures=sift_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        self.match_ratio = match_ratio
        self.min_matches = min_matches
        self.constellation_radius = constellation_radius
        
        # Tracking state
        self.ekf = None
        self.reference_frame = None
        self.reference_keypoints = None
        self.reference_descriptors = None
        self.target_point = None
        self.tracking_initialized = False
        
        # Constellation tracking (multiple points around target)
        self.constellation_ekfs = []
        self.constellation_points = []
        
        # Performance metrics
        self.fps_buffer = deque(maxlen=30)
        self.last_frame_time = time.time()
        
        # Occlusion detection
        self.frames_without_match = 0
        self.max_frames_without_match = 30
        
        # Store parameters for EKF initialization
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        
    def initialize_tracking(self, frame, target_point):
        """
        Initialize tracking by selecting a target point and extracting features.
        
        Args:
            frame: Initial RGB frame
            target_point: (x, y) tuple of the target point to track
        """
        self.target_point = np.array(target_point, dtype=np.float32)
        self.reference_frame = frame.copy()
        
        # Detect SIFT features in reference frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.reference_keypoints, self.reference_descriptors = self.sift.detectAndCompute(gray, None)
        
        if self.reference_descriptors is None or len(self.reference_keypoints) < self.min_matches:
            print(f"[WARNING] Not enough features detected: {len(self.reference_keypoints) if self.reference_keypoints else 0}")
            return False
        
        # Initialize main EKF at target point
        # Start with very high trust in measurements during initialization
        self.ekf = ExtendedKalmanFilter(
            initial_position=target_point,
            process_noise=self.process_noise,
            measurement_noise=self.measurement_noise
        )
        
        # Track initialization phase (trust measurements more initially)
        self.initialization_frames = 0
        self.initialization_period = 30  # First 30 frames
        
        # Create constellation of tracking points around target
        self._initialize_constellation(target_point)
        
        self.tracking_initialized = True
        self.frames_without_match = 0
        
        print(f"[INFO] Tracking initialized at point ({target_point[0]:.1f}, {target_point[1]:.1f})")
        print(f"[INFO] Detected {len(self.reference_keypoints)} SIFT features")
        print(f"[INFO] Constellation: {len(self.constellation_points)} auxiliary points")
        
        return True
    
    def _initialize_constellation(self, center_point):
        """
        Initialize constellation of tracking points around the target.
        Provides redundancy for occlusion handling.
        """
        self.constellation_points = []
        self.constellation_ekfs = []
        
        # Create points in a circular pattern around target
        num_constellation_points = 6
        for i in range(num_constellation_points):
            angle = 2 * np.pi * i / num_constellation_points
            offset_x = self.constellation_radius * np.cos(angle)
            offset_y = self.constellation_radius * np.sin(angle)
            
            constellation_point = np.array([
                center_point[0] + offset_x,
                center_point[1] + offset_y
            ], dtype=np.float32)
            
            # Create EKF for this constellation point
            ekf = ExtendedKalmanFilter(
                initial_position=constellation_point,
                process_noise=self.process_noise,
                measurement_noise=self.measurement_noise
            )
            
            self.constellation_points.append(constellation_point)
            self.constellation_ekfs.append(ekf)
    
    def _match_features(self, frame):
        """
        Match SIFT features between reference frame and current frame.
        
        Returns:
            good_matches: List of good matches
            current_keypoints: Keypoints in current frame
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_keypoints, current_descriptors = self.sift.detectAndCompute(gray, None)
        
        if current_descriptors is None:
            return [], current_keypoints
        
        # Match features using KNN
        matches = self.matcher.knnMatch(self.reference_descriptors, current_descriptors, k=2)
        
        # Apply Lowe's ratio test for robust matching
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.match_ratio * n.distance:
                    good_matches.append(m)
        
        return good_matches, current_keypoints
    
    def _estimate_target_position(self, good_matches, current_keypoints):
        """
        Estimate target point position in current frame using matched features.
        Uses homography to transform the target point.
        
        Returns:
            estimated_position: (x, y) or None if estimation failed
        """
        if len(good_matches) < self.min_matches:
            return None
        
        # Extract matched point coordinates
        src_pts = np.float32([self.reference_keypoints[m.queryIdx].pt for m in good_matches])
        dst_pts = np.float32([current_keypoints[m.trainIdx].pt for m in good_matches])
        
        # Compute homography using RANSAC for outlier rejection
        try:
            # Use stricter RANSAC threshold for better outlier rejection
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
            
            if H is None:
                return None
            
            # Check if we have enough inliers
            inlier_count = np.sum(mask)
            if inlier_count < self.min_matches:
                return None
            
            # Transform target point using homography
            target_homogeneous = np.array([[self.target_point[0], self.target_point[1], 1.0]])
            transformed = (H @ target_homogeneous.T).T
            
            # Convert from homogeneous coordinates
            estimated_position = transformed[0, :2] / transformed[0, 2]
            
            # Sanity check: reject if position is way outside frame
            if (estimated_position[0] < -100 or estimated_position[0] > 740 or
                estimated_position[1] < -100 or estimated_position[1] > 580):
                return None
            
            return estimated_position
            
        except Exception as e:
            print(f"[WARNING] Homography estimation failed: {e}")
            return None
    
    def _update_constellation(self, tracked_position):
        """
        Update constellation of tracking points.
        Keep them in fixed geometric relationship to the tracked target.
        Only use their EKFs during occlusion for recovery.
        """
        if tracked_position is None:
            # During occlusion, let constellation EKFs predict
            for ekf in self.constellation_ekfs:
                ekf.predict()
            return
        
        # Update constellation to maintain geometric relationship with tracked position
        # Calculate the offset from initial target to current tracked position
        offset = tracked_position - self.target_point
        
        # Update each constellation point by applying the same offset
        for i, (initial_point, ekf) in enumerate(zip(self.constellation_points, self.constellation_ekfs)):
            # New position maintains the same geometric relationship
            new_position = initial_point + offset
            # Update EKF directly (for velocity estimation and occlusion prediction)
            ekf.update(new_position)
    
    def _recover_from_constellation(self):
        """
        Attempt to recover target position from constellation when primary tracking fails.
        Uses the geometric relationship between constellation and target.
        
        Returns:
            recovered_position: (x, y) or None if recovery failed
        """
        # Get current constellation positions from their EKFs
        constellation_current = [ekf.get_position() for ekf in self.constellation_ekfs]
        
        # Calculate centroid of constellation
        centroid = np.mean(constellation_current, axis=0)
        
        # The target should be at the centroid (constellation is arranged around it)
        return centroid
    
    def track(self, frame):
        """
        Track the target point in a new frame.
        
        Args:
            frame: Current RGB frame
            
        Returns:
            tracked_position: (x, y) tuple of tracked point
            confidence: Tracking confidence (0.0 to 1.0)
            status: Tracking status string
        """
        if not self.tracking_initialized:
            return None, 0.0, "Not initialized"
        
        # Match features
        good_matches, current_keypoints = self._match_features(frame)
        
        # Estimate target position from SIFT matches
        estimated_position = self._estimate_target_position(good_matches, current_keypoints)
        
        # Update constellation (for redundancy)
        self._update_constellation(good_matches, current_keypoints)
        
        # Determine tracking status and update EKF
        if estimated_position is not None and len(good_matches) >= self.min_matches:
            # Adaptive filtering: with many good matches, trust measurement almost completely
            # With fewer matches, use more filtering
            match_quality = len(good_matches) / (self.min_matches * 3)
            
            if len(good_matches) >= self.min_matches * 2:
                # Excellent tracking: use measurement almost directly (95% measurement)
                tracked_position = 0.95 * estimated_position + 0.05 * self.ekf.get_position()
                self.ekf.update(estimated_position)  # Still update filter for velocity estimation
                status = f"Tracking [EXCELLENT] ({len(good_matches)} matches)"
          Determine tracking status and update EKF
        tracked_position = None(estimated_position)
                status = f"Tracking ({len(good_matches)} matches)"
                confidence = min(1.0, match_quality)
            
            self.frames_without_match = 0
            self.initialization_frames += 1
            
        else:
            # No direct measurement: use prediction
            self.frames_without_match += 1
            
            if self.frames_without_match < 5:
                # Short occlusion: trust EKF prediction
                tracked_position = self.ekf.predict()
                confidence = 0.7
                status = "Predicting (short occlusion)"
                
            else:
                # Longer occlusion: try constellation recovery
                recovered_position = self._recover_from_constellation()
                
                if recovered_position is not None:
                    tracked_position = self.ekf.update(recovered_position)
                    confidence = 0.5
                    status = f"Constellation recovery ({self.frames_without_match} frames)"
                else:
                    # Pure prediction
                    tracked_position = self.ekf.predict()
                    confidence = max(0.1, 0.8 - self.frames_without_match * 0.05)
                    status = f"Pure prediction ({self.frames_without_match} frames)"
        
        # Check if tracking is lost
        if self.ekf.is_tracking_lost():
        # Update constellation to follow tracked position
        self._update_constellation(tracked_position)
        
        # Check if tracking is lost
            status = "Tracking lost"
        
        # Update FPS
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_frame_time) if (current_time - self.last_frame_time) > 0 else 0
        self.fps_buffer.append(fps)
        self.last_frame_time = current_time
        
        return tracked_position, confidence, status
    
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
            int(255 * (1 - confidence)),  # Blue
            int(255 * confidence),          # Green
            int(128 * confidence)           # Red
        )
        
        # Draw crosshair at tracked position
        size = 20
        cv2.line(vis_frame, (x - size, y), (x + size, y), color, 2)
        cv2.line(vis_frame, (x, y - size), (x, y + size), color, 2)
        cv2.circle(vis_frame, (x, y), 10, color, 2)
        
        # Draw velocity vector
        velocity = self.get_velocity()
        vel_scale = 5
        vel_end = (int(x + velocity[0] * vel_scale), int(y + velocity[1] * vel_scale))
        cv2.arrowedLine(vis_frame, (x, y), vel_end, (0, 255, 255), 2, tipLength=0.3)
        
        # Draw constellation points
        for ekf in self.constellation_ekfs:
            pos = ekf.get_position()
            cv2.circle(vis_frame, (int(pos[0]), int(pos[1])), 5, (255, 128, 0), -1)
        
        # Draw uncertainty ellipse
        if self.ekf is not None:
            uncertainty = self.ekf.get_uncertainty()
            axes = (int(uncertainty[0] * 2), int(uncertainty[1] * 2))
            cv2.ellipse(vis_frame, (x, y), axes, 0, 0, 360, (255, 255, 0), 1)
        
        # Draw status information
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(vis_frame, f"Status: {status}", (10, 30), font, 0.6, (255, 255, 255), 2)
        cv2.putText(vis_frame, f"Confidence: {confidence:.2f}", (10, 60), font, 0.6, (255, 255, 255), 2)
        cv2.putText(vis_frame, f"Position: ({x}, {y})", (10, 90), font, 0.6, (255, 255, 255), 2)
        cv2.putText(vis_frame, f"Velocity: ({velocity[0]:.1f}, {velocity[1]:.1f})", (10, 120), font, 0.6, (255, 255, 255), 2)
        cv2.putText(vis_frame, f"FPS: {self.get_fps():.1f}", (10, 150), font, 0.6, (255, 255, 255), 2)
        
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
    
    # Initialize tracker with highly responsive parameters
    tracker = HeartTracker(
        process_noise=200.0,   # Very high process noise = instant response
        measurement_noise=1.0, # Very low measurement noise = trust SIFT almost completely
        sift_features=1000,
        match_ratio=0.75,
        min_matches=8,
        constellation_radius=50
    )
    
    print("\n[INFO] Click on the target point on the heart to start tracking")
    print("[INFO] Press 'q' to quit, 'r' to reset tracking, 'p' to pause")
    print("[INFO] Press '+'/'-' to adjust process noise")
    print("[INFO] Press '['/']' to adjust measurement noise")
    
    # Mouse callback for selecting target point
    target_selected = False
    clicked_point = None
    paused = False
    
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
            
            # Track if initialized and not paused
            if tracker.tracking_initialized and not paused:
                tracked_pos, confidence, status = tracker.track(frame)
                vis_frame = tracker.visualize(frame, tracked_pos, confidence, status)
            else:
                vis_frame = frame.copy()
                if not tracker.tracking_initialized:
                    cv2.putText(vis_frame, "Click to select target point", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                elif paused:
                    cv2.putText(vis_frame, "PAUSED - Press 'p' to resume", 
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
            elif key == ord('p'):
                paused = not paused
                print(f"\n[INFO] {'Paused' if paused else 'Resumed'}")
            elif key == ord('+') or key == ord('='):
                tracker.process_noise *= 1.2
                if tracker.ekf:
                    tracker.ekf.adjust_noise_parameters(process_noise=tracker.process_noise)
                print(f"\n[INFO] Process noise: {tracker.process_noise:.2f}")
            elif key == ord('-'):
                tracker.process_noise /= 1.2
                if tracker.ekf:
                    tracker.ekf.adjust_noise_parameters(process_noise=tracker.process_noise)
                print(f"\n[INFO] Process noise: {tracker.process_noise:.2f}")
            elif key == ord('['):
                tracker.measurement_noise /= 1.2
                if tracker.ekf:
                    tracker.ekf.adjust_noise_parameters(measurement_noise=tracker.measurement_noise)
                print(f"\n[INFO] Measurement noise: {tracker.measurement_noise:.2f}")
            elif key == ord(']'):
                tracker.measurement_noise *= 1.2
                if tracker.ekf:
                    tracker.ekf.adjust_noise_parameters(measurement_noise=tracker.measurement_noise)
                print(f"\n[INFO] Measurement noise: {tracker.measurement_noise:.2f}")
    
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    
    finally:
        # Cleanup
        camera.stop()
        cv2.destroyAllWindows()
        print("[INFO] Cleanup complete")


if __name__ == "__main__":
    main()
