"""
Heart Surface Tracking System using SIFT + Optical Flow + Extended Kalman Filter

ANTI-FLICKER ARCHITECTURE:
- SIFT for initial feature detection (once)
- Lucas-Kanade optical flow for frame-to-frame tracking (smooth, consistent)
- EKF for noise filtering and occlusion prediction
- SIFT re-detection only when optical flow quality degrades

This eliminates flickering caused by inconsistent SIFT matches every frame.
"""

import cv2
import numpy as np
import pyrealsense2 as rs
from extended_kalman_filter import ExtendedKalmanFilter
import time
from collections import deque


class HeartTrackerOpticalFlow:
    """
    Main tracking system using hybrid SIFT + Optical Flow approach.
    """
    
    def __init__(self, 
                 process_noise=100.0,  # More smoothing than before
                 measurement_noise=5.0,  # Trust optical flow measurements
                 min_features=15,
                 constellation_radius=150):
        """
        Initialize the heart tracker with optical flow.
        
        Args:
            process_noise: EKF process noise (motion model trust)
            measurement_noise: EKF measurement noise (optical flow trust)
            min_features: Minimum features for valid tracking
            constellation_radius: Radius for constellation points
        """
        # SIFT detector (used only for initial detection and re-detection)
        self.sift = cv2.SIFT_create(nfeatures=1000)
        self.min_features = min_features
        
        # Lucas-Kanade optical flow parameters
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
        # Tracking state
        self.ekf = None
        self.target_point = None
        self.tracking_initialized = False
        
        # Optical flow tracking
        self.prev_gray = None
        self.tracked_features = None  # Features tracked with optical flow
        self.feature_ages = None
        self.max_feature_age = 60  # Re-detect after 60 frames (2 sec @ 30fps)
        
        # Reference frame for SIFT re-detection
        self.reference_keypoints = None
        self.reference_descriptors = None
        
        # Constellation tracking
        self.constellation_radius = constellation_radius
        self.constellation_points = []
        self.constellation_ekfs = []
        
        # EKF parameters
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        
        # Performance metrics
        self.fps_buffer = deque(maxlen=30)
        self.last_frame_time = time.time()
        self.frames_without_match = 0
        
    def initialize_tracking(self, frame, target_point):
        """
        Initialize tracking by selecting target and detecting SIFT features.
        """
        self.target_point = np.array(target_point, dtype=np.float32)
        
        # Detect SIFT features
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.reference_keypoints, self.reference_descriptors = self.sift.detectAndCompute(gray, None)
        
        if not self.reference_keypoints or len(self.reference_keypoints) < self.min_features:
            print(f"[ERROR] Not enough features: {len(self.reference_keypoints) if self.reference_keypoints else 0}")
            return False
        
        # Select features near target for optical flow tracking
        self._select_tracking_features(target_point)
        
        # Initialize EKF
        self.ekf = ExtendedKalmanFilter(
            initial_position=target_point,
            process_noise=self.process_noise,
            measurement_noise=self.measurement_noise
        )
        
        # Initialize constellation
        self._initialize_constellation(target_point)
        
        # Store frame for optical flow
        self.prev_gray = gray.copy()
        self.tracking_initialized = True
        self.frames_without_match = 0
        
        print(f"[INFO] Tracking initialized")
        print(f"[INFO] SIFT features: {len(self.reference_keypoints)}")
        print(f"[INFO] Optical flow features: {len(self.tracked_features)}")
        print(f"[INFO] Constellation points: {len(self.constellation_points)}")
        
        return True
    
    def _select_tracking_features(self, center_point):
        """Select SIFT features near target for optical flow."""
        tracking_radius = 120
        selected = []
        
        for kp in self.reference_keypoints:
            dist = np.linalg.norm(np.array(kp.pt) - center_point)
            if dist < tracking_radius:
                selected.append(kp.pt)
        
        # If not enough nearby, use closest ones
        if len(selected) < 10:
            distances = [np.linalg.norm(np.array(kp.pt) - center_point) 
                        for kp in self.reference_keypoints]
            closest_idx = np.argsort(distances)[:30]
            selected = [self.reference_keypoints[i].pt for i in closest_idx]
        
        self.tracked_features = np.array(selected, dtype=np.float32).reshape(-1, 1, 2)
        self.feature_ages = np.zeros(len(selected), dtype=np.int32)
    
    def _initialize_constellation(self, center_point):
        """Create constellation of points around target."""
        self.constellation_points = []
        self.constellation_ekfs = []
        
        num_points = 6
        for i in range(num_points):
            angle = 2 * np.pi * i / num_points
            offset_x = self.constellation_radius * np.cos(angle)
            offset_y = self.constellation_radius * np.sin(angle)
            
            point = np.array([
                center_point[0] + offset_x,
                center_point[1] + offset_y
            ], dtype=np.float32)
            
            ekf = ExtendedKalmanFilter(
                initial_position=point,
                process_noise=self.process_noise,
                measurement_noise=self.measurement_noise
            )
            
            self.constellation_points.append(point)
            self.constellation_ekfs.append(ekf)
    
    def track(self, frame):
        """
        Track using optical flow. Falls back to SIFT re-detection if needed.
        """
        if not self.tracking_initialized:
            return None, 0.0, "Not initialized"
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Track features with Lucas-Kanade optical flow
        new_features, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.tracked_features, None, **self.lk_params
        )
        
        # Filter good features
        good_new = new_features[status.flatten() == 1]
        good_old = self.tracked_features[status.flatten() == 1]
        good_ages = self.feature_ages[status.flatten() == 1]
        
        # Check if re-detection needed
        avg_age = np.mean(good_ages) if len(good_ages) > 0 else 0
        needs_redetection = (len(good_new) < self.min_features or avg_age > self.max_feature_age)
        
        if needs_redetection:
            print(f"[INFO] SIFT re-detection (features: {len(good_new)}, age: {avg_age:.0f})")
            return self._track_with_sift(frame, gray)
        
        # Estimate position from optical flow
        estimated_position = self._estimate_from_optical_flow(good_old, good_new)
        
        if estimated_position is None:
            return self._track_with_sift(frame, gray)
        
        # Update tracking state
        self.tracked_features = good_new.reshape(-1, 1, 2)
        self.feature_ages = good_ages + 1
        self.prev_gray = gray.copy()
        
        # Smoothed update with EKF
        tracked_position = self.ekf.update(estimated_position)
        
        quality = len(good_new) / max(20, self.min_features)
        confidence = min(1.0, quality)
        status_msg = f"Optical Flow ({len(good_new)} features)"
        
        self.frames_without_match = 0
        
        # Update constellation
        self._update_constellation(tracked_position)
        
        # Update FPS
        self._update_fps()
        
        return tracked_position, confidence, status_msg
    
    def _estimate_from_optical_flow(self, old_features, new_features):
        """
        Estimate target position from optical flow displacements.
        Uses median displacement for robustness.
        """
        if len(old_features) < 3:
            return None
        
        # Calculate displacements
        displacements = new_features - old_features
        
        # Use median (robust to outliers)
        median_disp = np.median(displacements, axis=0).flatten()
        
        # Apply to target
        estimated_pos = self.target_point + median_disp
        
        # Sanity check
        x, y = float(estimated_pos[0]), float(estimated_pos[1])
        if x < -100 or x > 740 or y < -100 or y > 580:
            return None
        
        # Update reference
        self.target_point = estimated_pos
        
        return estimated_pos
    
    def _track_with_sift(self, frame, gray):
        """
        Fall back to SIFT re-detection.
        """
        # Detect new SIFT features
        current_kp, current_desc = self.sift.detectAndCompute(gray, None)
        
        if not current_desc or not self.reference_descriptors:
            return self.ekf.predict(), 0.3, "SIFT failed - predicting"
        
        # Match features
        matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(
            self.reference_descriptors, current_desc, k=2
        )
        
        # Lowe's ratio test
        good_matches = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < self.min_features:
            return self.ekf.predict(), 0.3, f"Few matches ({len(good_matches)}) - predicting"
        
        # Estimate position with homography
        src_pts = np.float32([self.reference_keypoints[m.queryIdx].pt for m in good_matches])
        dst_pts = np.float32([current_kp[m.trainIdx].pt for m in good_matches])
        
        try:
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
            
            if H is None:
                return self.ekf.predict(), 0.3, "Homography failed - predicting"
            
            # Transform target
            target_h = np.array([[self.target_point[0], self.target_point[1], 1.0]])
            transformed = (H @ target_h.T).T
            estimated_pos = transformed[0, :2] / transformed[0, 2]
            
            # Re-select features for optical flow
            self.reference_keypoints = current_kp
            self.reference_descriptors = current_desc
            self._select_tracking_features(estimated_pos)
            self.prev_gray = gray.copy()
            self.target_point = estimated_pos
            
            # Update EKF
            tracked_pos = self.ekf.update(estimated_pos)
            confidence = min(1.0, len(good_matches) / (self.min_features * 2))
            
            self.frames_without_match = 0
            
            # Update constellation
            self._update_constellation(tracked_pos)
            self._update_fps()
            
            return tracked_pos, confidence, f"SIFT Re-detect ({len(good_matches)} matches)"
            
        except Exception as e:
            print(f"[WARNING] SIFT tracking failed: {e}")
            return self.ekf.predict(), 0.2, "SIFT error - predicting"
    
    def _update_constellation(self, tracked_position):
        """Update constellation to follow target."""
        if tracked_position is None:
            for ekf in self.constellation_ekfs:
                ekf.predict()
            return
        
        # Maintain geometric relationship
        offset = tracked_position - self.target_point
        
        for i, (init_point, ekf) in enumerate(zip(self.constellation_points, self.constellation_ekfs)):
            new_pos = init_point + offset
            ekf.update(new_pos)
    
    def _update_fps(self):
        """Update FPS calculation."""
        current_time = time.time()
        dt = current_time - self.last_frame_time
        if dt > 0:
            self.fps_buffer.append(1.0 / dt)
        self.last_frame_time = current_time
    
    def get_fps(self):
        """Get average FPS."""
        return np.mean(self.fps_buffer) if len(self.fps_buffer) > 0 else 0.0
    
    def get_velocity(self):
        """Get estimated velocity."""
        return self.ekf.get_velocity() if self.ekf else np.array([0.0, 0.0])
    
    def visualize(self, frame, tracked_position, confidence, status):
        """Draw tracking visualization."""
        vis_frame = frame.copy()
        
        if tracked_position is None:
            return vis_frame
        
        x, y = int(tracked_position[0]), int(tracked_position[1])
        
        # Color based on confidence
        color = (
            int(255 * (1 - confidence)),
            int(255 * confidence),
            int(128 * confidence)
        )
        
        # Draw crosshair
        size = 20
        cv2.line(vis_frame, (x - size, y), (x + size, y), color, 2)
        cv2.line(vis_frame, (x, y - size), (x, y + size), color, 2)
        cv2.circle(vis_frame, (x, y), 10, color, 2)
        
        # Draw optical flow features
        if self.tracked_features is not None:
            for feat in self.tracked_features:
                fx, fy = feat[0]
                cv2.circle(vis_frame, (int(fx), int(fy)), 3, (0, 255, 255), -1)
        
        # Draw constellation
        for ekf in self.constellation_ekfs:
            pos = ekf.get_position()
            cv2.circle(vis_frame, (int(pos[0]), int(pos[1])), 5, (255, 128, 0), -1)
        
        # Draw velocity
        velocity = self.get_velocity()
        vel_scale = 5
        vel_end = (int(x + velocity[0] * vel_scale), int(y + velocity[1] * vel_scale))
        cv2.arrowedLine(vis_frame, (x, y), vel_end, (0, 255, 255), 2, tipLength=0.3)
        
        # Draw info
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(vis_frame, f"Status: {status}", (10, 30), font, 0.6, (255, 255, 255), 2)
        cv2.putText(vis_frame, f"Confidence: {confidence:.2f}", (10, 60), font, 0.6, (255, 255, 255), 2)
        cv2.putText(vis_frame, f"Position: ({x}, {y})", (10, 90), font, 0.6, (255, 255, 255), 2)
        cv2.putText(vis_frame, f"FPS: {self.get_fps():.1f}", (10, 120), font, 0.6, (255, 255, 255), 2)
        
        return vis_frame


class RealSenseCamera:
    """Wrapper for Intel RealSense 435i camera (RGB only)."""
    
    def __init__(self, width=640, height=480, fps=30):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.width = width
        self.height = height
        self.fps = fps
        self.is_running = False
        
    def start(self):
        try:
            self.pipeline.start(self.config)
            self.is_running = True
            print(f"[INFO] Camera started ({self.width}x{self.height} @ {self.fps} FPS)")
            return True
        except Exception as e:
            print(f"[ERROR] Camera start failed: {e}")
            return False
    
    def get_frame(self):
        if not self.is_running:
            return None
        
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)
            color_frame = frames.get_color_frame()
            
            if not color_frame:
                return None
            
            return np.asanyarray(color_frame.get_data())
            
        except Exception as e:
            print(f"[WARNING] Frame capture failed: {e}")
            return None
    
    def stop(self):
        if self.is_running:
            self.pipeline.stop()
            self.is_running = False
            print("[INFO] Camera stopped")


def main():
    """Main function."""
    print("=" * 60)
    print("Heart Tracking - Optical Flow (Anti-Flicker)")
    print("=" * 60)
    
    # Initialize camera
    camera = RealSenseCamera(width=640, height=480, fps=30)
    if not camera.start():
        print("[ERROR] Cannot start camera")
        return
    
    # Initialize tracker with smooth parameters
    tracker = HeartTrackerOpticalFlow(
        process_noise=100.0,  # Smooth filtering
        measurement_noise=5.0,  # Trust optical flow
        min_features=15,
        constellation_radius=150
    )
    
    print("\n[INFO] Click on target point to start tracking")
    print("[INFO] Press 'q' to quit, 'r' to reset")
    
    # Mouse callback
    target_selected = False
    clicked_point = None
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal clicked_point, target_selected
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_point = (x, y)
            target_selected = True
    
    cv2.namedWindow('Heart Tracker - Optical Flow')
    cv2.setMouseCallback('Heart Tracker - Optical Flow', mouse_callback)
    
    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                time.sleep(0.1)
                continue
            
            # Initialize if target selected
            if target_selected and not tracker.tracking_initialized:
                if tracker.initialize_tracking(frame, clicked_point):
                    print("[INFO] Tracking started!")
                else:
                    print("[ERROR] Failed to initialize")
                target_selected = False
            
            # Track
            if tracker.tracking_initialized:
                tracked_pos, confidence, status = tracker.track(frame)
                vis_frame = tracker.visualize(frame, tracked_pos, confidence, status)
            else:
                vis_frame = frame.copy()
                cv2.putText(vis_frame, "Click to select target",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow('Heart Tracker - Optical Flow', vis_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                tracker.tracking_initialized = False
                target_selected = False
    
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted")
    
    finally:
        camera.stop()
        cv2.destroyAllWindows()
        print("[INFO] Cleanup complete")


if __name__ == "__main__":
    main()
