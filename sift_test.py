import cv2
import numpy as np
import pyrealsense2 as rs

import cv2
import numpy as np
import pyrealsense2 as rs

def init_realsense_camera():
    """Initialize RealSense camera for RGB streaming only"""
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Enable RGB stream only (depth sensors are broken)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    try:
        profile = pipeline.start(config)
        print("RealSense camera initialized successfully")
        print("RGB stream: 640x480 @ 30fps")
        return pipeline, True
    except Exception as e:
        print(f"Failed to initialize RealSense camera: {e}")
        return None, False

def get_realsense_frame(pipeline):
    """Get a single RGB frame from RealSense camera"""
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        return None
    return np.asanyarray(color_frame.get_data())


def draw_tracked_object(frame, corners, status="Tracking"):
    """Draw the detected object polygon and info"""
    if corners is not None and len(corners) == 4:
        # Draw polygon
        pts = corners.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], True, (0, 255, 0), 3)
        
        # Compute center
        center = np.mean(corners, axis=0).astype(np.int32)
        cv2.circle(frame, tuple(center), 7, (0, 255, 0), -1)
        
        # Compute rotation angle from top edge
        dx = corners[1][0] - corners[0][0]
        dy = corners[1][1] - corners[0][1]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Draw info
        cv2.putText(frame, f"Center: ({center[0]}, {center[1]})", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Angle: {angle:.1f} deg", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, status, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "LOST - searching...", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return frame


def track_object_in_video(video_path, output_path=None):
    """
    Track an object in a video using ORB + homography.
    User selects ROI in first frame.
    
    Args:
        video_path: Path to video file, 0 for webcam, or 'realsense' for RealSense camera
        output_path: Path to save output video (optional)
    """
    # Initialize video source
    use_realsense = False
    pipeline = None
    
    if video_path == 'realsense':
        pipeline, success = init_realsense_camera()
        if not success:
            return
        use_realsense = True
    else:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return
    
    # Show live preview and wait for user to press a key to capture
    print("Live preview starting...")
    print("Position your object, then press SPACE or 's' to capture reference frame")
    print("Press 'q' to quit")
    
    first_frame = None
    while True:
        # Read frame from appropriate source
        if use_realsense:
            frame = get_realsense_frame(pipeline)
            if frame is None:
                print("Error: Cannot read frame from RealSense")
                pipeline.stop()
                return
        else:
            ret, frame = cap.read()
            if not ret:
                print("Error: Cannot read frame")
                return
        
        # Display preview
        preview = frame.copy()
        cv2.putText(preview, "Press SPACE or 's' to capture", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(preview, "Press 'q' to quit", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Live Preview - Position Object", preview)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' ') or key == ord('s'):
            first_frame = frame.copy()
            print("Frame captured!")
            break
        elif key == ord('q'):
            print("Cancelled by user")
            if use_realsense:
                pipeline.stop()
            else:
                cap.release()
            cv2.destroyAllWindows()
            return
    
    cv2.destroyWindow("Live Preview - Position Object")
    
    # Let user select ROI
    print("Select the object to track, then press ENTER or SPACE")
    roi = cv2.selectROI("Select Object", first_frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Object")
    
    x, y, w, h = [int(v) for v in roi]
    if w == 0 or h == 0:
        print("Error: No ROI selected")
        return
    
    # Extract reference image
    ref_img = first_frame[y:y+h, x:x+w]
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    
    # Store reference corners in reference image coordinate system
    ref_corners = np.float32([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ]).reshape(-1, 2)
    # Save initial shape for regularization
    def get_sides_and_angles(corners):
        sides = [np.linalg.norm(corners[i] - corners[(i+1)%4]) for i in range(4)]
        angles = []
        for i in range(4):
            v1 = corners[i] - corners[(i-1)%4]
            v2 = corners[(i+1)%4] - corners[i]
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angles.append(np.arccos(np.clip(cos_angle, -1, 1)))
        return np.array(sides), np.array(angles)
    initial_sides, initial_angles = get_sides_and_angles(ref_corners)
    initial_shape = (initial_sides, initial_angles)
    
    # Initialize feature detector - using SIFT for better rotation/scale invariance
    # Fallback to ORB if SIFT not available
    try:
        detector = cv2.SIFT_create(nfeatures=3000, contrastThreshold=0.03)
        print("Using SIFT detector (best for rotation/scale)")
    except:
        detector = cv2.ORB_create(nfeatures=3000, 
                                   scaleFactor=1.2, 
                                   nlevels=10, 
                                   edgeThreshold=10,
                                   patchSize=31)
        print("Using ORB detector")
    
    # Compute reference keypoints and descriptors
    kp_ref, des_ref = detector.detectAndCompute(ref_gray, None)
    
    if des_ref is None or len(kp_ref) < 4:
        print("Error: Not enough features in reference image. Choose a more textured object.")
        return
    
    print(f"Reference features: {len(kp_ref)}")
    
    # BFMatcher - use L2 for SIFT, Hamming for ORB
    try:
        # SIFT uses L2 norm
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        use_sift = True
    except:
        # ORB uses Hamming distance
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        use_sift = False
    
    # Video writer setup
    if use_realsense:
        fps = 30
        height, width = first_frame.shape[:2]
    else:
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    lost_frames = 0
    last_corners = None
    last_H = None  # Store last homography for smoothing
    corner_history = []  # For temporal smoothing
    
    # Optical flow backup
    prev_gray = None
    flow_points = None
    
    # Template update mechanism (careful to avoid drift)
    frames_since_update = 0
    update_interval = 30  # Update template every N frames if tracking is good
    
    # Velocity-based prediction
    velocity = np.zeros((4, 2))  # Velocity for each corner
    alpha_velocity = 0.3  # Smoothing factor for velocity
    
    # Bundle adjustment: store recent matches for periodic refinement
    match_buffer = []  # Store (src_pts, dst_pts) tuples
    bundle_interval = 30  # Re-estimate homography every N frames
    canonical_H = None  # Refined homography from bundle adjustment
    
    # Kalman Filter for position and rotation tracking
    # State: [x, y, vx, vy, theta, omega] - position, velocity, angle, angular velocity
    kf_state = None  # Will be initialized on first detection
    kf_P = None  # Covariance matrix
    dt = 1.0  # Time step (1 frame)
    
    # Kalman parameters
    process_noise_pos = 0.5  # Process noise for position
    process_noise_vel = 2.0  # Process noise for velocity
    process_noise_angle = 0.01  # Process noise for angle
    process_noise_omega = 0.05  # Process noise for angular velocity
    measurement_noise_base = 5.0  # Base measurement noise (modulated by inlier count)
    
    # Position anchoring for dead zone
    anchor_position = None
    anchor_angle = None
    dead_zone_frames = 0
    first_detection_set = False  # Track if we've set the initial anchor
    is_locked = False  # Track lock state for hysteresis
    lock_threshold = 3  # Frames needed to lock (reduced from 5)
    unlock_threshold = 5  # Frames needed to unlock - sustained movement required
    
    # Track stable position for anchor updates
    stable_position = None
    stable_angle = None
    stability_frames = 0
    stability_threshold = 10  # Frames needed to update anchor to new stable position
    allow_locking = True  # Prevent locking until anchor updates after movement
    
    # Exponential moving average for smoothing
    smoothed_position = None
    smoothed_angle = None
    alpha_smooth = 0.3  # Smoothing factor (lower = more smoothing)
    
    # Occlusion handling
    prev_inliers = 100  # Track previous frame's inlier count
    
    # Reset video to start (only for file-based video)
    if not use_realsense:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    print("Processing video... Press 'q' to quit early")
    
    while True:
        # Read frame from appropriate source
        if use_realsense:
            frame = get_realsense_frame(pipeline)
            if frame is None:
                break
        else:
            ret, frame = cap.read()
            if not ret:
                break
        
        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect features in current frame
        kp, des = detector.detectAndCompute(gray, None)
        
        corners = None
        status = "Tracking"
        
        # Adaptive ratio threshold - more lenient when struggling
        ratio_threshold = 0.85 if lost_frames > 3 else 0.8
        
        if des is not None and len(kp) >= 4:
            # Match descriptors
            matches = matcher.knnMatch(des_ref, des, k=2)
            
            # Apply Lowe's ratio test (adaptive threshold)
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < ratio_threshold * n.distance:
                        # Spatial filtering: if we have a prediction, only accept matches near it
                        if kf_state is not None:
                            match_pt = kp[m.trainIdx].pt
                            pred_center = kf_state[:2]
                            dist_to_pred = np.linalg.norm(np.array(match_pt) - pred_center)
                            # Reject matches more than 150px from predicted center
                            if dist_to_pred < 150:
                                good_matches.append(m)
                        else:
                            good_matches.append(m)
            
            # Need at least 6 matches for more robust homography
            if len(good_matches) >= 6:
                # Extract matched point coordinates
                src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                # Find homography with more lenient RANSAC
                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 
                                             ransacReprojThreshold=8.0,  # More lenient
                                             maxIters=2000,  # More iterations
                                             confidence=0.995)
                
                if H is not None:
                    # Homography sanity check: reject extreme transformations
                    if last_corners is not None and kf_state is not None:
                        # Test transform on reference corners
                        test_corners = cv2.perspectiveTransform(ref_corners.reshape(-1, 1, 2), H).reshape(-1, 2)
                        test_center = np.mean(test_corners, axis=0)
                        pred_center = kf_state[:2]
                        
                        # Check translation jump
                        translation_jump = np.linalg.norm(test_center - pred_center)
                        
                        # Check rotation jump
                        dx = test_corners[1][0] - test_corners[0][0]
                        dy = test_corners[1][1] - test_corners[0][1]
                        test_angle = np.arctan2(dy, dx)
                        pred_angle = kf_state[4]
                        angle_jump = abs(test_angle - pred_angle)
                        if angle_jump > np.pi:
                            angle_jump = 2*np.pi - angle_jump
                        
                        # Reject homography if jump is too large
                        if translation_jump > 100 or angle_jump > np.pi/4:  # 100px or 45° is suspicious
                            H = None
                            status = f"Rejected homography: jump too large ({translation_jump:.1f}px, {np.degrees(angle_jump):.1f}°)"
                    
                    # Count inliers
                    if H is not None:
                        inliers = np.sum(mask)
                    else:
                        inliers = 0
                    
                    # Store inlier matches for bundle adjustment
                    inlier_indices = mask.ravel() == 1
                    inlier_src = src_pts[inlier_indices]
                    inlier_dst = dst_pts[inlier_indices]
                    match_buffer.append((inlier_src, inlier_dst))
                    if len(match_buffer) > bundle_interval:
                        match_buffer.pop(0)
                    
                    # Bundle adjustment: every N frames, pool matches and re-estimate
                    if frame_count % bundle_interval == 0 and len(match_buffer) >= 10:
                        # Pool all inlier matches from buffer
                        all_src = np.vstack([src for src, dst in match_buffer])
                        all_dst = np.vstack([dst for src, dst in match_buffer])
                        
                        # Compute refined homography from pooled matches
                        H_refined, mask_refined = cv2.findHomography(
                            all_src, all_dst, cv2.RANSAC,
                            ransacReprojThreshold=5.0,  # Stricter for pooled data
                            maxIters=3000,
                            confidence=0.999
                        )
                        
                        if H_refined is not None:
                            canonical_H = H_refined
                            status = f"Tracking ({inliers}/{len(good_matches)} inliers) [BUNDLE]"
                    
                    # Use current H for tracking (don't lag behind with canonical)
                    
                    # Require sufficient inliers (at least 30% of matches)
                    min_inliers = max(6, int(len(good_matches) * 0.3))
                    if inliers >= min_inliers:
                        # Transform reference corners to current frame
                        corners = cv2.perspectiveTransform(ref_corners.reshape(-1, 1, 2), H)
                        corners = corners.reshape(-1, 2)
                        
                        # Temporal smoothing with history
                        corner_history.append(corners.copy())
                        if len(corner_history) > 5:
                            corner_history.pop(0)
                        # Weighted moving average for stability
                        if len(corner_history) >= 3:
                            recent = corner_history[-3:]
                            weights = np.array([0.2, 0.3, 0.5])
                            stacked = np.stack(recent, axis=0)
                            corners = np.average(stacked, axis=0, weights=weights)
                        # If inliers are too low, freeze corners
                        if inliers < 8:
                            corners = last_corners if last_corners is not None else corners

                        # Rigid box constraint: only allow rotation and translation, fix size/shape
                        def get_center_and_angle(pts):
                            center = np.mean(pts, axis=0)
                            dx = pts[1][0] - pts[0][0]
                            dy = pts[1][1] - pts[0][1]
                            angle = np.arctan2(dy, dx)
                            return center, angle
                        orig_w = np.linalg.norm(ref_corners[0] - ref_corners[1])
                        orig_h = np.linalg.norm(ref_corners[1] - ref_corners[2])
                        center, angle = get_center_and_angle(corners)
                        R = np.array([
                            [np.cos(angle), -np.sin(angle)],
                            [np.sin(angle),  np.cos(angle)]
                        ])
                        half_w = orig_w / 2
                        half_h = orig_h / 2
                        box = np.array([
                            [-half_w, -half_h],
                            [ half_w, -half_h],
                            [ half_w,  half_h],
                            [-half_w,  half_h]
                        ])
                        
                        # Set initial anchor from very first detection (before Kalman)
                        if not first_detection_set:
                            anchor_position = center.copy()
                            anchor_angle = angle
                            first_detection_set = True
                        
                        # Initialize Kalman Filter on first detection
                        if kf_state is None:
                            kf_state = np.array([center[0], center[1], 0.0, 0.0, angle, 0.0])
                            kf_P = np.eye(6) * 10.0
                        
                        # Occlusion recovery: if coming from low visibility, reset to anchor
                        if prev_inliers < 15 and inliers >= 15:
                            # Just recovered from occlusion - reset Kalman state to anchor
                            kf_state[:2] = anchor_position.copy()
                            kf_state[2:4] = 0.0  # Zero velocity
                            kf_state[4] = anchor_angle
                            kf_state[5] = 0.0  # Zero angular velocity
                            smoothed_position = anchor_position.copy()
                            smoothed_angle = anchor_angle
                            is_locked = True
                            allow_locking = True
                            stability_frames = 0
                            stable_position = None
                            status = f"Tracking ({inliers}/{len(good_matches)} inliers) [RECOVERED - LOCKED TO ANCHOR]"
                        
                        prev_inliers = inliers
                        
                        # Kalman Filter Prediction Step
                        # State transition matrix (constant velocity model)
                        F = np.array([
                            [1, 0, dt, 0, 0, 0],
                            [0, 1, 0, dt, 0, 0],
                            [0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 1, dt],
                            [0, 0, 0, 0, 0, 1]
                        ])
                        
                        # Predict state
                        kf_state_pred = F @ kf_state
                        
                        # Process noise covariance
                        Q = np.diag([
                            process_noise_pos, process_noise_pos,
                            process_noise_vel, process_noise_vel,
                            process_noise_angle, process_noise_omega
                        ])
                        
                        # Predict covariance
                        kf_P_pred = F @ kf_P @ F.T + Q
                        
                        # Measurement: [x, y, theta]
                        z = np.array([center[0], center[1], angle])
                        
                        # Calculate innovation (measurement - prediction)
                        pred_center = kf_state_pred[:2]
                        pred_angle = kf_state_pred[4]
                        innovation_pos = np.linalg.norm(z[:2] - pred_center)
                        innovation_angle = abs(z[2] - pred_angle)
                        if innovation_angle > np.pi:
                            innovation_angle = 2*np.pi - innovation_angle
                        
                        # Force lock during low visibility (occlusion)
                        if inliers < 15:
                            # Completely lock to anchor during occlusion
                            kf_state = kf_state_pred.copy()
                            kf_state[:2] = anchor_position
                            kf_state[2:4] = 0.0
                            kf_state[4] = anchor_angle
                            kf_state[5] = 0.0
                            kf_P = kf_P_pred
                            center = anchor_position
                            angle = anchor_angle
                            is_locked = True
                            status = f"Tracking ({inliers}/{len(good_matches)} inliers) [OCCLUSION - LOCKED]"
                        else:
                            # Dead zone: skip Kalman update for sub-pixel noise
                            dead_zone_pos = 1.5  # pixels - tolerate natural matching bias
                            dead_zone_angle = 0.02  # radians
                            
                            # Tighter threshold for locking
                            lock_pos = 1.0  # pixels
                            lock_angle = 0.015  # radians
                            
                            # Different thresholds depending on current state (hysteresis)
                            if is_locked:
                                # Use looser threshold to stay locked
                                check_pos = dead_zone_pos
                                check_angle = dead_zone_angle
                            else:
                                # Use tighter threshold to lock
                                check_pos = lock_pos
                                check_angle = lock_angle
                            
                            if innovation_pos < check_pos and innovation_angle < check_angle:
                                # Innovation too small - increment counter
                                dead_zone_frames += 1
                                
                                # Require multiple frames to lock (and only if locking is allowed)
                                if dead_zone_frames >= lock_threshold and not is_locked and allow_locking:
                                    is_locked = True
                                
                                # ALWAYS lock to anchor when in dead zone (prevent bias accumulation)
                                # Don't do Kalman updates with biased measurements
                                kf_state = kf_state_pred.copy()
                                kf_state[:2] = anchor_position
                                kf_state[2:4] = 0.0
                                kf_state[4] = anchor_angle
                                kf_state[5] = 0.0
                                kf_P = kf_P_pred
                                center = anchor_position
                                angle = anchor_angle
                                
                                if is_locked:
                                    status = f"Tracking ({inliers}/{len(good_matches)} inliers) [LOCKED]"
                                else:
                                    status = f"Tracking ({inliers}/{len(good_matches)} inliers) [DEAD ZONE {dead_zone_frames}/{lock_threshold}]"
                            else:
                                # Innovation above threshold
                                if is_locked:
                                    # Require sustained movement before unlocking
                                    dead_zone_frames -= 1
                                    if dead_zone_frames <= -unlock_threshold:  # Negative = consecutive movement frames
                                        # Sustained movement detected, unlock
                                        is_locked = False
                                        allow_locking = False  # Prevent re-locking to old anchor
                                        dead_zone_frames = 0
                                        status = f"Tracking ({inliers}/{len(good_matches)} inliers) [MOVEMENT DETECTED]"
                                    else:
                                        # Still locked, ignore transient movement
                                        kf_state = kf_state_pred.copy()
                                        kf_state[:2] = anchor_position
                                        kf_state[2:4] = 0.0
                                        kf_state[4] = anchor_angle
                                        kf_state[5] = 0.0
                                        kf_P = kf_P_pred
                                        center = anchor_position
                                        angle = anchor_angle
                                        status = f"Tracking ({inliers}/{len(good_matches)} inliers) [LOCKED - MOTION {-dead_zone_frames}/{unlock_threshold}]"
                                
                                if not is_locked:
                                    # Real movement detected - reset counter to zero
                                    if dead_zone_frames > 0:
                                        dead_zone_frames = 0
                                    
                                    # Track stable position for anchor updates
                                    # If position is stable at new location, update anchor
                                    # But only during good visibility (high inliers)
                                    if inliers < 15:  # Low visibility - reset stability tracking
                                        stability_frames = 0
                                        stable_position = None
                                        stable_angle = None
                                    
                                    if smoothed_position is not None and inliers >= 15:
                                        if stable_position is None:
                                            stable_position = smoothed_position.copy()
                                            stable_angle = smoothed_angle
                                            stability_frames = 1
                                        else:
                                            # Check if position is stable (within small radius)
                                            pos_diff = np.linalg.norm(smoothed_position - stable_position)
                                            angle_diff = abs(smoothed_angle - stable_angle)
                                            if angle_diff > np.pi:
                                                angle_diff = 2*np.pi - angle_diff
                                            
                                            if pos_diff < 0.5 and angle_diff < 0.01:  # Stable
                                                stability_frames += 1
                                                if stability_frames >= stability_threshold:
                                                    # Position has been stable for enough frames, update anchor
                                                    anchor_position = stable_position.copy()
                                                    anchor_angle = stable_angle
                                                    allow_locking = True  # Re-enable locking now that anchor is updated
                                                    print(f"\rAnchor updated to new stable position: ({anchor_position[0]:.1f}, {anchor_position[1]:.1f})   ", end='')
                                            else:  # Position changed, reset stability tracking
                                                stable_position = smoothed_position.copy()
                                                stable_angle = smoothed_angle
                                                stability_frames = 1
                                    
                                    # Proceed with Kalman update
                                
                                # Abnormal motion suppression: check if measurement deviates too much
                                # Calculate expected deviation (3σ)
                                pos_sigma = np.sqrt(kf_P_pred[0,0] + kf_P_pred[1,1])
                                angle_sigma = np.sqrt(kf_P_pred[4,4])
                                
                                is_outlier = (innovation_pos > 3*pos_sigma or 
                                             innovation_angle > 3*angle_sigma)
                                
                                if is_outlier and last_corners is not None:
                                    # Reject outlier measurement, use prediction only
                                    center = pred_center
                                    angle = pred_angle
                                    kf_state = kf_state_pred
                                    kf_P = kf_P_pred
                                    status = f"Tracking ({inliers}/{len(good_matches)} inliers) [OUTLIER REJECTED]"
                                else:
                                    # Kalman Filter Update Step
                                    # Measurement matrix
                                    H = np.array([
                                        [1, 0, 0, 0, 0, 0],
                                        [0, 1, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 1, 0]
                                    ])
                                    
                                    # Adaptive measurement noise based on inlier count
                                    # More inliers = more trust in measurement (lower noise)
                                    # Fewer inliers = less trust (higher noise)
                                    inlier_ratio = inliers / max(len(good_matches), 1)
                                    measurement_noise = measurement_noise_base / max(inlier_ratio, 0.1)
                                    
                                    R = np.diag([measurement_noise, measurement_noise, 0.1])
                                    
                                    # Innovation (measurement residual)
                                    y = z - H @ kf_state_pred
                                    
                                    # Normalize angle difference to [-π, π]
                                    if y[2] > np.pi:
                                        y[2] -= 2*np.pi
                                    elif y[2] < -np.pi:
                                        y[2] += 2*np.pi
                                    
                                    # Innovation covariance
                                    S = H @ kf_P_pred @ H.T + R
                                    
                                    # Kalman gain
                                    K = kf_P_pred @ H.T @ np.linalg.inv(S)
                                    
                                    # Update state
                                    kf_state = kf_state_pred + K @ y
                                    
                                    # Update covariance
                                    kf_P = (np.eye(6) - K @ H) @ kf_P_pred
                                    
                                    # Use Kalman filtered values
                                    center = kf_state[:2]
                                    angle = kf_state[4]
                                    
                                    # Apply exponential moving average smoothing to reduce jitter
                                    if smoothed_position is None:
                                        smoothed_position = center.copy()
                                        smoothed_angle = angle
                                    else:
                                        smoothed_position = alpha_smooth * center + (1 - alpha_smooth) * smoothed_position
                                        # Handle angle wrapping for smoothing
                                        angle_diff = angle - smoothed_angle
                                        if angle_diff > np.pi:
                                            angle_diff -= 2*np.pi
                                        elif angle_diff < -np.pi:
                                            angle_diff += 2*np.pi
                                        smoothed_angle = smoothed_angle + alpha_smooth * angle_diff
                                    
                                    center = smoothed_position
                                    angle = smoothed_angle
                                    
                                    # Normal status
                                    if inliers < 12:
                                        status = f"Tracking ({inliers}/{len(good_matches)} inliers) [LOW VIS]"
                                    else:
                                        status = f"Tracking ({inliers}/{len(good_matches)} inliers) [KF]"
                        
                        # Reconstruct box with Kalman-filtered center and angle
                        R = np.array([
                            [np.cos(angle), -np.sin(angle)],
                            [np.sin(angle),  np.cos(angle)]
                        ])
                        corners = (box @ R.T) + center

                        # Update velocity for prediction nigger
                        if last_corners is not None:
                            new_velocity = corners - last_corners
                            velocity = alpha_velocity * new_velocity + (1 - alpha_velocity) * velocity

                        last_corners = corners
                        last_H = H
                        lost_frames = 0
                        frames_since_update += 1
                        
                        # Update reference template if tracking is very stable
                        if inliers > len(good_matches) * 0.7 and frames_since_update > update_interval:
                            # Extract current view of tracked object
                            src_rect = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32)
                            M = cv2.getPerspectiveTransform(corners.astype(np.float32), src_rect)
                            warped = cv2.warpPerspective(gray, M, (w, h))
                            
                            # Recompute features on warped image
                            kp_new, des_new = detector.detectAndCompute(warped, None)
                            if des_new is not None and len(kp_new) >= len(kp_ref) * 0.5:
                                # Blend old and new descriptors
                                kp_ref = kp_new
                                des_ref = des_new
                                frames_since_update = 0
                                status = f"Tracking ({inliers}/{len(good_matches)} inliers) [UPDATED]"
                            else:
                                status = f"Tracking ({inliers}/{len(good_matches)} inliers)"
                        else:
                            status = f"Tracking ({inliers}/{len(good_matches)} inliers)"
                        
                        # Store points for optical flow backup
                        flow_points = corners.astype(np.float32).reshape(-1, 1, 2)
        
        # If no good detection, try optical flow backup or prediction
        if corners is None and prev_gray is not None and flow_points is not None and last_corners is not None:
            # Try optical flow as backup
            try:
                new_points, status_flow, _ = cv2.calcOpticalFlowPyrLK(
                    prev_gray, gray, flow_points, None,
                    winSize=(21, 21), maxLevel=3,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
                )
                
                if new_points is not None and np.all(status_flow == 1):
                    corners = new_points.reshape(-1, 2)
                    flow_points = new_points
                    status = f"Optical Flow Backup"
            except:
                pass
        
        # If still no detection, use velocity-based prediction or mark as lost
        if corners is None:
            lost_frames += 1
            if lost_frames < 20 and last_corners is not None:
                # Use velocity-based prediction
                if lost_frames < 8 and np.any(velocity != 0):
                    corners = last_corners + velocity * lost_frames
                    status = f"Velocity Prediction ({lost_frames} frames)"
                else:
                    # Keep showing last known position
                    corners = last_corners
                    status = f"Lost {lost_frames} frames"
            else:
                corners = None
                corner_history.clear()  # Clear history on complete loss
                velocity = np.zeros((4, 2))  # Reset velocity
        
        # Visualize
        vis_frame = frame.copy()
        vis_frame = draw_tracked_object(vis_frame, corners, status)
        
        # Show frame counter
        cv2.putText(vis_frame, f"Frame: {frame_count}", (width - 200, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Store current frame for optical flow
        prev_gray = gray.copy()
        
        # Display
        cv2.imshow("Object Tracking", vis_frame)
        
        if output_path:
            out.write(vis_frame)
        
        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Interrupted by user")
            break
    
    # Cleanup
    if use_realsense:
        pipeline.stop()
    else:
        cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()
    
    print(f"\nProcessing complete: {frame_count} frames")
    if output_path:
        print(f"Output saved to: {output_path}")


# Example usage
if __name__ == "__main__":
    print("=== ORB + Homography Object Tracker ===")
    print("This script will:")
    print("1. Open your camera/video")
    print("2. Let you draw a box around the object in the first frame")
    print("3. Track that object through the video")
    print("4. Show position, rotation, and recovery from occlusion\n")
    
    # Change these settings for your use case
    VIDEO_INPUT = "realsense"  # Use "realsense" for RealSense camera, 0 for webcam, or "path/to/video.mp4"
    VIDEO_OUTPUT = "tracked_output.mp4"  # Set to None to disable saving
    
    track_object_in_video(VIDEO_INPUT, VIDEO_OUTPUT)
