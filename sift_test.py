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
    
    # Dead zone for drift prevention
    movement_history = []  # Track recent movement magnitudes
    is_stationary = False
    stationary_threshold = 1.5  # pixels - below this is considered noise
    stationary_frames_needed = 5  # frames of low movement to be considered stationary
    anchor_position = None  # Locked position when stationary
    
    # Rotational dead zone
    angle_history = []  # Track recent angle changes
    is_rotationally_stationary = False
    angle_threshold = 0.5  # degrees - below this is considered noise
    anchor_angle = None  # Locked angle when rotationally stationary
    
    # Speed limit to prevent tracking errors
    max_speed = 5.0  # pixels per frame - anything above is likely a tracking error
    max_rotation_speed = 2.0  # degrees per frame    
    # Kalman Filter state: [x, y, vx, vy, θ, ω]
    kf_state = None  # Will be initialized on first good measurement
    kf_P = None  # Covariance matrix
    dt = 1.0  # Time step (1 frame)
    
    # Process noise (how much we trust motion model)
    process_noise_pos = 0.5  # Position process noise
    process_noise_vel = 2.0  # Velocity process noise
    process_noise_angle = 0.01  # Angle process noise (radians)
    process_noise_omega = 0.05  # Angular velocity process noise
    
    # Measurement noise (how much we trust observations)
    measurement_noise_pos = 2.0  # Will be adjusted based on inlier count
    measurement_noise_angle = 0.02  # Angle measurement noise
    
    # Abnormal motion detection
    motion_buffer = []  # Store recent velocities for outlier detection
    motion_buffer_size = 10    
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
                    # Count inliers
                    inliers = np.sum(mask)
                    
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
                        corners = (box @ R.T) + center

                        # Kalman Filter: predict and update
                        if kf_state is None:
                            # Initialize Kalman Filter on first good measurement
                            kf_state = np.array([center[0], center[1], 0.0, 0.0, angle, 0.0])
                            kf_P = np.eye(6) * 10.0  # Initial uncertainty
                        else:
                            # Kalman Prediction Step
                            # State transition: x' = x + vx*dt, y' = y + vy*dt, θ' = θ + ω*dt
                            F = np.array([
                                [1, 0, dt, 0,  0, 0],
                                [0, 1, 0,  dt, 0, 0],
                                [0, 0, 1,  0,  0, 0],
                                [0, 0, 0,  1,  0, 0],
                                [0, 0, 0,  0,  1, dt],
                                [0, 0, 0,  0,  0, 1]
                            ])
                            kf_state = F @ kf_state
                            
                            # Adjust process noise based on visibility
                            visibility_score = min(1.0, inliers / 20.0)
                            Q = np.diag([
                                process_noise_pos / (visibility_score + 0.1),
                                process_noise_pos / (visibility_score + 0.1),
                                process_noise_vel / (visibility_score + 0.1),
                                process_noise_vel / (visibility_score + 0.1),
                                process_noise_angle / (visibility_score + 0.1),
                                process_noise_omega / (visibility_score + 0.1)
                            ])
                            kf_P = F @ kf_P @ F.T + Q
                            
                            # Measurement from homography
                            z = np.array([center[0], center[1], angle])
                            predicted_pos = kf_state[:2]
                            predicted_angle = kf_state[4]
                            
                            # Abnormal motion detection
                            position_deviation = np.linalg.norm(center - predicted_pos)
                            angle_diff = abs(angle - predicted_angle)
                            if angle_diff > np.pi:
                                angle_diff = 2*np.pi - angle_diff
                            
                            # Check if measurement is abnormal (likely occlusion/error)
                            is_abnormal = False
                            if len(motion_buffer) >= 5:
                                # Check against recent motion trends
                                recent_velocities = np.array([v[:2] for v in motion_buffer[-5:]])
                                vel_mean = np.mean(recent_velocities, axis=0)
                                vel_std = np.std(recent_velocities, axis=0) + 1e-6
                                current_vel = center - predicted_pos
                                deviation = np.abs(current_vel - vel_mean) / vel_std
                                if np.any(deviation > 2.5) or position_deviation > 15.0:
                                    is_abnormal = True
                            
                            # Kalman Update Step (only if not abnormal)
                            if not is_abnormal and inliers >= 6:
                                # Measurement matrix H: we observe x, y, θ
                                H = np.array([
                                    [1, 0, 0, 0, 0, 0],
                                    [0, 1, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 1, 0]
                                ])
                                
                                # Measurement noise adjusted by visibility
                                R = np.diag([
                                    measurement_noise_pos * (1.0 / (visibility_score + 0.1)),
                                    measurement_noise_pos * (1.0 / (visibility_score + 0.1)),
                                    measurement_noise_angle
                                ])
                                
                                # Innovation
                                y = z - H @ kf_state
                                # Normalize angle difference
                                if y[2] > np.pi:
                                    y[2] -= 2*np.pi
                                elif y[2] < -np.pi:
                                    y[2] += 2*np.pi
                                
                                # Innovation covariance
                                S = H @ kf_P @ H.T + R
                                # Kalman gain
                                K = kf_P @ H.T @ np.linalg.inv(S)
                                # Update state
                                kf_state = kf_state + K @ y
                                # Update covariance
                                kf_P = (np.eye(6) - K @ H) @ kf_P
                                
                                status = f"Tracking ({inliers}/{len(good_matches)} inliers) [KF]"
                            else:
                                # Use prediction only (occlusion or abnormal motion)
                                if is_abnormal:
                                    status = f"Tracking ({inliers}/{len(good_matches)} inliers) [KF-REJECT]"
                                else:
                                    status = f"Tracking ({inliers}/{len(good_matches)} inliers) [KF-PREDICT]"
                            
                            # Update motion buffer
                            motion_buffer.append(kf_state[2:4].copy())
                            if len(motion_buffer) > motion_buffer_size:
                                motion_buffer.pop(0)
                            
                            # Use Kalman filtered position and angle
                            center = kf_state[:2]
                            angle = kf_state[4]
                            
                            # Reconstruct corners with Kalman filtered values
                            R = np.array([
                                [np.cos(angle), -np.sin(angle)],
                                [np.sin(angle),  np.cos(angle)]
                            ])
                            corners = (box @ R.T) + center
                        
                        # Note: Speed limiting and dead zones removed - Kalman Filter handles
                        # noise, drift, and abnormal motion internally

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
