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


def select_rotated_roi(frame):
    """Custom ROI selector with rotation support (45-degree steps)"""
    print("Select ROI with mouse, then rotate with 'r'/'R' keys")
    print("Press ENTER/SPACE to confirm, 'q' to cancel")
    
    # First, get axis-aligned ROI
    roi = cv2.selectROI("Select Object (then rotate if needed)", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Object (then rotate if needed)")
    
    x, y, w, h = [int(v) for v in roi]
    if w == 0 or h == 0:
        return None, None, None
    
    # Allow user to rotate the ROI
    rotation_angle = 0  # degrees
    confirmed = False
    
    while not confirmed:
        # Create display frame
        display = frame.copy()
        
        # Calculate rotated corners
        center = np.array([x + w/2, y + h/2])
        angle_rad = np.radians(rotation_angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        
        # Corners of axis-aligned box relative to center
        half_w, half_h = w/2, h/2
        corners_local = np.array([
            [-half_w, -half_h],
            [ half_w, -half_h],
            [ half_w,  half_h],
            [-half_w,  half_h]
        ])
        
        # Rotate and translate
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        corners = (corners_local @ R.T) + center
        
        # Draw rotated box
        pts = corners.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(display, [pts], True, (0, 255, 0), 2)
        cv2.circle(display, tuple(center.astype(int)), 5, (0, 255, 0), -1)
        
        # Show instructions
        cv2.putText(display, f"Rotation: {rotation_angle} deg", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, "Press 'r' to rotate +45deg, 'R' for -45deg", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(display, "Press ENTER/SPACE to confirm", (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow("Rotate ROI", display)
        
        key = cv2.waitKey(0) & 0xFF
        if key == ord('r'):
            rotation_angle = (rotation_angle + 45) % 360
        elif key == ord('R'):
            rotation_angle = (rotation_angle - 45) % 360
        elif key == ord(' ') or key == 13:  # Space or Enter
            confirmed = True
        elif key == ord('q'):
            cv2.destroyWindow("Rotate ROI")
            return None, None, None
    
    cv2.destroyWindow("Rotate ROI")
    
    # Return the rotated corners and the center/dimensions
    return corners, (x, y, w, h), rotation_angle


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
    
    # Let user select rotated ROI
    print("Select the object to track (you can rotate after selection)")
    roi_corners, roi_data, roi_rotation = select_rotated_roi(first_frame)
    
    if roi_corners is None:
        print("Error: No ROI selected")
        if use_realsense:
            pipeline.stop()
        else:
            cap.release()
        return
    
    x, y, w, h = roi_data
    
    # Extract rotated reference image using perspective transform
    # Map the rotated corners back to an axis-aligned rectangle
    src_pts = roi_corners.astype(np.float32)
    dst_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    ref_img = cv2.warpPerspective(first_frame, M, (w, h))
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

                        # Speed limit: prevent sudden jumps from tracking errors
                        if last_corners is not None:
                            last_center = np.mean(last_corners, axis=0)
                            movement_mag = np.linalg.norm(center - last_center)
                            
                            # Check position speed
                            if movement_mag > max_speed:
                                # Clamp to max speed in the same direction
                                direction = (center - last_center) / (movement_mag + 1e-6)
                                center = last_center + direction * max_speed
                                status = f"Tracking ({inliers}/{len(good_matches)} inliers) [SPEED LIMIT]"
                            
                            # Check rotation speed
                            last_angle = np.arctan2(
                                last_corners[1][1] - last_corners[0][1],
                                last_corners[1][0] - last_corners[0][0]
                            )
                            angle_change = abs(np.degrees(angle - last_angle))
                            if angle_change > 180:
                                angle_change = 360 - angle_change
                            
                            if angle_change > max_rotation_speed:
                                # Keep previous angle
                                angle = last_angle
                                if "SPEED LIMIT" not in status:
                                    status = f"Tracking ({inliers}/{len(good_matches)} inliers) [ROT LIMIT]"
                            
                            # Reconstruct with potentially limited center and angle
                            R = np.array([
                                [np.cos(angle), -np.sin(angle)],
                                [np.sin(angle),  np.cos(angle)]
                            ])
                            corners = (box @ R.T) + center

                        # Dead zone: detect stationary state and prevent drift
                        if last_corners is not None:
                            movement_mag = np.linalg.norm(center - np.mean(last_corners, axis=0))
                            movement_history.append(movement_mag)
                            if len(movement_history) > 10:
                                movement_history.pop(0)
                            
                            # Track angle changes
                            last_angle = np.arctan2(
                                last_corners[1][1] - last_corners[0][1],
                                last_corners[1][0] - last_corners[0][0]
                            )
                            angle_change = abs(np.degrees(angle - last_angle))
                            # Normalize to 0-180 range
                            if angle_change > 180:
                                angle_change = 360 - angle_change
                            angle_history.append(angle_change)
                            if len(angle_history) > 10:
                                angle_history.pop(0)
                            
                            # Check if stationary (consistently low movement)
                            if len(movement_history) >= stationary_frames_needed:
                                avg_movement = np.mean(movement_history[-stationary_frames_needed:])
                                
                                if avg_movement < stationary_threshold:
                                    # Object is stationary - freeze position
                                    if not is_stationary:
                                        # Just became stationary - set anchor
                                        anchor_position = center.copy()
                                        is_stationary = True
                                    # Use anchored position
                                    center = anchor_position
                                    status = f"Tracking ({inliers}/{len(good_matches)} inliers) [LOCKED]"
                                else:
                                    # Object is moving - allow tracking
                                    is_stationary = False
                                    anchor_position = None
                            
                            # Check if rotationally stationary
                            if len(angle_history) >= stationary_frames_needed:
                                avg_angle_change = np.mean(angle_history[-stationary_frames_needed:])
                                
                                if avg_angle_change < angle_threshold:
                                    # Object is rotationally stationary - freeze angle
                                    if not is_rotationally_stationary:
                                        # Just became rotationally stationary - set anchor
                                        anchor_angle = angle
                                        is_rotationally_stationary = True
                                    # Use anchored angle
                                    angle = anchor_angle
                                    if is_stationary:
                                        status = f"Tracking ({inliers}/{len(good_matches)} inliers) [LOCKED+ROT]"
                                else:
                                    # Object is rotating - allow tracking
                                    is_rotationally_stationary = False
                                    anchor_angle = None
                        
                        # Reconstruct box with (possibly locked) center and angle
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
