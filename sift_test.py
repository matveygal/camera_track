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
                        # Geometric regularization: keep shape close to initial
                        sides, angles = get_sides_and_angles(corners)
                        init_sides, init_angles = initial_shape
                        # Clamp side lengths and angles to within 20% of initial
                        sides = np.clip(sides, 0.8*init_sides, 1.2*init_sides)
                        angles = np.clip(angles, 0.8*init_angles, 1.2*init_angles)
                        # Area and aspect ratio constraints
                        def quad_area(pts):
                            return 0.5 * abs(
                                pts[0,0]*pts[1,1] + pts[1,0]*pts[2,1] + pts[2,0]*pts[3,1] + pts[3,0]*pts[0,1]
                                - pts[1,0]*pts[0,1] - pts[2,0]*pts[1,1] - pts[3,0]*pts[2,1] - pts[0,0]*pts[3,1]
                            )
                        initial_area = quad_area(ref_corners)
                        area = quad_area(corners)
                        min_area = 0.7 * initial_area
                        max_area = 1.3 * initial_area
                        if area < min_area or area > max_area:
                            scale = np.sqrt(initial_area / (area + 1e-6))
                            center = np.mean(corners, axis=0)
                            corners = (corners - center) * scale + center
                        # Aspect ratio constraint
                        def aspect_ratio(pts):
                            w = (np.linalg.norm(pts[0]-pts[1]) + np.linalg.norm(pts[2]-pts[3])) / 2
                            h = (np.linalg.norm(pts[1]-pts[2]) + np.linalg.norm(pts[3]-pts[0])) / 2
                            return w / (h + 1e-6)
                        initial_ar = aspect_ratio(ref_corners)
                        ar = aspect_ratio(corners)
                        if ar < 0.7*initial_ar or ar > 1.3*initial_ar:
                            # Adjust width/height to restore aspect ratio
                            center = np.mean(corners, axis=0)
                            scale_w = np.sqrt(initial_ar/ar) if ar > initial_ar else 1.0
                            scale_h = np.sqrt(ar/initial_ar) if ar < initial_ar else 1.0
                            for i in range(4):
                                vec = corners[i] - center
                                if i % 2 == 0:
                                    vec[0] *= scale_w
                                    vec[1] *= scale_h
                                else:
                                    vec[0] *= scale_w
                                    vec[1] *= scale_h
                                corners[i] = center + vec
                        # Limit per-corner movement (max 10px per frame)
                        if last_corners is not None:
                            max_move = 10.0
                            delta = corners - last_corners
                            delta = np.clip(delta, -max_move, max_move)
                            corners = last_corners + delta
                        # Update velocity for prediction
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
