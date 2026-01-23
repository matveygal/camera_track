import cv2
import numpy as np

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
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
    
    # Read first frame
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Cannot read first frame")
        return
    
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
    ]).reshape(-1, 1, 2)
    
    # Initialize feature detector - using SIFT for better rotation/scale invariance
    # Fallback to ORB if SIFT not available
    try:
        detector = cv2.SIFT_create(nfeatures=2000)
        print("Using SIFT detector (best for rotation/scale)")
    except:
        detector = cv2.ORB_create(nfeatures=2000, 
                                   scaleFactor=1.2, 
                                   nlevels=8, 
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
    
    # Reset to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    print("Processing video... Press 'q' to quit early")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect features in current frame
        kp, des = detector.detectAndCompute(gray, None)
        
        corners = None
        status = "Tracking"
        
        if des is not None and len(kp) >= 4:
            # Match descriptors
            matches = matcher.knnMatch(des_ref, des, k=2)
            
            # Apply Lowe's ratio test (more lenient for better recall)
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.8 * n.distance:  # More lenient ratio
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
                        corners = cv2.perspectiveTransform(ref_corners, H)
                        corners = corners.reshape(-1, 2)
                        
                        # Temporal smoothing with history
                        corner_history.append(corners.copy())
                        if len(corner_history) > 5:
                            corner_history.pop(0)
                        
                        # Average with recent history for stability
                        if len(corner_history) >= 3:
                            # Use only last 3 frames for smoothing
                            recent = corner_history[-3:]
                            weights = np.array([0.2, 0.3, 0.5])
                            # Stack and compute weighted average
                            stacked = np.stack(recent, axis=0)
                            corners = np.average(stacked, axis=0, weights=weights)
                        
                        last_corners = corners
                        last_H = H
                        lost_frames = 0
                        status = f"Tracking ({inliers}/{len(good_matches)} inliers)"
        
        # If no good detection, use prediction or mark as lost
        if corners is None:
            lost_frames += 1
            if lost_frames < 15 and last_corners is not None:
                # Try to predict using last homography if available
                if lost_frames < 5 and last_H is not None:
                    # Use last known homography as prediction
                    corners = last_corners
                    status = f"Predicting ({lost_frames} frames)"
                else:
                    # Keep showing last known position
                    corners = last_corners
                    status = f"Lost {lost_frames} frames"
            else:
                corners = None
                corner_history.clear()  # Clear history on complete loss
        
        # Visualize
        vis_frame = frame.copy()
        vis_frame = draw_tracked_object(vis_frame, corners, status)
        
        # Show frame counter
        cv2.putText(vis_frame, f"Frame: {frame_count}", (width - 200, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display
        cv2.imshow("Object Tracking", vis_frame)
        
        if output_path:
            out.write(vis_frame)
        
        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Interrupted by user")
            break
    
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
    print("1. Open your video")
    print("2. Let you draw a box around the object in the first frame")
    print("3. Track that object through the video")
    print("4. Show position, rotation, and recovery from occlusion\n")
    
    # Change these paths for your use case
    VIDEO_INPUT = "videos/video_20260123_101224.avi"  # Use 0 for webcam, or "path/to/video.mp4"
    VIDEO_OUTPUT = "tracked_output.mp4"  # Set to None to disable saving
    
    track_object_in_video(VIDEO_INPUT, VIDEO_OUTPUT)
