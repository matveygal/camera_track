#!/usr/bin/env python3
"""
Track a point of interest relative to a stable reference frame (heart assembly)
User clicks on point → we track it relative to matched features from reference image
No black dot needed - works on any surface!
Uses feature matching (not optical flow) so it can recover from occlusions
"""
import pyrealsense2 as rs
import numpy as np
import cv2

# ORB feature detector (fast and robust)
orb = cv2.ORB_create(nfeatures=500)

# Feature matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

def select_reference_frame_and_point(frame):
    """Let user select reference region (heart assembly) and point of interest"""
    print("\n=== STEP 1: Select the heart assembly region ===")
    print("Draw a box around the entire heart assembly, then press ENTER")
    
    roi = cv2.selectROI("Select Heart Assembly", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Heart Assembly")
    
    x, y, w, h = roi
    if w == 0 or h == 0:
        print("No region selected!")
        return None, None, None, None, None
    
    # Extract and store reference image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    reference_image = gray[y:y+h, x:x+w].copy()
    
    # Detect keypoints and compute descriptors in reference region
    keypoints, descriptors = orb.detectAndCompute(reference_image, None)
    
    if keypoints is None or len(keypoints) < 10:
        print("Not enough features found in assembly region!")
        return None, None, None, None, None
    
    print(f"Found {len(keypoints)} features in reference image")
    
    # Convert keypoints to full frame coordinates for visualization
    vis_frame = frame.copy()
    for kp in keypoints:
        px, py = kp.pt
        cv2.circle(vis_frame, (int(px + x), int(py + y)), 3, (0, 255, 0), -1)
    
    print("\n=== STEP 2: Click on the point you want to track ===")
    print("Click on the specific point of interest (e.g., where dot would be)")
    
    clicked_point = [None]
    
    def mouse_callback(event, mx, my, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_point[0] = np.array([mx, my], dtype=np.float32)
            cv2.circle(vis_frame, (mx, my), 8, (0, 0, 255), -1)
            cv2.imshow("Click Point of Interest", vis_frame)
    
    cv2.namedWindow("Click Point of Interest")
    cv2.setMouseCallback("Click Point of Interest", mouse_callback)
    cv2.imshow("Click Point of Interest", vis_frame)
    
    print("Waiting for click...")
    while clicked_point[0] is None:
        if cv2.waitKey(100) & 0xFF == ord('q'):
            cv2.destroyWindow("Click Point of Interest")
            return None, None, None
    
    cv2.waitKey(500)
    cv2.destroyWindow("Click Point of Interest")
    
    point_of_interest = clicked_point[0]
    print(f"Point of interest: ({point_of_interest[0]:.1f}, {point_of_interest[1]:.1f})")
    
    # Calculate relative position of point to reference features
    # Use centroid of features as reference origin
    centroid = np.mean(corners[:, 0], axis=0)
    relative_pos = point_of_interest - centroiROI origin
    # We need to store position relative to the ROI, not absolute
    relative_pos = point_of_interest - np.array([x, y])
    
    print(f"Relative position from ROI origin: ({relative_pos[0]:.1f}, {relative_pos[1]:.1f})")
    
    return reference_image, keypoints, descriptoRealSense
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    print("Starting RealSense camera...")
    pipeline.start(config)
    
    # State
    reference_features = None
    relative_pimage = None
    reference_keypoints = None
    reference_descriptors = None
    relative_position = None
    reference_roi = False
    
    print("\nControls:")
    print("  c - Capture reference frame and select point")
    print("  r - Reset tracking")
    print("  q - Quit")
    
    try:
        # Create window
        cv2.namedWindow('Relative Position Tracking', cv2.WINDOW_NORMAL)
        cv2.moveWindow('Relative Position Tracking', 50, 50)
        cv2.resizeWindow('Relative Position Tracking', 640, 480)
        
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            
            frame = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(frame,descriptors is not None:
                # Detect features in current frame
                current_keypoints, current_descriptors = orb.detectAndCompute(gray, None)
                
                if current_descriptors is not None and len(current_keypoints) >= 10:
                    # Match features between reference and current frame
                    matches = bf.match(reference_descriptors, current_descriptors)
                    
                    # Sort by distance (best matches first)
                    matches = sorted(matches, key=lambda x: x.distance)
                    
                    # Keep only good matches (top 50 or those with distance < threshold)
                    good_matches = [m for m in matches if m.distance < 50][:50]
                    
                    if len(good_matches) < 10:
                        cv2.putText(frame, f"WEAK TRACKING ({len(good_matches)} matches) - Move camera back", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.6, (0, 165, 255), 2)
                    else:
                        # Extract matched keypoints
                        ref_pts = np.float32([reference_keypoints[m.queryIdx].pt for m in good_matches])
                        curr_pts = np.float32([current_keypoints[m.trainIdx].pt for m in good_matches])
                        
                        # Find homography (transformation from reference to current)
                        H, mask = cv2.findHomography(ref_pts, curr_pts, cv2.RANSAC, 5.0)
                        
                        if H is not None:
                            # Transform the relative position using homography
                            rel_point = np.array([[relative_position]], dtype=np.float32)
                            transformed_point = cv2.perspectiveTransform(rel_point, H)
                            
                            # Add ROI offset
                            px = int(transformed_point[0][0][0] + reference_roi[0])
                            py = int(transformed_point[0][0][1] + reference_roi[1])
                            
                            # Draw matched features (green)
                            for i, m in enumerate(good_matches):
                                if mask[i]:
                                    pt = current_keypoints[m.trainIdx].pt
                                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)
                            
                            # Draw point of interest (red - large)
                            cv2.circle(frame, (px, py), 8, (0, 0, 255), -1)
                            cv2.circle(frame, (px, py), 25, (0, 0, 255), 2)
                            cv2.drawMarker(frame, (px, py), (0, 255, 255), 
                                          cv2.MARKER_CROSS, 30, 2)
                            
                            # Display info
                            cv2.putText(frame, f"Point: ({px}, {py})", 
                                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                       0.7, (0, 0, 255), 2)
                            cv2.putText(frame, f"Matches: {len(good_matches)} ({sum(mask)}/{len(mask)} inliers)", 
                                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                                       0.6, (0, 2     (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                       0.6, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "NO FEATURES DETECTED - Check lighting/occlusion", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.6, (0, 0, 255oid: ({int(centroid[0])}, {int(centroid[1])})", 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.5, (255, 0, 0), 2)
            else:
                cv2.putText(frame, "Press 'c' to start tracking", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (255, 255, 0), 2)
            
            # Update old frame
            old_gray = gray.copy()
            
            # Display controls
            cv2.putText(frame, "c=capture | r=reset | q=quit", 
                       (10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Relatiimage, reference_keypoints, reference_descriptors, relative_position, reference_roi = result
                    tracking = True
                    print("Tracking started!")
            elif key == ord('r'):
                # Reset
                reference_image = None
                reference_keypoints = None
                reference_descriptors = None
                relative_position = None
                reference_roit[0] is not None:
                    reference_features, relative_position, roi = result
                    old_gray = gray.copy()
                    tracking = True
                    print("Tracking started!")
            elif key == ord('r'):
                # Reset
                reference_features = None
                relative_position = None
                old_gray = None
                tracking = False
                print("Tracking reset")
    
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("\nCamera stopped.")

if __name__ == "__main__":
    main()
