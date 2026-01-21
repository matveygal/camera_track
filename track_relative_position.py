#!/usr/bin/env python3
"""
Track a point of interest relative to a stable reference frame (heart assembly)
User clicks on point → we track it relative to tracked features on the frame
No black dot needed - works on any surface!
"""
import pyrealsense2 as rs
import numpy as np
import cv2

# Feature detection parameters
feature_params = dict(
    maxCorners=50,
    qualityLevel=0.01,
    minDistance=10,
    blockSize=7
)

# Optical flow parameters
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

def select_reference_frame_and_point(frame):
    """Let user select reference region (heart assembly) and point of interest"""
    print("\n=== STEP 1: Select the heart assembly region ===")
    print("Draw a box around the entire heart assembly, then press ENTER")
    
    roi = cv2.selectROI("Select Heart Assembly", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Heart Assembly")
    
    x, y, w, h = roi
    if w == 0 or h == 0:
        print("No region selected!")
        return None, None, None
    
    # Find good features in the assembly region
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi_gray = gray[y:y+h, x:x+w]
    
    corners = cv2.goodFeaturesToTrack(roi_gray, **feature_params)
    
    if corners is None or len(corners) < 4:
        print("Not enough features found in assembly region!")
        return None, None, None
    
    # Convert corners to full frame coordinates
    corners[:, 0, 0] += x
    corners[:, 0, 1] += y
    
    print(f"Found {len(corners)} tracking features in assembly")
    
    # Draw features on frame
    vis_frame = frame.copy()
    for corner in corners:
        px, py = corner[0]
        cv2.circle(vis_frame, (int(px), int(py)), 3, (0, 255, 0), -1)
    
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
    relative_pos = point_of_interest - centroid
    
    print(f"Relative position from feature centroid: ({relative_pos[0]:.1f}, {relative_pos[1]:.1f})")
    
    return corners, relative_pos, (x, y, w, h)

def main():
    # Configure RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    print("Starting RealSense camera...")
    pipeline.start(config)
    
    # State
    reference_features = None
    relative_position = None
    old_gray = None
    tracking = False
    
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
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if tracking and reference_features is not None and old_gray is not None:
                # Track reference features
                new_features, status, error = cv2.calcOpticalFlowPyrLK(
                    old_gray, gray, reference_features, None, **lk_params
                )
                
                # Filter out lost features
                good_new = new_features[status == 1]
                good_old = reference_features[status == 1]
                
                if len(good_new) < 4:
                    cv2.putText(frame, "LOST TRACKING - Press 'c' to recapture", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.6, (0, 0, 255), 2)
                    tracking = False
                else:
                    # Update reference features
                    reference_features = good_new.reshape(-1, 1, 2)
                    
                    # Calculate new centroid
                    centroid = np.mean(reference_features[:, 0], axis=0)
                    
                    # Calculate point of interest position
                    point_of_interest = centroid + relative_position
                    px, py = int(point_of_interest[0]), int(point_of_interest[1])
                    
                    # Draw tracked features (green)
                    for feature in reference_features:
                        fx, fy = feature[0]
                        cv2.circle(frame, (int(fx), int(fy)), 3, (0, 255, 0), -1)
                    
                    # Draw centroid (blue)
                    cv2.circle(frame, (int(centroid[0]), int(centroid[1])), 5, (255, 0, 0), -1)
                    
                    # Draw point of interest (red - large)
                    cv2.circle(frame, (px, py), 8, (0, 0, 255), -1)
                    cv2.circle(frame, (px, py), 25, (0, 0, 255), 2)
                    cv2.drawMarker(frame, (px, py), (0, 255, 255), 
                                  cv2.MARKER_CROSS, 30, 2)
                    
                    # Display info
                    cv2.putText(frame, f"Point: ({px}, {py})", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (0, 0, 255), 2)
                    cv2.putText(frame, f"Features: {len(reference_features)}", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.6, (0, 255, 0), 2)
                    cv2.putText(frame, f"Centroid: ({int(centroid[0])}, {int(centroid[1])})", 
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
            
            cv2.imshow('Relative Position Tracking', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Capture reference and point
                result = select_reference_frame_and_point(frame)
                if result[0] is not None:
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
