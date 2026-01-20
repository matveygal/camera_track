#!/usr/bin/env python3
"""
Track black dot using sparse optical flow (Lucas-Kanade)
Handles local deformation by tracking just the center point
"""
import pyrealsense2 as rs
import numpy as np
import cv2

def select_dot_point(frame):
    """Let user select ROI around dot, then find the best feature point"""
    print("Select region around the black dot, then press ENTER or SPACE")
    roi = cv2.selectROI("Select Dot Region", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Dot Region")
    
    x, y, w, h = roi
    if w == 0 or h == 0:
        return None
    
    # Extract ROI
    roi_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
    
    # Find the best corner/feature point in the ROI
    feature_params = dict(
        maxCorners=1,
        qualityLevel=0.01,
        minDistance=10,
        blockSize=7
    )
    
    corners = cv2.goodFeaturesToTrack(roi_gray, **feature_params)
    
    if corners is None:
        print("Warning: No feature found, using ROI center")
        point = np.array([[[x + w/2, y + h/2]]], dtype=np.float32)
    else:
        # Convert corner coordinates from ROI to full frame
        point = corners.copy()
        point[0][0][0] += x
        point[0][0][1] += y
    
    print(f"Initial tracking point: ({point[0][0][0]:.1f}, {point[0][0][1]:.1f})")
    return point

def main():
    # Configure RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    print("Starting RealSense camera...")
    pipeline.start(config)
    
    # Optical flow parameters
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )
    
    point = None
    old_gray = None
    
    print("\nControls:")
    print("  c - Capture/recapture dot point")
    print("  r - Reset tracking")
    print("  q - Quit")
    
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            
            frame = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # If we have a point to track
            if point is not None and old_gray is not None:
                # Calculate optical flow
                new_point, status, error = cv2.calcOpticalFlowPyrLK(
                    old_gray, gray, point, None, **lk_params
                )
                
                if status[0][0] == 1:  # Successfully tracked
                    point = new_point
                    x, y = point[0][0]
                    
                    # Draw tracking point
                    cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
                    cv2.circle(frame, (int(x), int(y)), 20, (0, 255, 0), 2)
                    
                    # Display coordinates
                    cv2.putText(frame, f"X: {x:.1f}, Y: {y:.1f}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.6, (0, 255, 0), 2)
                    cv2.putText(frame, f"Error: {error[0][0]:.2f}", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.6, (0, 255, 0), 2)
                else:
                    # Lost tracking
                    cv2.putText(frame, "TRACKING LOST - Press 'c' to recapture", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.6, (0, 0, 255), 2)
                    point = None
            else:
                # No tracking active
                cv2.putText(frame, "Press 'c' to capture dot", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (0, 255, 255), 2)
            
            # Update old frame
            old_gray = gray.copy()
            
            cv2.imshow('Optical Flow Tracking', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Capture new point
                point = select_dot_point(frame)
                if point is not None:
                    old_gray = gray.copy()
                    print("Point captured! Tracking started.")
            elif key == ord('r'):
                # Reset tracking
                point = None
                old_gray = None
                print("Tracking reset.")
    
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("\nCamera stopped.")

if __name__ == "__main__":
    main()
