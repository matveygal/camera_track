import pyrealsense2 as rs
import numpy as np
import cv2

print("Initializing RealSense camera...")

# Configure streams  
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)
print("Camera started! Press 'q' to quit")

# Setup SimpleBlobDetector with parameters tuned for a black dot
params = cv2.SimpleBlobDetector_Params()

# Filter by Area (size of the dot)
params.filterByArea = True
params.minArea = 10
params.maxArea = 500

# Filter by Circularity (how round it is)
params.filterByCircularity = True
params.minCircularity = 0.5

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.7

# Filter by Inertia (elongation)
params.filterByInertia = True
params.minInertiaRatio = 0.5

# Create detector
detector = cv2.SimpleBlobDetector_create(params)

# Centroid tracking variables
tracked_dot = None  # (x, y) of tracked dot
MAX_DISTANCE = 50   # Max distance between frames for same dot (pixels)

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        
        if not color_frame:
            continue
        
        # Convert to numpy
        frame = np.asanyarray(color_frame.get_data())
        
        # Preprocess: convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Invert so black becomes white (SimpleBlobDetector expects white blobs)
        inverted = cv2.bitwise_not(gray)
        
        # Detect blobs
        keypoints = detector.detect(inverted)
        
        # Draw ALL detected blobs for debugging (in yellow)
        for i, kp in enumerate(keypoints):
            x, y = int(kp.pt[0]), int(kp.pt[1])
            radius = int(kp.size / 2)
            cv2.circle(frame, (x, y), radius, (0, 255, 255), 1)
            cv2.putText(frame, str(i), (x + 5, y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        # Find the best match for our tracked dot
        current_dot = None
        
        if len(keypoints) > 0:
            if tracked_dot is None:
                # No previous tracking - pick the first detected blob
                kp = keypoints[0]
                current_dot = (int(kp.pt[0]), int(kp.pt[1]))
            else:
                # Find blob closest to previous position (centroid tracking)
                min_distance = float('inf')
                best_match = None
                
                for kp in keypoints:
                    x, y = int(kp.pt[0]), int(kp.pt[1])
                    # Calculate Euclidean distance from previous position
                    distance = np.sqrt((x - tracked_dot[0])**2 + (y - tracked_dot[1])**2)
                    
                    if distance < min_distance:
                        min_distance = distance
                        best_match = (x, y)
                
                # Only accept if within reasonable distance (same dot, not a jump)
                if min_distance < MAX_DISTANCE:
                    current_dot = best_match
                else:
                    # Lost tracking - reset
                    tracked_dot = None
        
        # Update tracked position
        if current_dot is not None:
            tracked_dot = current_dot
        
        # Draw result
        if tracked_dot is not None:
            cx, cy = tracked_dot
            
            # Draw circle and crosshair (in GREEN for tracked dot)
            cv2.circle(frame, (cx, cy), 20, (0, 255, 0), 3)
            cv2.drawMarker(frame, (cx, cy), (0, 255, 0), 
                          cv2.MARKER_CROSS, 25, 3)
            
            # Display position
            cv2.putText(frame, f"Dot: ({cx}, {cy})", 
                       (cx + 20, cy - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, "TRACKING", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Searching for dot...", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Show candidate blobs count
        cv2.putText(frame, f"Blobs: {len(keypoints)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Display
        cv2.imshow('Centroid Tracking', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("Camera stopped")
