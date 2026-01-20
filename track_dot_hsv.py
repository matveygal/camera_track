import pyrealsense2 as rs
import numpy as np
import cv2
from collections import deque

print("Initializing RealSense camera...")

# Configure streams  
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)
print("Camera started! Press 'q' to quit")

# HSV range for black dot (tuned values)
LOWER_BOUND = np.array([0, 0, 45])
UPPER_BOUND = np.array([179, 255, 89])

# Dot detection parameters
MIN_DOT_AREA = 10
MAX_DOT_AREA = 500
MIN_CIRCULARITY = 0.4  # Dots are circular
FRAME_HISTORY = 5  # Frames to average for temporal consistency

# Track history of detected positions
position_history = deque(maxlen=FRAME_HISTORY)

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        
        if not color_frame:
            continue
        
        # Convert to numpy
        frame = np.asanyarray(color_frame.get_data())
        
        # Blur to reduce noise
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        
        # Convert to HSV color space
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        # Create mask for black color
        mask = cv2.inRange(hsv, LOWER_BOUND, UPPER_BOUND)
        
        # Clean up mask with morphological operations
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours - look for small, circular, isolated dots
        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            
            # Filter by area
            if area < MIN_DOT_AREA or area > MAX_DOT_AREA:
                continue
            
            # Calculate circularity
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # Only keep circular blobs
            if circularity > MIN_CIRCULARITY:
                # Get center
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    candidates.append((c, cx, cy, area, circularity))
        
        # Track the most stable dot using temporal consistency
        best_dot = None
        if len(candidates) > 0:
            # If we have history, prefer dots close to previous positions
            if len(position_history) > 0:
                avg_prev_x = np.mean([p[0] for p in position_history])
                avg_prev_y = np.mean([p[1] for p in position_history])
                
                # Find candidate closest to previous average position
                min_dist = float('inf')
                for candidate in candidates:
                    _, cx, cy, _, _ = candidate
                    dist = np.sqrt((cx - avg_prev_x)**2 + (cy - avg_prev_y)**2)
                    if dist < min_dist:
                        min_dist = dist
                        best_dot = candidate
                
                # Only accept if it's reasonably close (not a jump to random noise)
                if min_dist > 100:  # Too far from previous position
                    best_dot = None
            else:
                # No history yet, take the most circular one
                best_dot = max(candidates, key=lambda x: x[4])  # Sort by circularity
        
        # Draw result
        if best_dot is not None:
            c, cx, cy, area, circularity = best_dot
            
            # Add to history
            position_history.append((cx, cy))
            
            # Compute the minimum enclosing circle
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            
            # Draw the circle and centroid
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            
            # Display info
            cv2.putText(frame, f"Dot: ({cx}, {cy})", 
                       (cx + 10, cy - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Circ: {circularity:.2f}", 
                       (cx + 10, cy + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        else:
            cv2.putText(frame, "No dot detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Show candidate count
        cv2.putText(frame, f"Candidates: {len(candidates)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Stack original and mask side by side
        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined = np.hstack((frame, mask_colored))
        
        # Display
        cv2.imshow('Tracking', combined)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("Camera stopped")
