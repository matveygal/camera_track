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
FRAME_HISTORY = 20  # Frames to track
MIN_CONSECUTIVE_FRAMES = 15  # Dot must appear in this many CONSECUTIVE frames

# Track history of ALL detected candidates per frame
candidates_history = deque(maxlen=FRAME_HISTORY)
tracked_dot = None  # Currently tracked dot position

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
                    candidates.append((cx, cy, area, circularity))
        
        # Add current frame candidates to history
        candidates_history.append(candidates)
        
        # Find the most stable dot - appears CONSECUTIVELY in recent frames
        best_dot = None
        max_consecutive = 0
        
        if len(candidates_history) >= MIN_CONSECUTIVE_FRAMES:
            # For each current candidate, check consecutive appearance
            for cx, cy, area, circ in candidates:
                # Count consecutive frames with a dot near this position (going backwards)
                consecutive_count = 0
                for past_candidates in reversed(list(candidates_history)):
                    found_in_frame = False
                    for past_cx, past_cy, _, _ in past_candidates:
                        # Check if past candidate is close to current position
                        dist = np.sqrt((cx - past_cx)**2 + (cy - past_cy)**2)
                        if dist < 20:  # Within 20 pixels = same dot
                            found_in_frame = True
                            break
                    
                    if found_in_frame:
                        consecutive_count += 1
                    else:
                        # Break the consecutive chain
                        break
                
                if consecutive_count > max_consecutive:
                    max_consecutive = consecutive_count
                    best_dot = (cx, cy, area, circ, consecutive_count)
            
            # Only accept if it appears consecutively for minimum frames
            if best_dot and best_dot[4] < MIN_CONSECUTIVE_FRAMES:
                best_dot = None
        
        # Draw result
        if best_dot is not None:
            cx, cy, area, circularity, stable_count = best_dot
            
            # Draw the circle and centroid
            cv2.circle(frame, (cx, cy), 15, (0, 255, 255), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            
            # Display info
            cv2.putText(frame, f"Dot: ({cx}, {cy})", 
                       (cx + 20, cy - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Consecutive: {stable_count}/{MIN_CONSECUTIVE_FRAMES}", 
                       (cx + 20, cy + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            cv2.putText(frame, "Waiting for stable dot...", (10, 30),
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
