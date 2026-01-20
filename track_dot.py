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

# Parameters for black dot detection
BLACK_THRESHOLD = 80  # Pixels darker than this are considered black (increased to catch dark gray)
MIN_DOT_AREA = 10     # Minimum area in pixels for a dot
MAX_DOT_AREA = 800    # Maximum area to exclude large black regions
MIN_CIRCULARITY = 0.3 # How circular the blob should be (0-1, 1 = perfect circle)

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        
        if not color_frame:
            continue
        
        # Convert to numpy array
        color_image = np.asanyarray(color_frame.get_data())
        
        # Convert to grayscale
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        
        # Threshold to find black pixels
        _, black_mask = cv2.threshold(gray, BLACK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Display frame for visualization
        display_image = color_image.copy()
        
        # Filter contours to find the dot
        best_dot = None
        best_circularity = 0
        
        # Debug: count candidates
        candidates = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < MIN_DOT_AREA or area > MAX_DOT_AREA:
                continue
            
            # Calculate circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            candidates.append((area, circularity))
            
            # Keep the most circular dot within size range
            if circularity > MIN_CIRCULARITY and circularity > best_circularity:
                best_circularity = circularity
                best_dot = contour
        
        # Debug info
        cv2.putText(display_image, f"Candidates: {len(candidates)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Draw the detected dot
        if best_dot is not None:
            # Get the center
            M = cv2.moments(best_dot)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Draw crosshair on the dot
                cv2.drawMarker(display_image, (cx, cy), (0, 255, 0), 
                             cv2.MARKER_CROSS, 20, 2)
                
                # Draw contour
                cv2.drawContours(display_image, [best_dot], -1, (0, 255, 0), 2)
                
                # Display position
                cv2.putText(display_image, f"Dot: ({cx}, {cy})", (cx + 10, cy - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            cv2.putText(display_image, "No dot detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Show the mask for debugging
        black_mask_colored = cv2.cvtColor(black_mask, cv2.COLOR_GRAY2BGR)
        combined = np.hstack((display_image, black_mask_colored))
        
        cv2.imshow('Dot Tracker | Black Mask (Press Q to quit)', combined)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("Camera stopped")
