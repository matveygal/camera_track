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
BLACK_THRESHOLD = 40  # Very strict - only truly black pixels
MIN_DOT_AREA = 15     # Minimum area in pixels for a dot
MAX_DOT_AREA = 300    # Small dots only
MIN_CIRCULARITY = 0.5 # Must be reasonably circular
CENTER_CROP_RATIO = 0.6  # Focus on center 60% of frame

# Temporal filtering
FRAME_HISTORY = 5  # Number of frames to average
black_history = []  # Store history of black masks

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        
        if not color_frame:
            continue
        
        # Convert to numpy array
        color_image = np.asanyarray(color_frame.get_data())
        
        # Crop to center region
        h, w = color_image.shape[:2]
        crop_h = int(h * CENTER_CROP_RATIO)
        crop_w = int(w * CENTER_CROP_RATIO)
        y1 = (h - crop_h) // 2
        x1 = (w - crop_w) // 2
        y2 = y1 + crop_h
        x2 = x1 + crop_w
        
        center_crop = color_image[y1:y2, x1:x2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(center_crop, cv2.COLOR_BGR2GRAY)
        
        # Very strict threshold - only truly black pixels
        _, black_mask = cv2.threshold(gray, BLACK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
        
        # Add to history
        black_history.append(black_mask.astype(np.float32) / 255.0)
        if len(black_history) > FRAME_HISTORY:
            black_history.pop(0)
        
        # Average across frames - only pixels that are consistently black will remain
        if len(black_history) >= FRAME_HISTORY:
            avg_black = np.mean(black_history, axis=0)
            # Pixels must be black in at least 80% of frames
            consistent_black = (avg_black > 0.8).astype(np.uint8) * 255
        else:
            # Not enough frames yet, use current frame
            consistent_black = black_mask
        
        # Find contours
        contours, _ = cv2.findContours(consistent_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
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
            # Get the center (adjust for crop offset)
            M = cv2.moments(best_dot)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"]) + x1
                cy = int(M["m01"] / M["m00"]) + y1
                
                # Draw crosshair on the dot
                cv2.drawMarker(display_image, (cx, cy), (0, 255, 0), 
                             cv2.MARKER_CROSS, 20, 2)
                
                # Draw contour (adjust for crop offset)
                adjusted_contour = best_dot + np.array([x1, y1])
                cv2.drawContours(display_image, [adjusted_contour], -1, (0, 255, 0), 2)
                
                # Display position
                cv2.putText(display_image, f"Dot: ({cx}, {cy})", (cx + 10, cy - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            cv2.putText(display_image, "No dot detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw crop region for reference
        cv2.rectangle(display_image, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(display_image, "Search Area", (x1 + 10, y1 + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Show the mask for debugging - pad cropped mask to full size
        full_consistent_black = np.zeros((h, w), dtype=np.uint8)
        full_consistent_black[y1:y2, x1:x2] = consistent_black
        consistent_black_colored = cv2.cvtColor(full_consistent_black, cv2.COLOR_GRAY2BGR)
        combined = np.hstack((display_image, consistent_black_colored))
        
        cv2.imshow('Dot Tracker | Black Mask (Press Q to quit)', combined)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("Camera stopped")
