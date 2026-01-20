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
MIN_DOT_AREA = 5      # Very relaxed minimum
MAX_DOT_AREA = 1000   # Very relaxed maximum
CENTER_CROP_RATIO = 0.6  # Focus on center 60% of frame

# Setup blob detector
params = cv2.SimpleBlobDetector_Params()

# Filter by area
params.filterByArea = True
params.minArea = MIN_DOT_AREA
params.maxArea = MAX_DOT_AREA

# Filter by circularity - VERY RELAXED
params.filterByCircularity = True
params.minCircularity = 0.1

# Filter by convexity - VERY RELAXED
params.filterByConvexity = True
params.minConvexity = 0.1

# Filter by color (looking for dark blobs)
params.filterByColor = False  # Disabled - causes issues sometimes

# Filter by inertia (shape) - VERY RELAXED
params.filterByInertia = True
params.minInertiaRatio = 0.01

# Threshold steps
params.minThreshold = 10
params.maxThreshold = 200
params.thresholdStep = 10

detector = cv2.SimpleBlobDetector_create(params)

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
        
        # Invert so black blobs become white for blob detector
        inverted = cv2.bitwise_not(gray)
        
        # Detect blobs
        keypoints = detector.detect(inverted)
        
        # Also try on the original gray image (not inverted)
        keypoints_gray = detector.detect(gray)
        
        # Use whichever found more blobs
        if len(keypoints_gray) > len(keypoints):
            keypoints = keypoints_gray
            detection_method = "gray"
        else:
            detection_method = "inverted"
        
        # Display frame for visualization
        display_image = color_image.copy()
        
        # Draw crop region for reference
        cv2.rectangle(display_image, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(display_image, "Search Area", (x1 + 10, y1 + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.putText(display_image, f"Blobs found: {len(keypoints)} ({detection_method})", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Draw ALL detected blobs for debugging
        for i, kp in enumerate(keypoints):
            cx_temp = int(kp.pt[0]) + x1
            cy_temp = int(kp.pt[1]) + y1
            radius_temp = int(kp.size / 2)
            # Draw all blobs in yellow
            cv2.circle(display_image, (cx_temp, cy_temp), radius_temp, (0, 255, 255), 1)
            cv2.putText(display_image, str(i), (cx_temp + 5, cy_temp + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        # Draw detected dots
        if len(keypoints) > 0:
            # Take the first (most confident) blob
            kp = keypoints[0]
            # Adjust coordinates for crop offset
            cx = int(kp.pt[0]) + x1
            cy = int(kp.pt[1]) + y1
            radius = int(kp.size / 2)
            
            # Draw circle around the dot
            cv2.circle(display_image, (cx, cy), radius, (0, 255, 0), 2)
            
            # Draw crosshair on the dot
            cv2.drawMarker(display_image, (cx, cy), (0, 255, 0), 
                         cv2.MARKER_CROSS, 20, 2)
            
            # Display position
            cv2.putText(display_image, f"Dot: ({cx}, {cy})", (cx + 10, cy - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            cv2.putText(display_image, "No dot detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Show inverted image for debugging
        inverted_colored = cv2.cvtColor(inverted, cv2.COLOR_GRAY2BGR)
        full_inverted = np.zeros((h, w, 3), dtype=np.uint8)
        full_inverted[y1:y2, x1:x2] = inverted_colored
        combined = np.hstack((display_image, full_inverted))
        
        cv2.imshow('Dot Tracker | Black Mask (Press Q to quit)', combined)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("Camera stopped")
