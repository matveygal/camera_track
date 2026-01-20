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
print("Adjust HSV values if needed using trackbars")

# Create window with trackbars for HSV tuning
cv2.namedWindow('Tracking')

# Default HSV range for black (tune these with trackbars)
cv2.createTrackbar('H Lower', 'Tracking', 0, 179, lambda x: None)
cv2.createTrackbar('H Upper', 'Tracking', 179, 179, lambda x: None)
cv2.createTrackbar('S Lower', 'Tracking', 0, 255, lambda x: None)
cv2.createTrackbar('S Upper', 'Tracking', 255, 255, lambda x: None)
cv2.createTrackbar('V Lower', 'Tracking', 0, 255, lambda x: None)
cv2.createTrackbar('V Upper', 'Tracking', 50, 255, lambda x: None)  # Black = low value

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
        
        # Get current trackbar positions
        h_lower = cv2.getTrackbarPos('H Lower', 'Tracking')
        h_upper = cv2.getTrackbarPos('H Upper', 'Tracking')
        s_lower = cv2.getTrackbarPos('S Lower', 'Tracking')
        s_upper = cv2.getTrackbarPos('S Upper', 'Tracking')
        v_lower = cv2.getTrackbarPos('V Lower', 'Tracking')
        v_upper = cv2.getTrackbarPos('V Upper', 'Tracking')
        
        # Create HSV range
        lower_bound = np.array([h_lower, s_lower, v_lower])
        upper_bound = np.array([h_upper, s_upper, v_upper])
        
        # Create mask for black color
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        # Clean up mask with morphological operations
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours - look for small circular dots, not giant blobs
        dot_contours = []
        for c in contours:
            area = cv2.contourArea(c)
            # Only keep small blobs (the dot, not the entire frame)
            if 10 < area < 500:  # Adjust these if needed
                dot_contours.append(c)
        
        # Only proceed if we found candidate dots
        if len(dot_contours) > 0:
            # Find the largest of the small contours (the dot)
            c = max(dot_contours, key=cv2.contourArea)
            
            # Compute the minimum enclosing circle and centroid
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            
            if M["m00"] != 0:
                center_x = int(M["m10"] / M["m00"])
                center_y = int(M["m01"] / M["m00"])
                
                # Only proceed if the radius meets a minimum size
                if radius > 5:
                    # Draw the circle and centroid
                    cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                    cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                    
                    # Display position
                    cv2.putText(frame, f"Dot: ({center_x}, {center_y})", 
                               (center_x + 10, center_y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
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
