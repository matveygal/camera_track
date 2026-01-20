import pyrealsense2 as rs
import numpy as np
import cv2

print("Initializing RealSense camera...")

# Configure streams
pipeline = rs.pipeline()
config = rs.config()

# Enable color and depth streams
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)
print("Camera started!")
print("Adjust sliders to detect your black dot")
print("Press 'q' to quit")

# Create window and trackbars for tuning
cv2.namedWindow('Controls')
cv2.createTrackbar('Black Threshold', 'Controls', 50, 255, lambda x: None)
cv2.createTrackbar('Min Area', 'Controls', 10, 1000, lambda x: None)
cv2.createTrackbar('Max Area', 'Controls', 500, 5000, lambda x: None)

try:
    while True:
        # Wait for frames
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue
        
        # Convert to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # Get trackbar values
        black_threshold = cv2.getTrackbarPos('Black Threshold', 'Controls')
        min_area = cv2.getTrackbarPos('Min Area', 'Controls')
        max_area = cv2.getTrackbarPos('Max Area', 'Controls')
        
        # Convert to grayscale
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        
        # Threshold to find black regions (black = low values)
        _, binary = cv2.threshold(gray, black_threshold, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create visualization images
        output = color_image.copy()
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), 
            cv2.COLORMAP_JET
        )
        
        # Filter contours by area and find the dot
        dot_found = False
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                # Get the center of the contour
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Get depth at this point (in millimeters)
                    depth_value = depth_frame.get_distance(cx, cy) * 1000  # Convert to mm
                    
                    # Draw on the image
                    cv2.drawContours(output, [contour], -1, (0, 255, 0), 2)
                    cv2.circle(output, (cx, cy), 5, (0, 0, 255), -1)
                    
                    # Display information
                    text = f"Pos: ({cx}, {cy})"
                    depth_text = f"Depth: {depth_value:.1f}mm"
                    cv2.putText(output, text, (cx + 10, cy - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(output, depth_text, (cx + 10, cy + 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Mark on depth map too
                    cv2.circle(depth_colormap, (cx, cy), 5, (255, 255, 255), -1)
                    
                    dot_found = True
                    
                    # Print to console
                    print(f"Dot at ({cx:3d}, {cy:3d}) | Depth: {depth_value:6.1f}mm", end='\r')
        
        if not dot_found:
            cv2.putText(output, "No dot detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Stack images side by side
        combined = np.hstack((output, depth_colormap))
        
        # Show binary mask in small window
        binary_color = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        binary_resized = cv2.resize(binary_color, (320, 240))
        
        # Display
        cv2.imshow('Tracking: RGB | Depth', combined)
        cv2.imshow('Binary Mask (adjust thresholds)', binary_resized)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("\nCamera stopped")
