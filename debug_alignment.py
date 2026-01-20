import pyrealsense2 as rs
import numpy as np
import cv2

print("=== DEPTH ALIGNMENT DEBUG ===")
print("This shows RGB and depth side-by-side to verify alignment")
print("Press 'q' to quit")
print()

# Configure streams  
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Create alignment object - aligns depth to color
# This transforms depth frame to match color camera's perspective
align_to = rs.stream.color
align = rs.align(align_to)

# Get depth scale (to convert depth units to meters)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print(f"Depth scale: {depth_scale} (depth value * scale = meters)")

try:
    while True:
        # Wait for frames
        frames = pipeline.wait_for_frames()
        
        # Align depth frame to color frame
        aligned_frames = align.process(frames)
        
        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not aligned_depth_frame or not color_frame:
            continue
        
        # Convert to numpy arrays
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # Apply colormap to depth (for visualization)
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), 
            cv2.COLORMAP_JET
        )
        
        # Get depth at center point (for testing)
        h, w = depth_image.shape
        center_x, center_y = w // 2, h // 2
        depth_value = depth_image[center_y, center_x]
        distance_meters = depth_value * depth_scale
        distance_mm = distance_meters * 1000
        
        # Draw crosshair at center
        cv2.drawMarker(color_image, (center_x, center_y), (0, 255, 0),
                      cv2.MARKER_CROSS, 20, 2)
        cv2.drawMarker(depth_colormap, (center_x, center_y), (255, 255, 255),
                      cv2.MARKER_CROSS, 20, 2)
        
        # Display depth at center
        cv2.putText(color_image, f"Center depth: {distance_mm:.1f}mm", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(depth_colormap, f"Center depth: {distance_mm:.1f}mm", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Instructions
        cv2.putText(color_image, "RGB (aligned)", 
                   (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(depth_colormap, "Depth (aligned to RGB)", 
                   (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Stack side by side
        combined = np.hstack((color_image, depth_colormap))
        
        # Display
        cv2.imshow('Alignment Test - RGB | Depth', combined)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("\nAlignment info:")
    print("- Depth frame was transformed to match RGB camera viewpoint")
    print("- Now both frames share the same (x,y) coordinate system")
    print("- A pixel at (x,y) in RGB corresponds to same (x,y) in depth")
    print("- Depth value * depth_scale = distance in meters")
