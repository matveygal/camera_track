import pyrealsense2 as rs
import numpy as np
import cv2

print("Initializing RealSense camera...")

# Create a context object to manage devices
ctx = rs.context()
devices = ctx.query_devices()

if len(devices) == 0:
    print("No RealSense devices found!")
    exit(1)

print(f"Found {len(devices)} RealSense device(s)")
for i, dev in enumerate(devices):
    print(f"Device {i}: {dev.get_info(rs.camera_info.name)}")
    print(f"  Serial: {dev.get_info(rs.camera_info.serial_number)}")

# Configure streams
pipeline = rs.pipeline()
config = rs.config()

# Enable color and depth streams
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
print("\nStarting streams...")
pipeline.start(config)
print("Streams started! Press 'q' to quit")

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
        
        # Colorize depth for visualization
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), 
            cv2.COLORMAP_JET
        )
        
        # Stack images side by side
        combined = np.hstack((color_image, depth_colormap))
        
        # Display streams
        cv2.imshow('RealSense: Color | Depth (Press Q to quit)', combined)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("Camera stopped")