import cv2
import numpy as np
import pyrealsense2 as rs

def stream_realsense():
    """Stream RGB video from Intel RealSense camera"""
    
    # Configure RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Enable RGB stream only (depth sensors are broken)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    try:
        # Start streaming
        profile = pipeline.start(config)
        print("RealSense camera started successfully")
        print("RGB stream: 640x480 @ 30fps")
        print("Press 'q' to quit")
        
        while True:
            # Wait for frames
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            
            if not color_frame:
                continue
            
            # Convert to numpy array
            frame = np.asanyarray(color_frame.get_data())
            
            # Display the frame
            cv2.imshow('RealSense RGB Stream', frame)
            
            # Quit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        # Cleanup
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Stream stopped")


if __name__ == "__main__":
    stream_realsense()
