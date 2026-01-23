#!/usr/bin/env python3
"""
Simple script to capture RGB images from Intel RealSense 435i camera on mouse click.
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import os
from datetime import datetime


class RealSenseCapture:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Configure RGB stream
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # Start streaming
        self.pipeline.start(self.config)
        
        # Create output directory
        self.output_dir = "captured_images"
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.capture_flag = False
        
    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback to capture image on click"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.capture_flag = True
            
    def run(self):
        """Main capture loop"""
        window_name = "RealSense RGB - Click to Capture"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        print("RealSense camera started.")
        print("Click on the window to capture an image.")
        print("Press 'q' to quit.")
        
        try:
            while True:
                # Wait for frames
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                
                if not color_frame:
                    continue
                
                # Convert to numpy array
                color_image = np.asanyarray(color_frame.get_data())
                
                # Display the image
                display_image = color_image.copy()
                
                # Add text overlay
                cv2.putText(display_image, "Click to capture | Press 'q' to quit", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow(window_name, display_image)
                
                # Check if capture was requested
                if self.capture_flag:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = os.path.join(self.output_dir, f"capture_{timestamp}.png")
                    cv2.imwrite(filename, color_image)
                    print(f"Image saved: {filename}")
                    self.capture_flag = False
                
                # Check for quit
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                    
        finally:
            self.cleanup()
            
    def cleanup(self):
        """Stop the pipeline and close windows"""
        self.pipeline.stop()
        cv2.destroyAllWindows()
        print("RealSense camera stopped.")


def main():
    capture = RealSenseCapture()
    capture.run()


if __name__ == "__main__":
    main()
