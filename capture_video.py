#!/usr/bin/env python3
"""
Simple script to record video from Intel RealSense 435i camera.
Click to start recording, click again to stop and save.
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import os
from datetime import datetime


class RealSenseVideoCapture:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Configure RGB stream
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # Start streaming
        self.pipeline.start(self.config)
        
        # Create output directory
        self.output_dir = "captured_videos"
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.recording = False
        self.video_writer = None
        self.current_filename = None
        
    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback to toggle recording on click"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if not self.recording:
                # Start recording
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.current_filename = os.path.join(self.output_dir, f"video_{timestamp}.avi")
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                self.video_writer = cv2.VideoWriter(self.current_filename, fourcc, 30.0, (640, 480))
                self.recording = True
                print(f"Recording started: {self.current_filename}")
            else:
                # Stop recording
                self.recording = False
                if self.video_writer:
                    self.video_writer.release()
                    self.video_writer = None
                print(f"Recording stopped. Video saved: {self.current_filename}")
                self.current_filename = None
            
    def run(self):
        """Main capture loop"""
        window_name = "RealSense Video - Click to Start/Stop Recording"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        print("RealSense camera started.")
        print("Click to start recording, click again to stop and save.")
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
                
                # Save frame if recording
                if self.recording and self.video_writer:
                    self.video_writer.write(color_image)
                
                # Display the image
                display_image = color_image.copy()
                
                # Add text overlay
                if self.recording:
                    cv2.putText(display_image, "RECORDING - Click to stop", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    # Add recording indicator (red circle)
                    cv2.circle(display_image, (620, 20), 10, (0, 0, 255), -1)
                else:
                    cv2.putText(display_image, "Click to start recording | Press 'q' to quit", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow(window_name, display_image)
                
                # Check for quit
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                    
        finally:
            self.cleanup()
            
    def cleanup(self):
        """Stop recording and close everything"""
        if self.recording and self.video_writer:
            self.video_writer.release()
            print(f"Recording stopped. Video saved: {self.current_filename}")
        self.pipeline.stop()
        cv2.destroyAllWindows()
        print("RealSense camera stopped.")


def main():
    capture = RealSenseVideoCapture()
    capture.run()


if __name__ == "__main__":
    main()
