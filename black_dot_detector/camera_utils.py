#!/usr/bin/env python3
"""
Camera Utilities for Intel RealSense D435i
Provides reusable camera interface for capture, streaming, and error handling.
"""

import pyrealsense2 as rs
import numpy as np
import cv2
from typing import Optional, Tuple, Callable, Dict
import logging
import time

logger = logging.getLogger(__name__)


class RealSenseCamera:
    """
    Wrapper class for Intel RealSense D435i camera operations.
    Handles initialization, frame capture, streaming, and cleanup.
    """
    
    def __init__(self, 
                 rgb_resolution: Tuple[int, int] = (1280, 720),
                 depth_resolution: Tuple[int, int] = (640, 480),
                 framerate: int = 30,
                 enable_depth: bool = True,
                 serial_number: Optional[str] = None,
                 auto_exposure: bool = True):
        """
        Initialize RealSense camera configuration.
        
        Args:
            rgb_resolution: RGB stream resolution (width, height)
            depth_resolution: Depth stream resolution (width, height)
            framerate: Frames per second
            enable_depth: Whether to enable depth stream
            serial_number: Specific camera serial number (None for first available)
            auto_exposure: Enable automatic exposure
        """
        self.rgb_resolution = rgb_resolution
        self.depth_resolution = depth_resolution
        self.framerate = framerate
        self.enable_depth = enable_depth
        self.serial_number = serial_number
        self.auto_exposure = auto_exposure
        
        self.pipeline = None
        self.config = None
        self.align = None
        self.is_streaming = False
        
    def initialize(self) -> bool:
        """
        Initialize the camera pipeline.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Initializing RealSense camera...")
            
            # Create pipeline and config
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            
            # Enable specific device if serial number provided
            if self.serial_number:
                self.config.enable_device(self.serial_number)
                logger.info(f"Targeting camera with serial: {self.serial_number}")
            
            # Configure depth stream first if enabled
            if self.enable_depth:
                self.config.enable_stream(
                    rs.stream.depth,
                    self.depth_resolution[0],
                    self.depth_resolution[1],
                    rs.format.z16,
                    self.framerate
                )
                logger.info(f"Depth stream: {self.depth_resolution[0]}x{self.depth_resolution[1]} @ {self.framerate}fps")
            
            # Configure RGB stream
            self.config.enable_stream(
                rs.stream.color,
                self.rgb_resolution[0],
                self.rgb_resolution[1],
                rs.format.bgr8,
                self.framerate
            )
            logger.info(f"RGB stream: {self.rgb_resolution[0]}x{self.rgb_resolution[1]} @ {self.framerate}fps")
            
            # Start streaming
            profile = self.pipeline.start(self.config)
            
            # Get device and sensor info
            device = profile.get_device()
            logger.info(f"Device: {device.get_info(rs.camera_info.name)}")
            logger.info(f"Serial Number: {device.get_info(rs.camera_info.serial_number)}")
            
            # Configure auto-exposure
            if self.auto_exposure:
                color_sensor = device.first_color_sensor()
                if color_sensor.supports(rs.option.enable_auto_exposure):
                    color_sensor.set_option(rs.option.enable_auto_exposure, 1.0)
                    logger.info("Auto-exposure enabled")
            
            # Create align object to align depth frames to color frames
            if self.enable_depth:
                self.align = rs.align(rs.stream.color)
            
            self.is_streaming = True
            logger.info("Camera initialization successful!")
            
            # Let auto-exposure stabilize
            logger.info("Warming up camera...")
            for _ in range(30):
                self.pipeline.wait_for_frames()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize camera: {e}")
            self.cleanup()
            return False
    
    def capture_frame(self, 
                      align_depth: bool = False,
                      timeout_ms: int = 5000) -> Optional[Dict[str, np.ndarray]]:
        """
        Capture a single frame from the camera.
        
        Args:
            align_depth: Align depth frame to color frame
            timeout_ms: Timeout for waiting for frames
            
        Returns:
            Dictionary with 'color' and optionally 'depth' frames, or None on failure
        """
        if not self.is_streaming:
            logger.error("Camera is not streaming. Call initialize() first.")
            return None
        
        try:
            # Wait for frames
            frames = self.pipeline.wait_for_frames(timeout_ms)
            
            # Align depth to color if requested
            if align_depth and self.enable_depth and self.align:
                frames = self.align.process(frames)
            
            # Get color frame
            color_frame = frames.get_color_frame()
            if not color_frame:
                logger.warning("No color frame received")
                return None
            
            # Convert to numpy array
            color_image = np.asanyarray(color_frame.get_data())
            
            result = {'color': color_image}
            
            # Get depth frame if available
            if self.enable_depth:
                depth_frame = frames.get_depth_frame()
                if depth_frame:
                    depth_image = np.asanyarray(depth_frame.get_data())
                    result['depth'] = depth_image
                    
                    # Add depth frame object for distance queries
                    result['depth_frame'] = depth_frame
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to capture frame: {e}")
            return None
    
    def stream_frames(self, 
                      callback: Callable[[Dict[str, np.ndarray]], bool],
                      align_depth: bool = False,
                      display_fps: bool = False) -> None:
        """
        Continuous frame streaming with callback.
        
        Args:
            callback: Function called for each frame. Should return False to stop streaming.
            align_depth: Align depth frame to color frame
            display_fps: Print FPS information
        """
        if not self.is_streaming:
            logger.error("Camera is not streaming. Call initialize() first.")
            return
        
        logger.info("Starting frame stream... Press 'q' in callback to quit.")
        
        frame_count = 0
        start_time = time.time()
        last_fps_time = start_time
        
        try:
            while self.is_streaming:
                # Capture frame
                frame_data = self.capture_frame(align_depth=align_depth)
                
                if frame_data is None:
                    continue
                
                frame_count += 1
                
                # Calculate and display FPS
                if display_fps:
                    current_time = time.time()
                    if current_time - last_fps_time >= 1.0:
                        fps = frame_count / (current_time - start_time)
                        logger.info(f"FPS: {fps:.2f}")
                        last_fps_time = current_time
                
                # Call user callback
                try:
                    should_continue = callback(frame_data)
                    if should_continue is False:
                        logger.info("Callback requested stream stop")
                        break
                except Exception as e:
                    logger.error(f"Error in stream callback: {e}")
                    break
                    
        except KeyboardInterrupt:
            logger.info("Stream interrupted by user")
        except Exception as e:
            logger.error(f"Error during streaming: {e}")
        finally:
            logger.info(f"Stream ended. Total frames: {frame_count}")
    
    def get_depth_at_pixel(self, 
                           depth_frame, 
                           x: int, 
                           y: int) -> Optional[float]:
        """
        Get depth value at specific pixel coordinates.
        
        Args:
            depth_frame: RealSense depth frame object
            x: X coordinate
            y: Y coordinate
            
        Returns:
            Depth in meters, or None if unavailable
        """
        try:
            depth = depth_frame.get_distance(x, y)
            return depth if depth > 0 else None
        except Exception as e:
            logger.error(f"Failed to get depth at pixel ({x}, {y}): {e}")
            return None
    
    def cleanup(self) -> None:
        """Stop the pipeline and release resources."""
        if self.pipeline and self.is_streaming:
            try:
                logger.info("Stopping camera pipeline...")
                self.pipeline.stop()
                self.is_streaming = False
                logger.info("Camera pipeline stopped")
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        if not self.initialize():
            raise RuntimeError("Failed to initialize camera")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
    
    @staticmethod
    def list_available_cameras() -> list:
        """
        List all available RealSense cameras.
        
        Returns:
            List of dictionaries with camera information
        """
        cameras = []
        ctx = rs.context()
        
        for device in ctx.query_devices():
            camera_info = {
                'name': device.get_info(rs.camera_info.name),
                'serial_number': device.get_info(rs.camera_info.serial_number),
                'firmware_version': device.get_info(rs.camera_info.firmware_version),
            }
            cameras.append(camera_info)
        
        return cameras


def test_camera(config: Optional[dict] = None) -> bool:
    """
    Test camera functionality with live preview.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        True if test successful
    """
    logger.info("Testing RealSense camera...")
    
    # Use default config if not provided
    if config is None:
        config = {
            'rgb_resolution': (1280, 720),
            'depth_resolution': (640, 480),
            'framerate': 30,
            'enable_depth': True
        }
    
    try:
        # Initialize camera
        camera = RealSenseCamera(
            rgb_resolution=config.get('rgb_resolution', (1280, 720)),
            depth_resolution=config.get('depth_resolution', (640, 480)),
            framerate=config.get('framerate', 30),
            enable_depth=config.get('enable_depth', True)
        )
        
        if not camera.initialize():
            return False
        
        # Display frames
        def display_callback(frame_data):
            color_image = frame_data['color']
            
            # Add text overlay
            cv2.putText(color_image, "Camera Test - Press 'q' to quit", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display depth if available
            if 'depth' in frame_data:
                depth_image = frame_data['depth']
                # Normalize depth for visualization
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03),
                    cv2.COLORMAP_JET
                )
                
                # Resize depth to match color width for side-by-side display
                h, w = color_image.shape[:2]
                depth_resized = cv2.resize(depth_colormap, (w//2, h//2))
                
                # Display depth in corner
                color_image[0:h//2, w-w//2:w] = depth_resized
            
            cv2.imshow("RealSense Test", color_image)
            
            key = cv2.waitKey(1)
            return key != ord('q')
        
        camera.stream_frames(display_callback, align_depth=True, display_fps=True)
        
        camera.cleanup()
        cv2.destroyAllWindows()
        
        logger.info("Camera test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Camera test failed: {e}")
        return False


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # List available cameras
    print("\nAvailable RealSense Cameras:")
    print("=" * 50)
    cameras = RealSenseCamera.list_available_cameras()
    
    if not cameras:
        print("No RealSense cameras detected!")
        exit(1)
    
    for i, cam in enumerate(cameras, 1):
        print(f"\nCamera {i}:")
        for key, value in cam.items():
            print(f"  {key}: {value}")
    
    print("\n" + "=" * 50)
    print("\nStarting camera test...")
    print("Press 'q' in the camera window to quit\n")
    
    # Run test
    test_camera()
