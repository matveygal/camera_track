#!/usr/bin/env python3
"""
Black Dot Annotation Tool
GUI application for marking black dots on training images with camera capture support.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
import logging
import argparse

from camera_utils import RealSenseCamera
from utils import (
    load_config, load_annotations, save_annotations,
    draw_dot_annotation, get_image_files, load_image, save_image,
    ensure_directory, create_timestamp_filename, setup_logging,
    resize_image_keep_aspect
)

logger = logging.getLogger(__name__)


class DotAnnotator:
    """
    Interactive annotation tool for marking dot positions on images.
    Supports both file loading and camera capture modes.
    """
    
    def __init__(self, config: dict):
        """
        Initialize the annotator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.annotation_config = config['annotation']
        self.camera_config = config['camera']
        self.paths = config['paths']
        
        # Annotation state
        self.annotations = load_annotations(self.paths['annotations_file'])
        self.current_image = None
        self.current_image_path = None
        self.current_dot_position = None
        self.original_size = None
        self.scale_factor = 1.0
        
        # Image list for file mode
        self.image_files = []
        self.current_image_index = 0
        
        # Display settings
        self.display_width = self.annotation_config['display_width']
        self.display_height = self.annotation_config['display_height']
        self.window_name = "Black Dot Annotator"
        
        # Camera
        self.camera = None
        self.camera_mode = False
        
    def initialize_file_mode(self) -> bool:
        """
        Initialize annotation from existing files.
        
        Returns:
            True if files found
        """
        # Load images from both directories
        raw_images = get_image_files(self.paths['raw_images'])
        camera_images = get_image_files(self.paths['camera_captures'])
        
        self.image_files = sorted(raw_images + camera_images)
        
        if not self.image_files:
            logger.warning("No images found in raw_images or camera_captures directories")
            return False
        
        logger.info(f"Found {len(self.image_files)} images to annotate")
        
        # Find first unannotated image
        annotated_paths = {img['file_path'] for img in self.annotations.get('images', [])}
        
        for i, img_path in enumerate(self.image_files):
            if str(img_path) not in annotated_paths:
                self.current_image_index = i
                logger.info(f"Starting from image {i + 1}/{len(self.image_files)}")
                break
        
        return True
    
    def initialize_camera_mode(self) -> bool:
        """
        Initialize camera for capture mode.
        
        Returns:
            True if camera initialized successfully
        """
        logger.info("Initializing camera capture mode...")
        
        self.camera = RealSenseCamera(
            rgb_resolution=tuple(self.camera_config['rgb_resolution']),
            framerate=self.camera_config['framerate'],
            serial_number=self.camera_config.get('serial_number') or None
        )
        
        if not self.camera.initialize():
            logger.error("Failed to initialize camera")
            return False
        
        self.camera_mode = True
        logger.info("Camera initialized successfully")
        return True
    
    def capture_from_camera(self) -> Optional[np.ndarray]:
        """
        Capture a frame from the camera and save it.
        
        Returns:
            Captured image or None
        """
        if not self.camera or not self.camera.is_streaming:
            logger.error("Camera not initialized")
            return None
        
        # Capture frame
        frame_data = self.camera.capture_frame()
        
        if frame_data is None:
            logger.error("Failed to capture frame")
            return None
        
        image = frame_data['color']
        
        # Save captured image
        ensure_directory(self.paths['camera_captures'])
        filename = create_timestamp_filename("camera", "jpg")
        save_path = Path(self.paths['camera_captures']) / filename
        
        if save_image(image, str(save_path)):
            logger.info(f"Captured image saved to {save_path}")
            self.current_image_path = save_path
            return image
        else:
            logger.error("Failed to save captured image")
            return None
    
    def load_current_image(self) -> bool:
        """
        Load the current image from file list.
        
        Returns:
            True if loaded successfully
        """
        if not self.image_files:
            return False
        
        if self.current_image_index >= len(self.image_files):
            logger.info("All images have been processed!")
            return False
        
        img_path = self.image_files[self.current_image_index]
        image = load_image(str(img_path))
        
        if image is None:
            logger.error(f"Failed to load image: {img_path}")
            return False
        
        self.current_image = image
        self.current_image_path = img_path
        self.original_size = image.shape[:2]
        self.current_dot_position = None
        
        # Check if already annotated
        for annotation in self.annotations.get('images', []):
            if annotation['file_path'] == str(img_path):
                x, y = annotation['dot_position']
                self.current_dot_position = (x, y)
                logger.info(f"Loaded existing annotation: ({x}, {y})")
                break
        
        logger.info(f"Loaded image {self.current_image_index + 1}/{len(self.image_files)}: {img_path.name}")
        return True
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for annotation."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Scale coordinates back to original image size
            orig_x = int(x / self.scale_factor)
            orig_y = int(y / self.scale_factor)
            
            self.current_dot_position = (orig_x, orig_y)
            logger.info(f"Dot marked at: ({orig_x}, {orig_y})")
    
    def display_image(self) -> np.ndarray:
        """
        Prepare image for display with annotations.
        
        Returns:
            Display image
        """
        if self.current_image is None:
            return np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)
        
        # Resize for display
        display_image, self.scale_factor = resize_image_keep_aspect(
            self.current_image,
            self.display_width,
            self.display_height
        )
        
        # Draw annotation if exists
        if self.current_dot_position:
            x, y = self.current_dot_position
            # Scale coordinates for display
            display_x = int(x * self.scale_factor)
            display_y = int(y * self.scale_factor)
            
            display_image = draw_dot_annotation(
                display_image,
                display_x,
                display_y,
                radius=self.annotation_config['dot_marker_size'],
                color=tuple(self.annotation_config['marker_color']),
                thickness=2
            )
        
        # Add instructions
        instructions = [
            f"Image: {self.current_image_index + 1}/{len(self.image_files)}" if self.image_files else "Camera Mode",
            "Click to mark dot position",
            f"Keys: [{self.annotation_config['keys']['save_annotation']}]Save [{self.annotation_config['keys']['next_image']}]Next [{self.annotation_config['keys']['previous_image']}]Prev [{self.annotation_config['keys']['skip_image']}]Skip",
            f"      [{self.annotation_config['keys']['capture_camera']}]Capture [Q]Quit"
        ]
        
        y_offset = 25
        for instruction in instructions:
            cv2.putText(
                display_image,
                instruction,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )
            # Add background for better readability
            (text_width, text_height), _ = cv2.getTextSize(
                instruction, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                display_image,
                (8, y_offset - text_height - 2),
                (12 + text_width, y_offset + 2),
                (0, 0, 0),
                -1
            )
            cv2.putText(
                display_image,
                instruction,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )
            y_offset += 25
        
        return display_image
    
    def save_current_annotation(self) -> bool:
        """
        Save the current annotation.
        
        Returns:
            True if saved successfully
        """
        if self.current_dot_position is None:
            logger.warning("No dot position marked!")
            return False
        
        if self.current_image_path is None:
            logger.error("No image path available!")
            return False
        
        # Create annotation entry
        annotation_entry = {
            'file_path': str(self.current_image_path),
            'file_name': self.current_image_path.name,
            'dot_position': self.current_dot_position,
            'image_size': list(self.original_size) if self.original_size else list(self.current_image.shape[:2]),
            'source': 'camera' if self.camera_mode else 'file',
            'annotated_at': create_timestamp_filename("", "").split('.')[0]
        }
        
        # Remove existing annotation for this image if any
        self.annotations['images'] = [
            img for img in self.annotations.get('images', [])
            if img['file_path'] != str(self.current_image_path)
        ]
        
        # Add new annotation
        self.annotations['images'].append(annotation_entry)
        
        # Save to file
        save_annotations(self.annotations, self.paths['annotations_file'])
        
        logger.info(f"Annotation saved: {annotation_entry['file_name']} -> {self.current_dot_position}")
        return True
    
    def run_file_mode(self) -> None:
        """Run annotation in file mode."""
        if not self.initialize_file_mode():
            logger.error("No images found. Please add images to raw_images or camera_captures directories.")
            return
        
        # Load first image
        if not self.load_current_image():
            return
        
        # Create window
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        logger.info("Starting annotation mode. Click on dot positions and press keys to navigate.")
        
        while True:
            # Display current image
            display = self.display_image()
            cv2.imshow(self.window_name, display)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(self.annotation_config['keys']['quit']) or key == 27:  # q or ESC
                logger.info("Quitting annotation mode")
                break
            
            elif key == ord(self.annotation_config['keys']['save_annotation']):  # s
                if self.save_current_annotation():
                    # Move to next image
                    self.current_image_index += 1
                    if not self.load_current_image():
                        logger.info("All images annotated!")
                        break
            
            elif key == ord(self.annotation_config['keys']['next_image']):  # n
                self.current_image_index += 1
                if not self.load_current_image():
                    self.current_image_index = len(self.image_files) - 1
                    logger.info("Reached end of image list")
            
            elif key == ord(self.annotation_config['keys']['previous_image']):  # p
                self.current_image_index = max(0, self.current_image_index - 1)
                self.load_current_image()
            
            elif key == ord(self.annotation_config['keys']['skip_image']):  # k
                logger.info(f"Skipped image: {self.current_image_path.name}")
                self.current_image_index += 1
                if not self.load_current_image():
                    logger.info("All images processed!")
                    break
        
        cv2.destroyAllWindows()
    
    def run_camera_mode(self) -> None:
        """Run annotation in camera capture mode."""
        if not self.initialize_camera_mode():
            logger.error("Failed to initialize camera")
            return
        
        # Create window
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        logger.info("Camera mode started. Press 'c' to capture, click to annotate, 's' to save.")
        
        capturing = True
        
        try:
            while True:
                if capturing:
                    # Show live camera feed
                    frame_data = self.camera.capture_frame()
                    
                    if frame_data:
                        display = frame_data['color'].copy()
                        
                        # Resize for display
                        display, _ = resize_image_keep_aspect(
                            display, self.display_width, self.display_height
                        )
                        
                        # Add instructions
                        cv2.putText(
                            display, "Press 'C' to capture image",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                        )
                        cv2.putText(
                            display, "Press 'Q' to quit",
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                        )
                        
                        cv2.imshow(self.window_name, display)
                else:
                    # Show captured image for annotation
                    display = self.display_image()
                    cv2.imshow(self.window_name, display)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # q or ESC
                    break
                
                elif key == ord('c') and capturing:  # Capture
                    image = self.capture_from_camera()
                    if image is not None:
                        self.current_image = image
                        self.original_size = image.shape[:2]
                        self.current_dot_position = None
                        capturing = False
                        logger.info("Image captured. Click to annotate, then press 's' to save.")
                
                elif key == ord('s') and not capturing:  # Save annotation
                    if self.save_current_annotation():
                        capturing = True
                        logger.info("Annotation saved. Ready to capture next image.")
                
                elif key == ord('c') and not capturing:  # Return to capture without saving
                    capturing = True
                    logger.info("Discarded current capture. Ready for new capture.")
        
        finally:
            if self.camera:
                self.camera.cleanup()
            cv2.destroyAllWindows()
    
    def run(self, mode: str = "file") -> None:
        """
        Run the annotator.
        
        Args:
            mode: "file" or "camera"
        """
        if mode == "camera":
            self.run_camera_mode()
        else:
            self.run_file_mode()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Black Dot Annotation Tool")
    parser.add_argument(
        '--mode',
        type=str,
        choices=['file', 'camera'],
        default='file',
        help='Annotation mode: file (annotate existing images) or camera (capture and annotate)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    setup_logging(
        log_level=config['logging']['level'],
        log_file=config['logging']['log_file'] if config['logging']['save_logs'] else None
    )
    
    logger.info("=" * 50)
    logger.info("Black Dot Annotator")
    logger.info("=" * 50)
    logger.info(f"Mode: {args.mode}")
    
    # Create annotator and run
    annotator = DotAnnotator(config)
    annotator.run(mode=args.mode)
    
    logger.info("Annotation session completed")


if __name__ == "__main__":
    main()
