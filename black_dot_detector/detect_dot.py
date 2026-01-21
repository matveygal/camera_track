#!/usr/bin/env python3
"""
Black Dot Detection Application
Detect black dots from images, batch processing, or live camera feed.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import argparse
import json
from datetime import datetime

from ultralytics import YOLO

from camera_utils import RealSenseCamera
from utils import (
    load_config, setup_logging, ensure_directory,
    load_image, save_image, get_image_files,
    draw_detection_result, resize_image_keep_aspect,
    create_timestamp_filename
)

logger = logging.getLogger(__name__)


class DotDetector:
    """Black dot detection using trained model."""
    
    def __init__(self, config: dict, model_path: Optional[str] = None):
        """
        Initialize detector.
        
        Args:
            config: Configuration dictionary
            model_path: Path to model file (None for auto-detect)
        """
        self.config = config
        self.detection_config = config['detection']
        self.paths = config['paths']
        
        # Load model
        if model_path is None:
            model_path = self._find_best_model()
        
        if model_path is None:
            raise ValueError("No trained model found! Train a model first.")
        
        logger.info(f"Loading model: {model_path}")
        self.model = YOLO(str(model_path))
        
        # Detection parameters
        self.confidence_threshold = self.detection_config['confidence_threshold']
        self.iou_threshold = self.detection_config['iou_threshold']
        self.max_detections = self.detection_config['max_detections']
        
        # Feedback system
        self.feedback_enabled = config['feedback']['enabled']
        self.feedback_data = self._load_feedback()
        
        # Camera
        self.camera = None
        
    def _find_best_model(self) -> Optional[Path]:
        """Find the best trained model."""
        models_dir = Path(self.paths['models_dir'])
        
        # Check for best_model.pt
        best_model = models_dir / "best_model.pt"
        if best_model.exists():
            return best_model
        
        # Look for any .pt files
        pt_files = list(models_dir.glob("**/*.pt"))
        if pt_files:
            # Sort by modification time, newest first
            pt_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            return pt_files[0]
        
        logger.error("No trained model found!")
        return None
    
    def _load_feedback(self) -> dict:
        """Load feedback data."""
        feedback_file = Path(self.paths['feedback_file'])
        
        if feedback_file.exists():
            try:
                with open(feedback_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load feedback: {e}")
        
        return {
            "detections": [],
            "version": "1.0",
            "created": datetime.now().isoformat()
        }
    
    def _save_feedback(self) -> None:
        """Save feedback data."""
        if not self.feedback_enabled:
            return
        
        feedback_file = Path(self.paths['feedback_file'])
        ensure_directory(feedback_file.parent)
        
        self.feedback_data["last_updated"] = datetime.now().isoformat()
        
        try:
            with open(feedback_file, 'w') as f:
                json.dump(self.feedback_data, f, indent=2)
            logger.debug("Feedback saved")
        except Exception as e:
            logger.error(f"Failed to save feedback: {e}")
    
    def detect_single_image(self, 
                          image_path: str,
                          save_result: bool = True,
                          display: bool = True) -> Optional[Dict]:
        """
        Detect dot in a single image.
        
        Args:
            image_path: Path to image file
            save_result: Save annotated image
            display: Display result
            
        Returns:
            Detection result dictionary or None
        """
        logger.info(f"Processing image: {image_path}")
        
        # Load image
        image = load_image(image_path)
        if image is None:
            return None
        
        # Run detection
        results = self.model.predict(
            image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            max_det=self.max_detections,
            verbose=False
        )
        
        # Process results
        detection_result = self._process_detection(image, results[0], Path(image_path).name)
        
        if detection_result is None:
            logger.warning("No dot detected!")
            return None
        
        # Draw result
        annotated_image = draw_detection_result(
            image,
            detection_result['x'],
            detection_result['y'],
            detection_result['confidence'],
            **self.detection_config['display']
        )
        
        # Save result
        if save_result:
            output_dir = Path(self.paths['detections_dir'])
            ensure_directory(output_dir)
            
            output_filename = f"detected_{Path(image_path).stem}.jpg"
            output_path = output_dir / output_filename
            
            save_image(annotated_image, str(output_path))
            logger.info(f"Result saved to: {output_path}")
        
        # Display result
        if display:
            self._display_result(annotated_image, "Detection Result")
        
        # Record feedback
        if self.feedback_enabled:
            self._record_feedback(detection_result)
        
        return detection_result
    
    def detect_batch(self, 
                    images_dir: str,
                    save_results: bool = True) -> List[Dict]:
        """
        Batch process multiple images.
        
        Args:
            images_dir: Directory containing images
            save_results: Save annotated images
            
        Returns:
            List of detection results
        """
        logger.info(f"Batch processing images in: {images_dir}")
        
        # Get image files
        image_files = get_image_files(images_dir)
        
        if not image_files:
            logger.error(f"No images found in {images_dir}")
            return []
        
        logger.info(f"Found {len(image_files)} images to process")
        
        results = []
        
        for img_path in image_files:
            result = self.detect_single_image(
                str(img_path),
                save_result=save_results,
                display=False
            )
            
            if result:
                results.append(result)
        
        # Save batch results summary
        self._save_batch_results(results)
        
        logger.info(f"Batch processing complete. {len(results)}/{len(image_files)} detections successful")
        
        return results
    
    def detect_from_camera_capture(self) -> Optional[Dict]:
        """
        Capture single frame from camera and detect.
        
        Returns:
            Detection result or None
        """
        logger.info("Capturing from camera...")
        
        # Initialize camera
        if not self._initialize_camera():
            return None
        
        try:
            # Capture frame
            frame_data = self.camera.capture_frame()
            
            if frame_data is None:
                logger.error("Failed to capture frame")
                return None
            
            image = frame_data['color']
            
            # Save captured image
            ensure_directory(self.paths['camera_captures'])
            capture_filename = create_timestamp_filename("detection", "jpg")
            capture_path = Path(self.paths['camera_captures']) / capture_filename
            save_image(image, str(capture_path))
            
            # Run detection
            results = self.model.predict(
                image,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                max_det=self.max_detections,
                verbose=False
            )
            
            detection_result = self._process_detection(image, results[0], capture_filename)
            
            if detection_result is None:
                logger.warning("No dot detected in captured image!")
                return None
            
            # Draw and display result
            annotated_image = draw_detection_result(
                image,
                detection_result['x'],
                detection_result['y'],
                detection_result['confidence'],
                **self.detection_config['display']
            )
            
            # Save annotated result
            result_path = Path(self.paths['detections_dir']) / f"detected_{capture_filename}"
            ensure_directory(result_path.parent)
            save_image(annotated_image, str(result_path))
            
            self._display_result(annotated_image, "Camera Detection")
            
            # Record feedback
            if self.feedback_enabled:
                self._record_feedback(detection_result)
            
            return detection_result
            
        finally:
            self._cleanup_camera()
    
    def detect_live_camera(self) -> None:
        """Real-time detection from camera feed."""
        logger.info("Starting live camera detection...")
        
        if not self._initialize_camera():
            return
        
        window_name = "Live Black Dot Detection"
        cv2.namedWindow(window_name)
        
        frame_count = 0
        detection_count = 0
        
        display_config = self.detection_config['display']
        live_config = self.detection_config['live_camera']
        
        logger.info("Live detection started. Press 'q' to quit, 's' to save detection.")
        
        try:
            def process_frame(frame_data):
                nonlocal frame_count, detection_count
                
                frame_count += 1
                image = frame_data['color'].copy()
                
                # Run detection every frame (or skip frames for performance)
                if frame_count % 1 == 0:  # Process every frame
                    results = self.model.predict(
                        image,
                        conf=self.confidence_threshold,
                        iou=self.iou_threshold,
                        max_det=self.max_detections,
                        verbose=False
                    )
                    
                    # Process detection
                    detection = self._process_detection(image, results[0], f"frame_{frame_count}")
                    
                    if detection:
                        detection_count += 1
                        
                        # Draw detection
                        image = draw_detection_result(
                            image,
                            detection['x'],
                            detection['y'],
                            detection['confidence'],
                            **display_config
                        )
                        
                        # Save periodically if enabled
                        if live_config['save_detections'] and \
                           detection_count % live_config['detection_save_interval'] == 0:
                            self._save_detection_frame(image, detection)
                
                # Add FPS overlay
                if live_config['display_fps']:
                    cv2.putText(
                        image,
                        f"Frame: {frame_count} | Detections: {detection_count}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )
                
                # Add instructions
                cv2.putText(
                    image,
                    "Press 'q' to quit | 's' to save",
                    (10, image.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1
                )
                
                # Display
                cv2.imshow(window_name, image)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    return False  # Stop streaming
                elif key == ord('s'):
                    # Save current frame with detection
                    self._save_detection_frame(image, detection if detection else None)
                    logger.info("Detection saved")
                
                return True  # Continue streaming
            
            self.camera.stream_frames(
                process_frame,
                align_depth=False,
                display_fps=False
            )
            
        finally:
            cv2.destroyAllWindows()
            self._cleanup_camera()
            
        logger.info(f"Live detection ended. Processed {frame_count} frames, {detection_count} detections")
    
    def _process_detection(self, 
                          image: np.ndarray,
                          result,
                          image_name: str) -> Optional[Dict]:
        """Process YOLO detection result."""
        if len(result.boxes) == 0:
            return None
        
        # Get first (best) detection
        box = result.boxes[0]
        
        # Get bounding box coordinates
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        
        # Calculate center point
        x = int((x1 + x2) / 2)
        y = int((y1 + y2) / 2)
        
        # Get confidence
        confidence = float(box.conf[0].cpu().numpy())
        
        detection_result = {
            'image_name': image_name,
            'x': x,
            'y': y,
            'confidence': confidence,
            'bbox': [int(x1), int(y1), int(x2), int(y2)],
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Detection: ({x}, {y}) with confidence {confidence:.2%}")
        
        return detection_result
    
    def _save_detection_frame(self, image: np.ndarray, detection: Optional[Dict]) -> None:
        """Save detection frame to disk."""
        output_dir = Path(self.paths['detections_dir'])
        ensure_directory(output_dir)
        
        filename = create_timestamp_filename("live_detection", "jpg")
        output_path = output_dir / filename
        
        save_image(image, str(output_path))
        
        if detection and self.feedback_enabled:
            self._record_feedback(detection)
    
    def _record_feedback(self, detection: Dict) -> None:
        """Record detection in feedback system."""
        self.feedback_data['detections'].append(detection)
        self._save_feedback()
    
    def _save_batch_results(self, results: List[Dict]) -> None:
        """Save batch processing results to CSV/JSON."""
        output_dir = Path(self.paths['detections_dir'])
        ensure_directory(output_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as JSON
        json_path = output_dir / f"batch_results_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Batch results saved to: {json_path}")
        
        # Save as CSV
        import csv
        csv_path = output_dir / f"batch_results_{timestamp}.csv"
        
        if results:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
            
            logger.info(f"CSV results saved to: {csv_path}")
    
    def _display_result(self, image: np.ndarray, window_name: str) -> None:
        """Display detection result and wait for key."""
        # Resize for display if needed
        display_image, _ = resize_image_keep_aspect(image, 1280, 720)
        
        cv2.imshow(window_name, display_image)
        logger.info("Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def _initialize_camera(self) -> bool:
        """Initialize RealSense camera."""
        if self.camera and self.camera.is_streaming:
            return True
        
        camera_config = self.config['camera']
        
        self.camera = RealSenseCamera(
            rgb_resolution=tuple(camera_config['rgb_resolution']),
            framerate=camera_config['framerate'],
            serial_number=camera_config.get('serial_number') or None
        )
        
        return self.camera.initialize()
    
    def _cleanup_camera(self) -> None:
        """Cleanup camera resources."""
        if self.camera:
            self.camera.cleanup()
            self.camera = None


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Black Dot Detection Application")
    parser.add_argument(
        'mode',
        choices=['image', 'batch', 'camera', 'live'],
        help='Detection mode: image (single), batch (directory), camera (capture), live (stream)'
    )
    parser.add_argument(
        '--input',
        type=str,
        help='Input image or directory path (for image/batch modes)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to model file (optional, auto-detect if not provided)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Do not display results (save only)'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save results (display only)'
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
    logger.info("Black Dot Detector - Detection Application")
    logger.info("=" * 50)
    logger.info(f"Mode: {args.mode}")
    
    try:
        # Initialize detector
        detector = DotDetector(config, args.model)
        
        # Run detection based on mode
        if args.mode == 'image':
            if not args.input:
                logger.error("--input required for image mode")
                return
            
            detector.detect_single_image(
                args.input,
                save_result=not args.no_save,
                display=not args.no_display
            )
        
        elif args.mode == 'batch':
            if not args.input:
                logger.error("--input required for batch mode")
                return
            
            detector.detect_batch(
                args.input,
                save_results=not args.no_save
            )
        
        elif args.mode == 'camera':
            detector.detect_from_camera_capture()
        
        elif args.mode == 'live':
            detector.detect_live_camera()
        
        logger.info("Detection completed successfully!")
        
    except Exception as e:
        logger.error(f"Detection failed: {e}", exc_info=True)
        return


if __name__ == "__main__":
    main()
