#!/usr/bin/env python3
"""
General utility functions for the Black Dot Detector system.
Includes configuration loading, data management, visualization, and helper functions.
"""

import yaml
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
import shutil

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise


def save_config(config: dict, config_path: str = "config.yaml") -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    try:
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        logger.info(f"Configuration saved to {config_path}")
    except Exception as e:
        logger.error(f"Failed to save configuration: {e}")
        raise


def load_annotations(annotations_file: str) -> dict:
    """
    Load annotations from JSON file.
    
    Args:
        annotations_file: Path to annotations file
        
    Returns:
        Annotations dictionary
    """
    annotations_path = Path(annotations_file)
    
    if not annotations_path.exists():
        logger.info(f"No existing annotations found at {annotations_file}")
        return {"images": [], "version": "1.0", "created": datetime.now().isoformat()}
    
    try:
        with open(annotations_path, 'r') as f:
            annotations = json.load(f)
        logger.info(f"Loaded {len(annotations.get('images', []))} annotations")
        return annotations
    except Exception as e:
        logger.error(f"Failed to load annotations: {e}")
        return {"images": [], "version": "1.0", "created": datetime.now().isoformat()}


def save_annotations(annotations: dict, annotations_file: str) -> None:
    """
    Save annotations to JSON file.
    
    Args:
        annotations: Annotations dictionary
        annotations_file: Path to save annotations
    """
    try:
        annotations_path = Path(annotations_file)
        annotations_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add last modified timestamp
        annotations["last_modified"] = datetime.now().isoformat()
        
        with open(annotations_path, 'w') as f:
            json.dump(annotations, f, indent=2)
        
        logger.info(f"Saved {len(annotations.get('images', []))} annotations to {annotations_file}")
    except Exception as e:
        logger.error(f"Failed to save annotations: {e}")
        raise


def draw_dot_annotation(image: np.ndarray, 
                       x: int, 
                       y: int, 
                       radius: int = 10, 
                       color: Tuple[int, int, int] = (0, 255, 0),
                       thickness: int = 2) -> np.ndarray:
    """
    Draw a dot annotation on an image.
    
    Args:
        image: Input image
        x: X coordinate
        y: Y coordinate
        radius: Circle radius
        color: Circle color (BGR)
        thickness: Line thickness
        
    Returns:
        Image with annotation
    """
    annotated = image.copy()
    
    # Draw circle
    cv2.circle(annotated, (x, y), radius, color, thickness)
    
    # Draw crosshair
    cv2.line(annotated, (x - radius - 5, y), (x + radius + 5, y), color, thickness)
    cv2.line(annotated, (x, y - radius - 5), (x, y + radius + 5), color, thickness)
    
    return annotated


def draw_detection_result(image: np.ndarray,
                         x: int,
                         y: int,
                         confidence: float,
                         show_confidence: bool = True,
                         show_coordinates: bool = True,
                         dot_color: Tuple[int, int, int] = (0, 255, 0),
                         dot_radius: int = 10,
                         text_color: Tuple[int, int, int] = (255, 255, 255),
                         text_size: float = 0.6) -> np.ndarray:
    """
    Draw detection result on image with annotations.
    
    Args:
        image: Input image
        x: X coordinate of detection
        y: Y coordinate of detection
        confidence: Detection confidence score
        show_confidence: Display confidence score
        show_coordinates: Display coordinates
        dot_color: Color for dot marker (BGR)
        dot_radius: Radius of dot marker
        text_color: Color for text (BGR)
        text_size: Font size for text
        
    Returns:
        Annotated image
    """
    result = image.copy()
    
    # Draw dot marker
    result = draw_dot_annotation(result, x, y, dot_radius, dot_color, 2)
    
    # Prepare text
    text_lines = []
    if show_coordinates:
        text_lines.append(f"Position: ({x}, {y})")
    if show_confidence:
        text_lines.append(f"Confidence: {confidence:.2%}")
    
    # Draw text with background
    if text_lines:
        y_offset = y - dot_radius - 15
        for line in text_lines:
            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(
                line, cv2.FONT_HERSHEY_SIMPLEX, text_size, 2
            )
            
            # Draw background rectangle
            cv2.rectangle(
                result,
                (x - text_width // 2 - 5, y_offset - text_height - 5),
                (x + text_width // 2 + 5, y_offset + 5),
                (0, 0, 0),
                -1
            )
            
            # Draw text
            cv2.putText(
                result,
                line,
                (x - text_width // 2, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                text_size,
                text_color,
                2
            )
            
            y_offset -= text_height + 10
    
    return result


def resize_image_keep_aspect(image: np.ndarray, 
                            max_width: int, 
                            max_height: int) -> Tuple[np.ndarray, float]:
    """
    Resize image while maintaining aspect ratio.
    
    Args:
        image: Input image
        max_width: Maximum width
        max_height: Maximum height
        
    Returns:
        Tuple of (resized image, scale factor)
    """
    h, w = image.shape[:2]
    
    # Calculate scale factor
    scale_w = max_width / w
    scale_h = max_height / h
    scale = min(scale_w, scale_h, 1.0)  # Don't upscale
    
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized, scale
    
    return image, 1.0


def create_timestamp_filename(prefix: str = "capture", extension: str = "jpg") -> str:
    """
    Create a filename with timestamp.
    
    Args:
        prefix: Filename prefix
        extension: File extension
        
    Returns:
        Filename string
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    return f"{prefix}_{timestamp}.{extension}"


def ensure_directory(path: str) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_image_files(directory: str, 
                   extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp')) -> List[Path]:
    """
    Get all image files in a directory.
    
    Args:
        directory: Directory path
        extensions: Tuple of valid extensions
        
    Returns:
        List of image file paths
    """
    dir_path = Path(directory)
    
    if not dir_path.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return []
    
    image_files = []
    for ext in extensions:
        image_files.extend(dir_path.glob(f"*{ext}"))
        image_files.extend(dir_path.glob(f"*{ext.upper()}"))
    
    return sorted(image_files)


def load_image(image_path: str) -> Optional[np.ndarray]:
    """
    Load an image from file.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Image as numpy array or None on failure
    """
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return None
        return image
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return None


def save_image(image: np.ndarray, output_path: str) -> bool:
    """
    Save an image to file.
    
    Args:
        image: Image array
        output_path: Output file path
        
    Returns:
        True if successful
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        success = cv2.imwrite(str(output_path), image)
        if success:
            logger.debug(f"Image saved to {output_path}")
        else:
            logger.error(f"Failed to save image to {output_path}")
        return success
    except Exception as e:
        logger.error(f"Error saving image to {output_path}: {e}")
        return False


def setup_logging(log_level: str = "INFO", 
                 log_file: Optional[str] = None,
                 log_to_console: bool = True) -> None:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        log_to_console: Whether to log to console
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create formatters and handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers
    root_logger.handlers = []
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_file}")


def calculate_iou(box1: Tuple[int, int, int, int], 
                  box2: Tuple[int, int, int, int]) -> float:
    """
    Calculate Intersection over Union (IoU) for two bounding boxes.
    
    Args:
        box1: (x1, y1, x2, y2)
        box2: (x1, y1, x2, y2)
        
    Returns:
        IoU value
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection area
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def get_device(device_preference: str = "auto") -> str:
    """
    Get the best available device for PyTorch.
    
    Args:
        device_preference: "auto", "cuda", "mps", or "cpu"
        
    Returns:
        Device string
    """
    import torch
    
    if device_preference == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = "mps"
            logger.info("Using Apple Metal Performance Shaders (MPS)")
        else:
            device = "cpu"
            logger.info("Using CPU")
    else:
        device = device_preference
        logger.info(f"Using device: {device}")
    
    return device


def backup_file(file_path: str, backup_dir: str = "backups") -> Optional[Path]:
    """
    Create a backup of a file with timestamp.
    
    Args:
        file_path: File to backup
        backup_dir: Directory for backups
        
    Returns:
        Path to backup file or None
    """
    try:
        src_path = Path(file_path)
        if not src_path.exists():
            logger.warning(f"File does not exist: {file_path}")
            return None
        
        # Create backup directory
        backup_path = Path(backup_dir)
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # Create backup filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{src_path.stem}_{timestamp}{src_path.suffix}"
        backup_file = backup_path / backup_filename
        
        # Copy file
        shutil.copy2(src_path, backup_file)
        logger.info(f"Backup created: {backup_file}")
        
        return backup_file
        
    except Exception as e:
        logger.error(f"Failed to create backup: {e}")
        return None


if __name__ == "__main__":
    # Test utilities
    setup_logging(log_level="INFO")
    
    print("\nTesting utility functions...")
    print("=" * 50)
    
    # Test config loading
    try:
        config = load_config("config.yaml")
        print("✓ Configuration loaded successfully")
        print(f"  Camera resolution: {config['camera']['rgb_resolution']}")
    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")
    
    # Test timestamp filename
    filename = create_timestamp_filename("test", "jpg")
    print(f"✓ Generated filename: {filename}")
    
    # Test device detection
    try:
        device = get_device("auto")
        print(f"✓ Best device: {device}")
    except Exception as e:
        print(f"✗ Device detection failed: {e}")
    
    print("\nUtility tests completed!")
