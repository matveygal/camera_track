#!/usr/bin/env python3
"""
Training Pipeline for Black Dot Detection
Trains a model to detect black dot positions using YOLOv8 or custom architectures.
"""

import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import argparse
import yaml
from datetime import datetime
import json
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from ultralytics import YOLO
import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils import (
    load_config, load_annotations, ensure_directory,
    setup_logging, get_device, save_image, load_image
)

logger = logging.getLogger(__name__)


class DotDatasetPreparator:
    """Prepare dataset for training from annotations."""
    
    def __init__(self, config: dict):
        """
        Initialize dataset preparator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.paths = config['paths']
        self.training_config = config['training']
        self.augmentation_config = config['augmentation']
        
        self.annotations = load_annotations(self.paths['annotations_file'])
        self.dataset_dir = Path(self.paths['dataset_dir'])
        
    def create_yolo_format(self) -> bool:
        """
        Create YOLO format dataset from annotations.
        
        Returns:
            True if successful
        """
        logger.info("Creating YOLO format dataset...")
        
        # Create dataset directories
        train_images_dir = self.dataset_dir / "images" / "train"
        val_images_dir = self.dataset_dir / "images" / "val"
        train_labels_dir = self.dataset_dir / "labels" / "train"
        val_labels_dir = self.dataset_dir / "labels" / "val"
        
        for dir_path in [train_images_dir, val_images_dir, train_labels_dir, val_labels_dir]:
            ensure_directory(dir_path)
        
        # Get annotated images
        images = self.annotations.get('images', [])
        
        if not images:
            logger.error("No annotated images found!")
            return False
        
        logger.info(f"Found {len(images)} annotated images")
        
        # Split into train and validation
        train_val_split = self.training_config['train_val_split']
        train_images, val_images = train_test_split(
            images, 
            train_size=train_val_split, 
            random_state=42
        )
        
        logger.info(f"Train: {len(train_images)}, Validation: {len(val_images)}")
        
        # Process training images
        self._process_image_set(train_images, train_images_dir, train_labels_dir, "train")
        
        # Process validation images
        self._process_image_set(val_images, val_images_dir, val_labels_dir, "val")
        
        # Create dataset YAML file for YOLO
        self._create_dataset_yaml()
        
        logger.info("Dataset preparation complete!")
        return True
    
    def _process_image_set(self, 
                          images: List[Dict], 
                          images_dir: Path, 
                          labels_dir: Path,
                          split_name: str) -> None:
        """Process a set of images (train or val)."""
        logger.info(f"Processing {split_name} set...")
        
        for i, img_data in enumerate(tqdm(images, desc=f"Processing {split_name}")):
            # Load image
            img_path = Path(img_data['file_path'])
            
            if not img_path.exists():
                logger.warning(f"Image not found: {img_path}")
                continue
            
            image = load_image(str(img_path))
            if image is None:
                continue
            
            h, w = image.shape[:2]
            
            # Get dot position
            x, y = img_data['dot_position']
            
            # Apply augmentation if enabled and training set
            if self.augmentation_config['enabled'] and split_name == 'train':
                augmented = self._apply_augmentation(image, x, y)
                if augmented:
                    images_aug, positions_aug = augmented
                    
                    # Save original
                    self._save_image_and_label(
                        image, w, h, x, y,
                        images_dir, labels_dir,
                        f"{img_path.stem}_{i:04d}"
                    )
                    
                    # Save augmented versions
                    for j, (img_aug, (x_aug, y_aug)) in enumerate(zip(images_aug, positions_aug)):
                        self._save_image_and_label(
                            img_aug, w, h, x_aug, y_aug,
                            images_dir, labels_dir,
                            f"{img_path.stem}_{i:04d}_aug{j}"
                        )
                else:
                    # Save without augmentation
                    self._save_image_and_label(
                        image, w, h, x, y,
                        images_dir, labels_dir,
                        f"{img_path.stem}_{i:04d}"
                    )
            else:
                # Save without augmentation
                self._save_image_and_label(
                    image, w, h, x, y,
                    images_dir, labels_dir,
                    f"{img_path.stem}_{i:04d}"
                )
    
    def _save_image_and_label(self, 
                             image: np.ndarray,
                             w: int,
                             h: int,
                             x: int,
                             y: int,
                             images_dir: Path,
                             labels_dir: Path,
                             filename: str) -> None:
        """Save image and corresponding YOLO label."""
        # Save image
        img_path = images_dir / f"{filename}.jpg"
        save_image(image, str(img_path))
        
        # Create YOLO label (format: class x_center y_center width height - normalized)
        # For a point detection, we use a small bounding box around the point
        box_size = 0.02  # 2% of image size
        
        x_center = x / w
        y_center = y / h
        
        # Write label file
        label_path = labels_dir / f"{filename}.txt"
        with open(label_path, 'w') as f:
            f.write(f"0 {x_center:.6f} {y_center:.6f} {box_size:.6f} {box_size:.6f}\n")
    
    def _apply_augmentation(self, 
                           image: np.ndarray, 
                           x: int, 
                           y: int) -> Optional[Tuple[List[np.ndarray], List[Tuple[int, int]]]]:
        """
        Apply data augmentation to image and keypoint.
        
        Returns:
            Tuple of (augmented_images, augmented_positions) or None
        """
        h, w = image.shape[:2]
        
        # Define augmentation pipeline
        transform = A.Compose([
            A.Rotate(
                limit=self.augmentation_config['rotation_range'],
                p=0.7
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.7
            ),
            A.HorizontalFlip(p=self.augmentation_config['horizontal_flip']),
            A.VerticalFlip(p=self.augmentation_config['vertical_flip']),
            A.GaussNoise(p=self.augmentation_config['gaussian_noise']),
            A.Blur(blur_limit=3, p=self.augmentation_config['blur']),
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
        
        augmented_images = []
        augmented_positions = []
        
        # Generate multiple augmented versions
        num_augmentations = 3
        
        for _ in range(num_augmentations):
            try:
                augmented = transform(image=image, keypoints=[(x, y)])
                
                aug_image = augmented['image']
                aug_keypoints = augmented['keypoints']
                
                if aug_keypoints:
                    aug_x, aug_y = aug_keypoints[0]
                    
                    # Ensure coordinates are within bounds
                    aug_x = max(0, min(w - 1, int(aug_x)))
                    aug_y = max(0, min(h - 1, int(aug_y)))
                    
                    augmented_images.append(aug_image)
                    augmented_positions.append((aug_x, aug_y))
                    
            except Exception as e:
                logger.warning(f"Augmentation failed: {e}")
                continue
        
        if augmented_images:
            return augmented_images, augmented_positions
        return None
    
    def _create_dataset_yaml(self) -> None:
        """Create YAML file for YOLO training."""
        dataset_yaml = {
            'path': str(self.dataset_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'names': {
                0: 'black_dot'
            },
            'nc': 1  # number of classes
        }
        
        yaml_path = self.dataset_dir / "dataset.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_yaml, f, default_flow_style=False)
        
        logger.info(f"Dataset YAML created: {yaml_path}")


class DotDetectorTrainer:
    """Train black dot detection model."""
    
    def __init__(self, config: dict):
        """
        Initialize trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.training_config = config['training']
        self.paths = config['paths']
        
        self.device = get_device(self.training_config['device'])
        self.dataset_dir = Path(self.paths['dataset_dir'])
        self.models_dir = Path(self.paths['models_dir'])
        
        ensure_directory(self.models_dir)
        ensure_directory(self.paths['training_metrics_dir'])
    
    def train_yolo(self) -> bool:
        """
        Train YOLOv8 model.
        
        Returns:
            True if training successful
        """
        logger.info("Starting YOLOv8 training...")
        
        # Check if dataset exists
        dataset_yaml = self.dataset_dir / "dataset.yaml"
        if not dataset_yaml.exists():
            logger.error("Dataset not prepared! Run dataset preparation first.")
            return False
        
        # Initialize model
        model_size = self.training_config['model_size']
        model_name = f"yolov8{model_size}.pt"
        
        logger.info(f"Loading model: {model_name}")
        model = YOLO(model_name)
        
        # Training parameters
        train_params = {
            'data': str(dataset_yaml),
            'epochs': self.training_config['epochs'],
            'batch': self.training_config['batch_size'],
            'imgsz': self.training_config['image_size'],
            'device': self.device,
            'workers': self.training_config['workers'],
            'patience': self.training_config['patience'],
            'save': True,
            'save_period': self.training_config['save_frequency'],
            'project': str(self.models_dir),
            'name': f"dot_detector_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'exist_ok': True,
            'pretrained': self.training_config['pretrained'],
            'optimizer': 'Adam',
            'lr0': self.training_config['learning_rate'],
            'verbose': True,
        }
        
        # Start training
        logger.info("Training started...")
        logger.info(f"Parameters: {json.dumps(train_params, indent=2)}")
        
        try:
            results = model.train(**train_params)
            
            logger.info("Training completed successfully!")
            logger.info(f"Best model saved to: {model.trainer.best}")
            
            # Copy best model to main models directory
            best_model_path = Path(model.trainer.best)
            final_model_path = self.models_dir / "best_model.pt"
            shutil.copy2(best_model_path, final_model_path)
            logger.info(f"Best model copied to: {final_model_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False
    
    def evaluate_model(self, model_path: Optional[str] = None) -> Dict:
        """
        Evaluate trained model.
        
        Args:
            model_path: Path to model file (None for best model)
            
        Returns:
            Evaluation metrics dictionary
        """
        if model_path is None:
            model_path = self.models_dir / "best_model.pt"
        
        if not Path(model_path).exists():
            logger.error(f"Model not found: {model_path}")
            return {}
        
        logger.info(f"Evaluating model: {model_path}")
        
        model = YOLO(str(model_path))
        
        # Validate on test set
        dataset_yaml = self.dataset_dir / "dataset.yaml"
        metrics = model.val(data=str(dataset_yaml))
        
        # Extract key metrics
        results = {
            'precision': float(metrics.box.p.mean()) if hasattr(metrics.box, 'p') else 0.0,
            'recall': float(metrics.box.r.mean()) if hasattr(metrics.box, 'r') else 0.0,
            'mAP50': float(metrics.box.map50) if hasattr(metrics.box, 'map50') else 0.0,
            'mAP50-95': float(metrics.box.map) if hasattr(metrics.box, 'map') else 0.0,
        }
        
        logger.info("Evaluation Results:")
        for key, value in results.items():
            logger.info(f"  {key}: {value:.4f}")
        
        # Save metrics
        metrics_file = Path(self.paths['training_metrics_dir']) / "evaluation_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train Black Dot Detection Model")
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--prepare-only',
        action='store_true',
        help='Only prepare dataset without training'
    )
    parser.add_argument(
        '--train-only',
        action='store_true',
        help='Only train model (skip dataset preparation)'
    )
    parser.add_argument(
        '--evaluate',
        type=str,
        default=None,
        help='Evaluate model at specified path'
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
    logger.info("Black Dot Detector - Training Pipeline")
    logger.info("=" * 50)
    
    # Prepare dataset
    if not args.train_only and not args.evaluate:
        logger.info("\n" + "=" * 50)
        logger.info("Step 1: Dataset Preparation")
        logger.info("=" * 50)
        
        preparator = DotDatasetPreparator(config)
        if not preparator.create_yolo_format():
            logger.error("Dataset preparation failed!")
            return
        
        if args.prepare_only:
            logger.info("Dataset preparation complete. Exiting.")
            return
    
    # Train model
    if not args.evaluate:
        logger.info("\n" + "=" * 50)
        logger.info("Step 2: Model Training")
        logger.info("=" * 50)
        
        trainer = DotDetectorTrainer(config)
        if not trainer.train_yolo():
            logger.error("Training failed!")
            return
    
    # Evaluate model
    logger.info("\n" + "=" * 50)
    logger.info("Step 3: Model Evaluation")
    logger.info("=" * 50)
    
    trainer = DotDetectorTrainer(config)
    trainer.evaluate_model(args.evaluate)
    
    logger.info("\n" + "=" * 50)
    logger.info("Training pipeline completed!")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
