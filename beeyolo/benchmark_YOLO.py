#!/usr/bin/env python3
"""
YOLO Model Benchmarking Script

This script is designed to be used step-by-step in a notebook to:
1. Select models and datasets
2. Process images with class-specific evaluation
3. Create organized output folders
4. Generate performance figures
5. Create bounding box visualization grids
6. Compare performance across datasets
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from tqdm import tqdm
import cv2
from ultralytics import YOLO
import torch
from collections import defaultdict
# matplotlib and seaborn imports removed - no more figure generation
from datetime import datetime
import pandas as pd
import shutil


class YOLOBenchmark:
    """YOLO benchmark class for step-by-step notebook usage."""
    
    def __init__(self, models_dir: str = "models", datasets_dir: str = "datasets", 
                 benchmarks_dir: str = "benchmarks"):
        """
        Initialize the benchmark.
        
        Args:
            models_dir: Directory containing YOLO model files
            datasets_dir: Directory containing dataset folders
            benchmarks_dir: Directory for benchmark outputs
        """
        self.models_dir = Path(models_dir)
        self.datasets_dir = Path(datasets_dir)
        self.benchmarks_dir = Path(benchmarks_dir)
        self.results = {}
        self.selected_model = None
        self.selected_datasets = []
        
        # Ensure directories exist
        if not self.models_dir.exists():
            raise ValueError(f"Models directory {models_dir} does not exist")
        if not self.datasets_dir.exists():
            raise ValueError(f"Datasets directory {datasets_dir} does not exist")
        
        # Create benchmarks directory
        self.benchmarks_dir.mkdir(exist_ok=True)
        (self.benchmarks_dir / "temp_validation").mkdir(exist_ok=True)
    
    # ============================================================================
    # STEP 1: MODEL AND DATASET SELECTION
    # ============================================================================
    
    def get_available_models(self) -> List[str]:
        """Get list of available YOLO models."""
        model_files = list(self.models_dir.glob("*.pt"))
        return [f.stem for f in model_files]
    
    def get_available_datasets(self) -> List[str]:
        """Get list of available datasets."""
        dataset_dirs = [d for d in self.datasets_dir.iterdir() 
                       if d.is_dir() and (d / "data.yaml").exists()]
        return [d.name for d in dataset_dirs]
    
    def get_available_models_dict(self) -> Dict[str, str]:
        """Get dictionary of available models with names as keys."""
        models = self.get_available_models()
        return {name: name for name in models}
    
    def get_available_datasets_dict(self) -> Dict[str, str]:
        """Get dictionary of available datasets with names as keys."""
        datasets = self.get_available_datasets()
        return {name: name for name in datasets}
    
    def get_models_and_datasets_info(self) -> Dict[str, Dict]:
        """Get comprehensive information about available models and datasets."""
        info = {
            'models': {},
            'datasets': {}
        }
        
        # Get models info
        models = self.get_available_models()
        for model in models:
            model_path = self.models_dir / f"{model}.pt"
            if model_path.exists():
                file_size = model_path.stat().st_size / (1024 * 1024)  # MB
                info['models'][model] = {
                    'name': model,
                    'file_size_mb': round(file_size, 1),
                    'path': str(model_path)
                }
        
        # Get datasets info
        datasets = self.get_available_datasets()
        for dataset in datasets:
            try:
                config = self.load_dataset_info(dataset)
                image_paths_val, label_paths_val = self.get_dataset_images_and_labels(dataset, "val")
                image_paths_all, label_paths_all = self.get_dataset_images_and_labels(dataset, "all")
                
                info['datasets'][dataset] = {
                    'name': dataset,
                    'classes': config['names'],
                    'num_classes': config['nc'],
                    'val_images': len(image_paths_val),
                    'val_labels': len(label_paths_val),
                    'all_images': len(image_paths_all),
                    'all_labels': len(label_paths_all)
                }
            except Exception as e:
                info['datasets'][dataset] = {
                    'name': dataset,
                    'error': str(e)
                }
        
        return info
    
    def list_models_with_info(self) -> None:
        """List available models with additional information."""
        models = self.get_available_models()
        if not models:
            print("❌ No models found!")
            return
        
        print("📁 Available Models:")
        for i, model in enumerate(models, 1):
            model_path = self.models_dir / f"{model}.pt"
            if model_path.exists():
                file_size = model_path.stat().st_size / (1024 * 1024)  # Convert to MB
                print(f"  {i}. {model} ({file_size:.1f} MB)")
            else:
                print(f"  {i}. {model} (file not found)")
    
    def list_datasets_with_info(self) -> None:
        """List available datasets with additional information."""
        datasets = self.get_available_datasets()
        if not datasets:
            print("❌ No datasets found!")
            return
        
        print("📊 Available Datasets:")
        for i, dataset in enumerate(datasets, 1):
            try:
                config = self.load_dataset_info(dataset)
                print(f"  {i}. {dataset}")
                print(f"     Classes: {config['nc']} - {', '.join(config['names'])}")
                
                # Get dataset statistics
                image_paths, label_paths = self.get_dataset_images_and_labels(dataset, "val")
                print(f"     Val Images: {len(image_paths)}, Val Labels: {len(label_paths)}")
                
                image_paths_all, label_paths_all = self.get_dataset_images_and_labels(dataset, "all")
                print(f"     All Images: {len(image_paths_all)}, All Labels: {len(label_paths_all)}")
                print()
                
            except Exception as e:
                print(f"  {i}. {dataset} (Error loading: {e})")
    
    def select_model_by_name(self, model_name: str) -> bool:
        """
        Select a model by name (case-insensitive).
        
        Args:
            model_name: Name of the model (can be partial match)
        
        Returns:
            True if model exists and is selected
        """
        available_models = self.get_available_models()
        
        # Exact match first
        if model_name in available_models:
            return self.select_model(model_name)
        
        # Partial match
        matching_models = [m for m in available_models if model_name.lower() in m.lower()]
        
        if len(matching_models) == 1:
            print(f"Found matching model: {matching_models[0]}")
            return self.select_model(matching_models[0])
        elif len(matching_models) > 1:
            print(f"Multiple matches found for '{model_name}':")
            for m in matching_models:
                print(f"  - {m}")
            print("Please use the exact model name.")
            return False
        else:
            print(f"❌ No model found matching '{model_name}'")
            print(f"Available models: {available_models}")
            return False
    
    def select_datasets_by_names(self, dataset_names: List[str]) -> List[str]:
        """
        Select datasets by names (case-insensitive, supports partial matches).
        
        Args:
            dataset_names: List of dataset names (can be partial matches)
        
        Returns:
            List of successfully selected datasets
        """
        available_datasets = self.get_available_datasets()
        selected = []
        
        for name in dataset_names:
            # Exact match first
            if name in available_datasets:
                selected.append(name)
                print(f"✓ Dataset selected: {name}")
                continue
            
            # Partial match
            matching_datasets = [d for d in available_datasets if name.lower() in d.lower()]
            
            if len(matching_datasets) == 1:
                selected.append(matching_datasets[0])
                print(f"✓ Dataset selected: {matching_datasets[0]} (matched '{name}')")
            elif len(matching_datasets) > 1:
                print(f"Multiple matches found for '{name}':")
                for d in matching_datasets:
                    print(f"  - {d}")
                print("Please use the exact dataset name.")
            else:
                print(f"❌ No dataset found matching '{name}'")
        
        self.selected_datasets = selected
        return selected
    
    def select_model(self, model_name: str) -> bool:
        """
        Select a model to use for benchmarking.
        
        Args:
            model_name: Name of the model (without .pt extension)
        
        Returns:
            True if model exists and is selected
        """
        model_path = self.models_dir / f"{model_name}.pt"
        if model_path.exists():
            self.selected_model = model_name
            print(f"✓ Model selected: {model_name}")
            return True
        else:
            print(f"❌ Model {model_name} not found")
            return False
    
    def select_datasets(self, dataset_names: List[str]) -> List[str]:
        """
        Select datasets to benchmark.
        
        Args:
            dataset_names: List of dataset names to select
        
        Returns:
            List of successfully selected datasets
        """
        available_datasets = self.get_available_datasets()
        selected = []
        
        for name in dataset_names:
            if name in available_datasets:
                selected.append(name)
                print(f"✓ Dataset selected: {name}")
            else:
                print(f"❌ Dataset {name} not found")
        
        self.selected_datasets = selected
        return selected
    

    
    def select_data_mode(self, mode: str) -> bool:
        """
        Select the data mode for evaluation.
        
        Args:
            mode: "val" for validation only, "train" for training only, "all" for train+val combined
        
        Returns:
            True if mode selection is valid
        """
        valid_modes = ["val", "train", "all"]
        if mode not in valid_modes:
            print(f"❌ Invalid mode: {mode}")
            print(f"Valid modes: {valid_modes}")
            return False
        
        self.selected_mode = mode
        
        if mode == "val":
            print("✓ Data mode selected: Validation data only")
        elif mode == "train":
            print("✓ Data mode selected: Training data only")
        elif mode == "all":
            print("✓ Data mode selected: All data (train + validation)")
        
        return True
    
    def get_data_mode_info(self) -> Dict[str, str]:
        """Get information about available data modes."""
        return {
            "val": "Validation data only - standard evaluation on unseen data",
            "train": "Training data only - evaluate on data the model was trained on",
            "all": "All data combined - comprehensive evaluation on entire dataset"
        }
    
    # ============================================================================
    # STEP 2: DATASET INFORMATION AND PROCESSING
    # ============================================================================
    
    def load_dataset_info(self, dataset_name: str) -> Dict:
        """Load dataset configuration from data.yaml."""
        print(f"    load_dataset_info called with dataset_name: '{dataset_name}'")
        dataset_path = self.datasets_dir / dataset_name
        yaml_path = dataset_path / "data.yaml"
        print(f"    Looking for yaml at: {yaml_path}")
        
        if not yaml_path.exists():
            # Try to find data.yaml in subdirectories
            for subdir in dataset_path.iterdir():
                if subdir.is_dir():
                    sub_yaml = subdir / "data.yaml"
                    if sub_yaml.exists():
                        print(f"    Found data.yaml in subdirectory: {sub_yaml}")
                        yaml_path = sub_yaml
                        break
            else:
                raise ValueError(f"data.yaml not found in {dataset_path} or its subdirectories")
        
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def get_dataset_images_and_labels(self, dataset_name: str, mode: str = "val") -> Tuple[List[Path], List[Path]]:
        """
        Get image and label paths for a dataset.
        
        Args:
            dataset_name: Name of the dataset folder
            mode: "val" for validation only, "all" for train+val combined
        
        Returns:
            Tuple of (image_paths, label_paths)
        """
        dataset_path = self.datasets_dir / dataset_name
        config = self.load_dataset_info(dataset_name)
        
        image_paths = []
        label_paths = []
        
        if mode == "val":
            # Use only validation data
            val_images_dir = dataset_path / config['val']
            val_labels_dir = dataset_path / "val" / "labels"
            
            if val_images_dir.exists():
                image_paths.extend(list(val_images_dir.glob("*.png")))
                image_paths.extend(list(val_images_dir.glob("*.jpg")))
                image_paths.extend(list(val_images_dir.glob("*.jpeg")))
            
            if val_labels_dir.exists():
                label_paths.extend(list(val_labels_dir.glob("*.txt")))
                
        elif mode == "all":
            # Check if 'all' directory exists and use it if available
            all_dir = dataset_path / "all"
            if all_dir.exists() and (all_dir / "data.yaml").exists():
                print(f"    Using existing 'all' directory for image/label collection: {all_dir}")
                all_images_dir = all_dir / "images"
                all_labels_dir = all_dir / "labels"
                
                if all_images_dir.exists():
                    image_paths.extend(list(all_images_dir.glob("*.png")))
                    image_paths.extend(list(all_images_dir.glob("*.jpg")))
                    image_paths.extend(list(all_images_dir.glob("*.jpeg")))
                
                if all_labels_dir.exists():
                    label_paths.extend(list(all_labels_dir.glob("*.txt")))
            else:
                print(f"    'all' directory not found, manually combining train and val data")
                # Use both train and validation data
                train_images_dir = dataset_path / config['train']
                train_labels_dir = dataset_path / "train" / "labels"
                val_images_dir = dataset_path / config['val']
                val_labels_dir = dataset_path / "val" / "labels"
                
                # Add training images
                if train_images_dir.exists():
                    image_paths.extend(list(train_images_dir.glob("*.png")))
                    image_paths.extend(list(train_images_dir.glob("*.jpg")))
                    image_paths.extend(list(train_images_dir.glob("*.jpeg")))
                
                if train_labels_dir.exists():
                    label_paths.extend(list(train_labels_dir.glob("*.txt")))
                
                # Add validation images
                if val_images_dir.exists():
                    image_paths.extend(list(val_images_dir.glob("*.png")))
                    image_paths.extend(list(val_images_dir.glob("*.jpg")))
                    image_paths.extend(list(val_images_dir.glob("*.jpeg")))
                
                if val_labels_dir.exists():
                    label_paths.extend(list(val_labels_dir.glob("*.txt")))
        
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'val' or 'all'")
        
        # Sort paths for consistent ordering
        image_paths.sort()
        label_paths.sort()
        
        return image_paths, label_paths
    
    # ============================================================================
    # STEP 3: CORE EVALUATION FUNCTIONS
    # ============================================================================
    
    def load_ground_truth_labels(self, label_path: Path, num_classes: int) -> List[List]:
        """Load ground truth labels from a YOLO format label file."""
        if not label_path.exists():
            return []
        
        labels = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    labels.append([class_id, x_center, y_center, width, height])
        
        return labels
    
    def convert_yolo_to_absolute(self, yolo_coords: List, img_width: int, img_height: int) -> List:
        """Convert YOLO normalized coordinates to absolute pixel coordinates."""
        class_id, x_center, y_center, width, height = yolo_coords
        
        x_center_abs = x_center * img_width
        y_center_abs = y_center * img_height
        width_abs = width * img_width
        height_abs = height * img_height
        
        return [class_id, x_center_abs, y_center_abs, width_abs, height_abs]
    
    def calculate_iou(self, box1: List, box2: List) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
        try:
            if len(box1) < 4 or len(box2) < 4:
                print(f"Debug IoU: Box1 has {len(box1)} elements, Box2 has {len(box2)} elements")
                return 0.0
            
            x1, y1, w1, h1 = box1[1:5]
            x2, y2, w2, h2 = box2[1:5]
            
            if not all(isinstance(x, (int, float)) for x in [x1, y1, w1, h1, x2, y2, w2, h2]):
                print(f"Debug IoU: Non-numeric coordinates detected")
                print(f"Debug IoU: Box1 types: {[type(x) for x in [x1, y1, w1, h1]]}")
                print(f"Debug IoU: Box2 types: {[type(x) for x in [x2, y2, w2, h2]]}")
                print(f"Debug IoU: Box1 values: {[x1, y1, w1, h1]}")
                print(f"Debug IoU: Box2 values: {[x2, y2, w2, h2]}")
                return 0.0
            
            # Debug: Print the coordinates being compared
            print(f"Debug IoU: Box1: x={x1:.1f}, y={y1:.1f}, w={w1:.1f}, h={h1:.1f}")
            print(f"Debug IoU: Box2: x={x2:.1f}, y={y2:.1f}, w={w2:.1f}, h={h2:.1f}")
            
            # Convert to corner coordinates
            x1_min, x1_max = x1 - w1/2, x1 + w1/2
            y1_min, y1_max = y1 - h1/2, y1 + h1/2
            x2_min, x2_max = x2 - w2/2, x2 + w2/2
            y2_min, y2_max = y2 - h2/2, y2 + h2/2
            
            print(f"Debug IoU: Box1 corners: ({x1_min:.1f}, {y1_min:.1f}) to ({x1_max:.1f}, {y1_max:.1f})")
            print(f"Debug IoU: Box2 corners: ({x2_min:.1f}, {y2_min:.1f}) to ({x2_max:.1f}, {y2_max:.1f})")
            
            # Calculate intersection
            x_left = max(x1_min, x2_min)
            y_top = max(y1_min, y2_min)
            x_right = min(x1_max, x2_max)
            y_bottom = min(y1_max, y2_max)
            
            print(f"Debug IoU: Intersection: ({x_left:.1f}, {y_top:.1f}) to ({x_right:.1f}, {y_bottom:.1f})")
            
            if x_right < x_left or y_bottom < y_top:
                print(f"Debug IoU: No intersection - boxes don't overlap")
                return 0.0
            
            intersection = (x_right - x_left) * (y_bottom - y_top)
            
            # Calculate union
            area1 = w1 * h1
            area2 = w2 * h2
            union = area1 + area2 - intersection
            
            iou = intersection / union if union > 0 else 0.0
            print(f"Debug IoU: Areas - Box1: {area1:.1f}, Box2: {area2:.1f}, Intersection: {intersection:.1f}, Union: {union:.1f}, IoU: {iou:.3f}")
            
            return iou
            
        except Exception as e:
            print(f"Debug IoU: Exception: {e}")
            return 0.0
    
    def evaluate_model_on_dataset(self, model_path: str, dataset_name: str, 
                                mode: str = "val", conf_threshold: float = 0.5, 
                                iou_threshold: float = 0.5) -> Dict:
        """
        Evaluate a YOLO model on a specific dataset using Ultralytics built-in validation.
        
        Args:
            model_path: Path to the YOLO model file
            dataset_name: Name of the dataset folder
            mode: "val" for validation only, "all" for train+val combined
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for matching detections
        
        Returns:
            Dictionary containing evaluation metrics and visualization data
        """
        print(f"Evaluating model {model_path} on dataset {dataset_name} ({mode} mode)")
        print(f"DEBUG: evaluate_model_on_dataset called with dataset_name='{dataset_name}'")
        
        # Load model
        model = YOLO(model_path)
        print(f"✓ Model loaded: {model_path}")
        
        # Check model's class names
        if hasattr(model, 'names'):
            print(f"Model class names: {model.names}")
        else:
            print("Warning: Model doesn't have class names attribute")
        
        # Load dataset info
        print(f"DEBUG: About to call load_dataset_info with dataset_name='{dataset_name}'")
        config = self.load_dataset_info(dataset_name)
        num_classes = config['nc']
        class_names = config['names']
        print(f"Dataset: {dataset_name}")
        print(f"Classes: {class_names}")
        print(f"Mode: {mode}")
        
        # Create a temporary data.yaml for the selected mode
        print(f"  Creating temp data.yaml for dataset: '{dataset_name}', mode: '{mode}'")
        print(f"DEBUG: About to call _create_temp_data_yaml with dataset_name='{dataset_name}'")
        temp_data_yaml, dataset_config = self._create_temp_data_yaml(dataset_name, mode)
        
        try:
            # Use Ultralytics built-in validation
            print("Running Ultralytics validation...")
            temp_validation_dir = self.benchmarks_dir / "temp_validation"
            print(f"  Validation results will be saved to: {temp_validation_dir}")
            
            results = model.val(
                data=str(temp_data_yaml),
                split='val',  # Always use 'val' split since we control the data
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False,
                save_txt=True,  # Save predictions for analysis
                save_conf=True,  # Save confidence scores
                project=str(temp_validation_dir),
                name=f'{dataset_name}_{mode}',
                exist_ok=True
            )
            
            # Always evaluate all classes - no single-class filtering
            
            print("✓ Validation completed successfully")
            
            # Extract metrics from results
            metrics = results.results_dict
            
            # Debug: Print available metrics
            print(f"Available Ultralytics metrics: {list(metrics.keys())}")
            print(f"Sample metrics: {dict(list(metrics.items())[:5])}")
            
            # Convert to our expected format
            # Get the actual counts from the dataset
            # Use the config returned from _create_temp_data_yaml
            images, labels = self.get_dataset_images_and_labels(dataset_name, mode)
            
            results_dict = {
                'model_path': model_path,
                'dataset_name': dataset_name,
                'mode': mode,
                'total_images': len(images),  # Actual count of images
                'total_ground_truth': sum(len(self.load_ground_truth_labels(Path(label), num_classes)) for label in labels),  # Total GT boxes
                'total_detections': 0,  # Will be calculated from predictions if needed
                'total_correct': 0,  # Will be calculated from predictions if needed
                'overall_precision': float(metrics.get('metrics/precision(B)', 0.0)),
                'overall_recall': float(metrics.get('metrics/recall(B)', 0.0)),
                'overall_f1_score': float(metrics.get('metrics/mAP50(B)', 0.0)),
                'overall_map50': float(metrics.get('metrics/mAP50(B)', 0.0)),
                'overall_map50_95': float(metrics.get('metrics/mAP50-95(B)', 0.0)),
                'conf_threshold': float(conf_threshold),
                'iou_threshold': float(iou_threshold),
                'ultralytics_metrics': metrics,
                'processing_stats': {
                    'processed_images': len(images),
                    'skipped_images': 0,
                    'skipped_no_labels': 0,
                    'skipped_no_target_class': 0
                }
            }
            
            
            print(f"  ✓ Benchmark evaluation completed successfully")
            print(f"  Returning results for {dataset_name}")
            
            return results_dict
            
        finally:
            # Clean up temporary files
            try:
                if 'temp_data_yaml' in locals() and temp_data_yaml.exists():
                    print(f"  Cleaning up temporary file: {temp_data_yaml}")
                    temp_data_yaml.unlink()
                if 'temp_data_yaml' in locals():
                    temp_dir = temp_data_yaml.parent
                    if temp_dir.exists():
                        print(f"  Cleaning up temporary directory: {temp_dir}")
                        import shutil
                        shutil.rmtree(temp_dir)
                        print(f"  ✓ Temporary files cleaned up successfully")
                
                # Clean up temp_validation folder after creating results
                temp_validation_dir = self.benchmarks_dir / "temp_validation"
                if temp_validation_dir.exists():
                    print(f"  Cleaning up temp_validation directory: {temp_validation_dir}")
                    shutil.rmtree(temp_validation_dir)
                    print(f"  ✓ Temp validation directory cleaned up successfully")
                    
            except Exception as e:
                print(f"Warning: Error during cleanup: {e}")
                print(f"  This is not critical - benchmark results are still valid")
                # Cleanup errors don't affect benchmark results
    
    def _create_temp_data_yaml(self, dataset_name: str, mode: str) -> tuple[Path, Dict]:
        """Create a temporary data.yaml file for the selected mode."""
        print(f"    _create_temp_data_yaml called with dataset_name: '{dataset_name}', mode: '{mode}'")
        print(f"    DEBUG: _create_temp_data_yaml received dataset_name='{dataset_name}'")
        import tempfile
        
        # Create temporary directory
        temp_dir = Path(tempfile.mkdtemp())
        temp_data_yaml = temp_dir / "temp_data.yaml"
        
        print(f"  Creating temporary data.yaml at: {temp_data_yaml}")
        
        # Load original dataset config
        print(f"  Loading dataset config for: {dataset_name}")
        print(f"  Dataset path: {self.datasets_dir / dataset_name}")
        print(f"  DEBUG: About to call load_dataset_info with dataset_name='{dataset_name}'")
        print(f"  DEBUG: dataset_name value is still '{dataset_name}'")
        config = self.load_dataset_info(dataset_name)
        print(f"  ✓ Dataset config loaded successfully")
        
        # Always use multi-class evaluation - no single-class filtering
        if mode == "val":
            # Use only validation data
            new_config = {
                'path': str(self.datasets_dir / dataset_name),
                'train': None,  # No training data
                'val': config['val'],
                'nc': config['nc'],
                'names': config['names']
            }
        elif mode == "train":
            # Use only training data
            new_config = {
                'path': str(self.datasets_dir / dataset_name),
                'train': config['train'],
                'val': None,  # No validation data
                'nc': config['nc'],
                'names': config['names']
            }
        elif mode == "all":
            # Check if 'all' directory exists and use it if available
            all_dir = self.datasets_dir / dataset_name / "all"
            if all_dir.exists() and (all_dir / "data.yaml").exists():
                print(f"  Using existing 'all' directory: {all_dir}")
                new_config = {
                    'path': str(all_dir),
                    'train': None,
                    'val': 'images',  # Point to the all/images directory
                    'nc': config['nc'],
                    'names': config['names']
                }
            else:
                print(f"  'all' directory not found, manually combining train and val data")
                # Combine train and val into a single dataset
                # Create a temporary combined directory
                combined_dir = temp_dir / "combined"
                combined_dir.mkdir(exist_ok=True)
                
                # Copy images and labels
                train_images_dir = self.datasets_dir / dataset_name / config['train']
                train_labels_dir = self.datasets_dir / dataset_name / "train" / "labels"
                val_images_dir = self.datasets_dir / dataset_name / config['val']
                val_labels_dir = self.datasets_dir / dataset_name / "val" / "labels"
                
                combined_images_dir = combined_dir / "images"
                combined_labels_dir = combined_dir / "labels"
                combined_images_dir.mkdir(exist_ok=True)
                combined_labels_dir.mkdir(exist_ok=True)
                
                # Copy files
                if train_images_dir.exists():
                    for img_file in train_images_dir.glob("*.png"):
                        shutil.copy2(img_file, combined_images_dir)
                if val_images_dir.exists():
                    for img_file in val_images_dir.glob("*.png"):
                        shutil.copy2(img_file, combined_images_dir)
                
                if train_labels_dir.exists():
                    for lbl_file in train_labels_dir.glob("*.txt"):
                        shutil.copy2(lbl_file, combined_labels_dir)
                if val_labels_dir.exists():
                    for lbl_file in val_labels_dir.glob("*.txt"):
                        shutil.copy2(lbl_file, combined_labels_dir)
                
                new_config = {
                    'path': str(combined_dir),
                    'train': None,
                    'val': 'images',  # Point to the combined images directory
                    'nc': config['nc'],
                    'names': config['names']
                }
        
        # Write temporary config
        with open(temp_data_yaml, 'w') as f:
            yaml.dump(new_config, f, default_flow_style=False)
        
        return temp_data_yaml, config
    


    # ============================================================================
    # STEP 4: RUNNING BENCHMARKS AND SAVING RESULTS
    # ============================================================================
    
    def run_benchmark(self, model_name: str, dataset_name: str, mode: str = "val",
                     conf_threshold: float = 0.5, iou_threshold: float = 0.5) -> Dict:
        """
        Run benchmark for a specific model and dataset.
        
        Args:
            model_name: Name of the model file (without .pt extension)
            dataset_name: Name of the dataset folder
            mode: "val" for validation only, "all" for train+val combined
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for matching detections
        
        Returns:
            Dictionary containing evaluation results
        """
        model_path = self.models_dir / f"{model_name}.pt"
        
        if not model_path.exists():
            raise ValueError(f"Model {model_path} not found")
        
        results = self.evaluate_model_on_dataset(
            str(model_path), dataset_name, mode, conf_threshold, iou_threshold
        )
        
        print(f"  Storing benchmark results...")
        
        # Store results
        key = f"{model_name}_{dataset_name}_{mode}"
        if not hasattr(self, 'results'):
            self.results = {}
        self.results[key] = results
        
        print(f"  ✓ Results stored with key: {key}")
        
        return results
    
    def run_all_benchmarks(self, conf_threshold: float = 0.5, 
                          iou_threshold: float = 0.5) -> Dict[str, Dict]:
        """
        Run benchmarks for all selected datasets with the selected model and data mode.
        
        Args:
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for matching detections
        
        Returns:
            Dictionary of results for each dataset
        """
        if not self.selected_model:
            raise ValueError("No model selected. Use select_model() first.")
        
        if not self.selected_datasets:
            raise ValueError("No datasets selected. Use select_datasets() first.")
        
        if not hasattr(self, 'selected_mode') or not self.selected_mode:
            raise ValueError("No data mode selected. Use select_data_mode() first.")
        
        all_results = {}
        
        for ds_name in self.selected_datasets:
            print(f"\n{'='*60}")
            print(f"Running benchmark on {ds_name}")
            print(f"Data mode: {self.selected_mode}")
            print(f"{'='*60}")
            
            try:
                results = self.run_benchmark(
                    self.selected_model, ds_name, self.selected_mode, conf_threshold, iou_threshold
                )
                all_results[ds_name] = results
                
                # Save results to dataset-specific folder
                print(f"  Saving results for {ds_name}...")
                try:
                    self.save_dataset_results(ds_name, results)
                    print(f"  ✓ Results saved for {ds_name}")
                except Exception as e:
                    print(f"  ❌ Error saving results for {ds_name}: {e}")
                    # Continue with other datasets even if one fails to save
                
            except Exception as e:
                print(f"Error benchmarking {ds_name}: {e}")
                continue
        
        # Store results
        self.results = all_results
        
        print(f"\n{'='*60}")
        print(f"Benchmark Results Summary:")
        print(f"{'='*60}")
        print(f"Total datasets processed: {len(all_results)}")
        for ds_name, result in all_results.items():
            print(f"  {ds_name}: {result.get('overall_precision', 0):.3f} precision, {result.get('overall_recall', 0):.3f} recall")
        
        return all_results
    
    def save_dataset_results(self, dataset_name: str, results: Dict) -> None:
        """
        Save benchmark results directly to the benchmarks folder.
        
        Args:
            dataset_name: Name of the dataset
            results: Benchmark results dictionary
        """
        try:
            print(f"    Saving results directly to benchmarks folder")
            
            # Save results as JSON directly in benchmarks folder
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_file = self.benchmarks_dir / f"benchmark_results_{dataset_name}_{timestamp}.json"
            print(f"    Saving JSON results to: {json_file}")
            
            # Convert numpy types and Path objects for JSON serialization
            def convert_for_json(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, Path):
                    return str(obj)
                elif isinstance(obj, dict):
                    return {key: convert_for_json(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_for_json(item) for item in obj]
                else:
                    return obj
            
            serializable_results = convert_for_json(results)
            
            with open(json_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            print(f"✓ Results saved to {json_file}")
            
            # Clean up any existing dataset subfolder to keep benchmarks folder clean
            dataset_subfolder = self.benchmarks_dir / dataset_name
            if dataset_subfolder.exists() and dataset_subfolder.is_dir():
                print(f"    Cleaning up dataset subfolder: {dataset_subfolder}")
                import shutil
                shutil.rmtree(dataset_subfolder)
                print(f"    ✓ Dataset subfolder cleaned up")
            
            # Skipping CSV save per user request; only JSON file is created
            
            # Performance figures generation removed per user request
            
            # Bounding box visualization grids removed per user request
                
        except Exception as e:
            print(f"    ❌ Error saving results for {dataset_name}: {e}")
    
    # CSV saving removed per user request
    
    # ============================================================================
    # STEP 5: CREATING PERFORMANCE FIGURES
    # ============================================================================
    
    # Performance figures generation removed per user request

    
    # ============================================================================
    # STEP 6: CREATING BOUNDING BOX VISUALIZATION GRIDS
    # ============================================================================
    
    # Bounding box visualization functions removed per user request
    
    # ============================================================================
    # STEP 7: DATASET COMPARISON AND SYNTHESIS (removed)
    # ============================================================================
    
    # (comparison features removed)
    
    # ============================================================================
    # UTILITY FUNCTIONS
    # ============================================================================
    
    def print_results(self, results: Dict):
        """Print benchmark results in a formatted way."""
        print("\n" + "="*80)
        print(f"BENCHMARK RESULTS")
        print("="*80)
        print(f"Model: {results['model_path']}")
        print(f"Dataset: {results['dataset_name']}")
        print(f"Mode: {results['mode']}")
        print(f"Total Images: {results['total_images']}")
        print(f"Total Ground Truth: {results['total_ground_truth']}")
        print(f"Total Detections: {results['total_detections']}")
        print(f"Total Correct: {results['total_correct']}")
        print(f"Confidence Threshold: {results['conf_threshold']}")
        print(f"IoU Threshold: {results['iou_threshold']}")
        print("\nOverall Metrics:")
        print(f"  Precision: {results['overall_precision']:.4f}")
        print(f"  Recall: {results['overall_recall']:.4f}")
        print(f"  F1-Score: {results['overall_f1_score']:.4f}")
        print(f"  mAP@0.5: {results.get('overall_map50', 0):.4f}")
        print(f"  mAP@0.5:0.95: {results.get('overall_map50_95', 0):.4f}")
        
        print("="*80)
    
    def get_summary_stats(self) -> pd.DataFrame:
        """Get summary statistics for all benchmark results."""
        if not self.results:
            return pd.DataFrame()
        
        summary_data = []
        for key, result in self.results.items():
            model_name = Path(result['model_path']).stem
            summary_data.append({
                'Model': model_name,
                'Dataset': result['dataset_name'],
                'Mode': result['mode'],
                'Images': result['total_images'],
                'Ground Truth': result['total_ground_truth'],
                'Precision': f"{result['overall_precision']:.4f}",
                'Recall': f"{result['overall_recall']:.4f}",
                'F1-Score': f"{result['overall_f1_score']:.4f}",
                'mAP@0.5': f"{result.get('ultralytics_metrics', {}).get('metrics/mAP50(B)', 0.0):.4f}",
                'mAP@0.5:0.95': f"{result.get('ultralytics_metrics', {}).get('metrics/mAP50-95(B)', 0.0):.4f}"
            })
        
        return pd.DataFrame(summary_data)
    
    def create_comparison_manually(self) -> None:
        """Manually trigger comparison figure creation if results exist."""
        if hasattr(self, 'results') and self.results:
            print("Creating comparison figures from existing results...")
            self.create_comparison_figures()
        else:
            print("No results available. Run benchmarks first.")
    
    def test_benchmark_setup(self) -> bool:
        """
        Test if the benchmark setup is working correctly.
        
        Returns:
            True if setup is working, False otherwise
        """
        try:
            print("Testing benchmark setup...")
            
            # Test directory creation
            test_dir = self.benchmarks_dir / "test"
            test_dir.mkdir(exist_ok=True)
            test_file = test_dir / "test.txt"
            test_file.write_text("test")
            test_file.unlink()
            test_dir.rmdir()
            
            # Test model detection
            models = self.get_available_models()
            print(f"✓ Found {len(models)} models: {models}")
            
            # Test dataset detection
            datasets = self.get_available_datasets()
            print(f"✓ Found {len(datasets)} datasets: {datasets}")
            
            # Test data.yaml loading for first dataset
            if datasets:
                try:
                    config = self.load_dataset_info(datasets[0])
                    print(f"✓ Successfully loaded config for {datasets[0]}: {config['nc']} classes")
                except Exception as e:
                    print(f"⚠ Warning: Could not load config for {datasets[0]}: {e}")
            
            print("✓ Benchmark setup test completed successfully")
            return True
            
        except Exception as e:
            print(f"❌ Benchmark setup test failed: {e}")
            return False
