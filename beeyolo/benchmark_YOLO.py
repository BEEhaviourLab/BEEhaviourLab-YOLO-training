#!/usr/bin/env python3
"""
YOLO Model Benchmarking Script

This script evaluates YOLO models on different datasets to assess their performance.
It supports both validation-only and all-data evaluation modes.
"""

import os
import argparse
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
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


class YOLOBenchmark:
    """Class for benchmarking YOLO models on different datasets."""
    
    def __init__(self, models_dir: str = "models", datasets_dir: str = "datasets"):
        """
        Initialize the benchmark.
        
        Args:
            models_dir: Directory containing YOLO model files
            datasets_dir: Directory containing dataset folders
        """
        self.models_dir = Path(models_dir)
        self.datasets_dir = Path(datasets_dir)
        self.results = {}
        
        # Ensure directories exist
        if not self.models_dir.exists():
            raise ValueError(f"Models directory {models_dir} does not exist")
        if not self.datasets_dir.exists():
            raise ValueError(f"Datasets directory {datasets_dir} does not exist")
    
    def get_available_models(self) -> List[str]:
        """Get list of available YOLO models."""
        model_files = list(self.models_dir.glob("*.pt"))
        return [f.stem for f in model_files]
    
    def get_available_datasets(self) -> List[str]:
        """Get list of available datasets."""
        dataset_dirs = [d for d in self.datasets_dir.iterdir() 
                       if d.is_dir() and (d / "data.yaml").exists()]
        return [d.name for d in dataset_dirs]
    
    def load_dataset_info(self, dataset_name: str) -> Dict:
        """Load dataset configuration from data.yaml."""
        dataset_path = self.datasets_dir / dataset_name
        yaml_path = dataset_path / "data.yaml"
        
        if not yaml_path.exists():
            raise ValueError(f"data.yaml not found in {dataset_path}")
        
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
    
    def load_ground_truth_labels(self, label_path: Path, num_classes: int) -> List[List]:
        """
        Load ground truth labels from a YOLO format label file.
        
        Args:
            label_path: Path to the label file
            num_classes: Number of classes in the dataset
        
        Returns:
            List of [class_id, x_center, y_center, width, height] for each detection
        """
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
        """
        Convert YOLO normalized coordinates to absolute pixel coordinates.
        
        Args:
            yolo_coords: [class_id, x_center, y_center, width, height] in normalized form
            img_width: Image width in pixels
            img_height: Image height in pixels
        
        Returns:
            [class_id, x_center, y_center, width, height] in absolute pixel coordinates
        """
        class_id, x_center, y_center, width, height = yolo_coords
        
        x_center_abs = x_center * img_width
        y_center_abs = y_center * img_height
        width_abs = width * img_width
        height_abs = height * img_height
        
        return [class_id, x_center_abs, y_center_abs, width_abs, height_abs]
    
    def calculate_iou(self, box1: List, box2: List) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            box1: [class_id, x_center, y_center, width, height, ...] in absolute coordinates
            box2: [class_id, x_center, y_center, width, height, ...] in absolute coordinates
        
        Returns:
            IoU value between 0 and 1
        """
        try:
            # Handle different box formats - extract x_center, y_center, width, height
            # box1 and box2 can be [class_id, x_center, y_center, width, height] or longer
            if len(box1) < 4 or len(box2) < 4:
                print(f"Warning: Box format issue - box1: {box1}, box2: {box2}")
                return 0.0
            
            # Extract coordinates safely
            x1, y1, w1, h1 = box1[1:5]  # Skip class_id, get x_center, y_center, width, height
            x2, y2, w2, h2 = box2[1:5]  # Skip class_id, get x_center, y_center, width, height
            
            # Validate coordinates
            if not all(isinstance(x, (int, float)) for x in [x1, y1, w1, h1, x2, y2, w2, h2]):
                print(f"Warning: Non-numeric coordinates - box1: {box1}, box2: {box2}")
                return 0.0
            
            # Convert to corner coordinates
            x1_min, x1_max = x1 - w1/2, x1 + w1/2
            y1_min, y1_max = y1 - h1/2, y1 + h1/2
            x2_min, x2_max = x2 - w2/2, x2 + w2/2
            y2_min, y2_max = y2 - h2/2, y2 + h2/2
            
            # Calculate intersection
            x_left = max(x1_min, x2_min)
            y_top = max(y1_min, y2_min)
            x_right = min(x1_max, x2_max)
            y_bottom = min(y1_max, y2_max)
            
            if x_right < x_left or y_bottom < y_top:
                return 0.0
            
            intersection = (x_right - x_left) * (y_bottom - y_top)
            
            # Calculate union
            area1 = w1 * h1
            area2 = w2 * h2
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            print(f"Error in calculate_iou: {e}")
            print(f"box1: {box1}, box2: {box2}")
            return 0.0
    
    def evaluate_model_on_dataset(self, model_path: str, dataset_name: str, 
                                mode: str = "val", conf_threshold: float = 0.5, 
                                iou_threshold: float = 0.5) -> Dict:
        """
        Evaluate a YOLO model on a specific dataset.
        
        Args:
            model_path: Path to the YOLO model file
            dataset_name: Name of the dataset folder
            mode: "val" for validation only, "all" for train+val combined
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for matching detections
        
        Returns:
            Dictionary containing evaluation metrics
        """
        print(f"Evaluating model {model_path} on dataset {dataset_name} ({mode} mode)")
        
        # Load model
        model = YOLO(model_path)
        
        # Load dataset info
        config = self.load_dataset_info(dataset_name)
        num_classes = config['nc']
        class_names = config['names']
        
        # Get image and label paths
        image_paths, label_paths = self.get_dataset_images_and_labels(dataset_name, mode)
        
        if not image_paths:
            raise ValueError(f"No images found in dataset {dataset_name}")
        
        print(f"Found {len(image_paths)} images to evaluate")
        
        # Initialize metrics
        total_gt = 0
        total_detections = 0
        total_correct = 0
        class_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
        
        # Process each image
        for img_path in tqdm(image_paths, desc="Processing images"):
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            img_height, img_width = img.shape[:2]
            
            # Debug: Print first image dimensions
            if img_path == image_paths[0]:
                print(f"Debug: First image dimensions: {img_width}x{img_height}")
            
            # Find corresponding label file
            label_path = None
            for lbl_path in label_paths:
                if lbl_path.stem == img_path.stem:
                    label_path = lbl_path
                    break
            
            # Load ground truth
            gt_labels = self.load_ground_truth_labels(label_path, num_classes)
            total_gt += len(gt_labels)
            
            # Debug: Print first ground truth labels
            if img_path == image_paths[0] and gt_labels:
                print(f"Debug: First GT label format: {gt_labels[0]}")
            
            # Run inference
            results = model(img, conf=conf_threshold, verbose=False)
            
            # Process detections
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        
                        # Convert to center format
                        x_center = (x1 + x2) / 2
                        y_center = (y1 + y2) / 2
                        width = x2 - x1
                        height = y2 - y1
                        
                        detections.append([cls, x_center, y_center, width, height, conf])
            
            # Debug: Print first detection format
            if img_path == image_paths[0] and detections:
                print(f"Debug: First detection format: {detections[0]}")
            
            total_detections += len(detections)
            
            # Match detections with ground truth
            matched_gt = set()
            matched_detections = set()
            
            for det_idx, detection in enumerate(detections):
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, gt_label in enumerate(gt_labels):
                    if gt_idx in matched_gt:
                        continue
                    
                    # Convert ground truth to absolute coordinates
                    gt_abs = self.convert_yolo_to_absolute(gt_label, img_width, img_height)
                    
                    # Check class match
                    if detection[0] == gt_abs[0]:  # Same class
                        # Debug: Print the first few comparisons to understand the data
                        if img_path == image_paths[0] and det_idx == 0 and gt_idx == 0:
                            print(f"Debug: Comparing detection {detection} with GT {gt_abs}")
                        
                        try:
                            iou = self.calculate_iou(detection, gt_abs)
                            if iou > best_iou and iou >= iou_threshold:
                                best_iou = iou
                                best_gt_idx = gt_idx
                        except Exception as e:
                            # Debug info for troubleshooting
                            print(f"Warning: IoU calculation failed for detection {detection} and gt {gt_abs}: {e}")
                            continue
                
                if best_gt_idx >= 0:
                    # True positive
                    matched_gt.add(best_gt_idx)
                    matched_detections.add(det_idx)
                    total_correct += 1
                    class_metrics[detection[0]]['tp'] += 1
                else:
                    # False positive
                    class_metrics[detection[0]]['fp'] += 1
            
            # Count false negatives (unmatched ground truth)
            for gt_idx, gt_label in enumerate(gt_labels):
                if gt_idx not in matched_gt:
                    class_metrics[gt_label[0]]['fn'] += 1
        
        # Calculate overall metrics
        precision = total_correct / total_detections if total_detections > 0 else 0
        recall = total_correct / total_gt if total_gt > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate per-class metrics
        class_results = {}
        for class_id in range(num_classes):
            tp = class_metrics[class_id]['tp']
            fp = class_metrics[class_id]['fp']
            fn = class_metrics[class_id]['fn']
            
            class_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            class_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            class_f1 = 2 * (class_precision * class_recall) / (class_precision + class_recall) if (class_precision + class_recall) > 0 else 0
            
            class_results[class_names[class_id]] = {
                'precision': class_precision,
                'recall': class_recall,
                'f1_score': class_f1,
                'true_positives': tp,
                'false_positives': fp,
                'false_negatives': fn
            }
        
        results = {
            'model_path': model_path,
            'dataset_name': dataset_name,
            'mode': mode,
            'total_images': len(image_paths),
            'total_ground_truth': total_gt,
            'total_detections': total_detections,
            'total_correct': total_correct,
            'overall_precision': precision,
            'overall_recall': recall,
            'overall_f1_score': f1_score,
            'class_metrics': class_results,
            'conf_threshold': conf_threshold,
            'iou_threshold': iou_threshold
        }
        
        return results
    
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
        
        # Store results
        key = f"{model_name}_{dataset_name}_{mode}"
        self.results[key] = results
        
        return results
    
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
        
        print("\nPer-Class Metrics:")
        for class_name, metrics in results['class_metrics'].items():
            print(f"  {class_name}:")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall: {metrics['recall']:.4f}")
            print(f"    F1-Score: {metrics['f1_score']:.4f}")
            print(f"    TP: {metrics['true_positives']}, FP: {metrics['false_positives']}, FN: {metrics['false_negatives']}")
        
        print("="*80)
    
    def save_results(self, output_file: str = None):
        """Save benchmark results to a JSON file."""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"benchmark_results_{timestamp}.json"
        
        # Convert Path objects to strings for JSON serialization
        serializable_results = {}
        for key, result in self.results.items():
            serializable_result = result.copy()
            serializable_result['model_path'] = str(result['model_path'])
            serializable_results[key] = serializable_result
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved to {output_file}")
    
    def generate_summary_report(self):
        """Generate a summary report of all benchmark results."""
        if not self.results:
            print("No results to summarize")
            return
        
        print("\n" + "="*100)
        print("BENCHMARK SUMMARY REPORT")
        print("="*100)
        
        # Group results by model
        model_results = defaultdict(list)
        for key, result in self.results.items():
            model_name = Path(result['model_path']).stem
            model_results[model_name].append(result)
        
        for model_name, results in model_results.items():
            print(f"\nModel: {model_name}")
            print("-" * 50)
            
            for result in results:
                dataset = result['dataset_name']
                mode = result['mode']
                f1 = result['overall_f1_score']
                precision = result['overall_precision']
                recall = result['overall_recall']
                
                print(f"  {dataset} ({mode}): F1={f1:.4f}, P={precision:.4f}, R={recall:.4f}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Benchmark YOLO models on datasets")
    parser.add_argument("--model", required=True, help="Model name (without .pt extension)")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--mode", choices=["val", "all"], default="val", 
                       help="Data selection mode: 'val' for validation only, 'all' for train+val combined")
    parser.add_argument("--conf-threshold", type=float, default=0.5, 
                       help="Confidence threshold for detections")
    parser.add_argument("--iou-threshold", type=float, default=0.5, 
                       help="IoU threshold for matching detections")
    parser.add_argument("--output", help="Output file for results")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--list-datasets", action="store_true", help="List available datasets")
    
    args = parser.parse_args()
    
    try:
        benchmark = YOLOBenchmark()
        
        if args.list_models:
            models = benchmark.get_available_models()
            print("Available models:")
            for model in models:
                print(f"  - {model}")
            return
        
        if args.list_datasets:
            datasets = benchmark.get_available_datasets()
            print("Available datasets:")
            for dataset in datasets:
                print(f"  - {dataset}")
            return
        
        # Run benchmark
        results = benchmark.run_benchmark(
            args.model, args.dataset, args.mode, 
            args.conf_threshold, args.iou_threshold
        )
        
        # Print results
        benchmark.print_results(results)
        
        # Save results
        if args.output:
            benchmark.save_results(args.output)
        else:
            benchmark.save_results()
        
        # Generate summary
        benchmark.generate_summary_report()
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
