# Authors: Cait Newport & Rachel Parkinson
# Date: 2025-01-22 / 2025-01-28
# Description: This script is used to train a YOLO model for an object detection task.

# To run this script, you need to have the following:
# 1. A dataset in the format of YOLOv8
# 2. A data.yaml file that describes the dataset
# 3. A YOLOv8 model checkpoint file
# There are several standard YOLOv8 pre-trained models available:
   # yolov8n.pt - Nano model (smallest, fastest, least accurate)
   # yolov8s.pt - Small model
   # yolov8m.pt - Medium model
   # yolov8l.pt - Large model
  # yolov8x.pt - Extra Large model (largest, slowest, most accurate)

# To run the script, in Terminal:
# python YOLO_model_training.py \
#     --project_name "triggerfish-detection" \ # Name of your Weights & Biases project
#     --run_name "yolov8-training-run-1" \ # Name of this specific training run
#     --data_yaml "path/to/data.yaml" \ # Path to your YOLO format data.yaml file
#     --train_path "path/to/train" \ # Path to your training images directory
#     --val_path "path/to/val" \ # Path to your validation images directory
#     --model_size "s"  # Choose a model size: n(ano), s(mall), m(edium), l(arge), x(large)
#     --epochs 100 \ # Number of training epochs (default 100)
#     --batch_size 16 \ # Training batch size (default 16)
#     --image_size 640 # Input image size (default 640)

# All parameters after model_size are optional and will use their default values if not specified. 



import os
import argparse
import wandb
import logging
from ultralytics import YOLO
from dotenv import load_dotenv

class YOLOTrainer:
    def __init__(self, project_name, run_name, data_yaml_path, train_path, val_path, model_size='n', 
                 epochs=100, batch_size=16, image_size=640):
        """Add dependency checks at initialization"""
        # Add logger
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

        try:
            import wandb
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("Required packages not installed. Please run:\npip install wandb ultralytics")
        
        self.project_name = project_name
        self.run_name = run_name
        self.data_yaml_path = data_yaml_path
        self.train_path = train_path
        self.val_path = val_path
        self.model_size = model_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.image_size = image_size
        self.model = None
        
    def init_wandb(self):
        """Initialize Weights & Biases tracking"""
        try:
            load_dotenv()  # Load environment variables from .env file
            api_key = os.getenv('WANDB_API_KEY')
            if not api_key:
                raise ValueError("WANDB_API_KEY not found in environment variables")
            
            wandb.login(key=api_key)
            wandb.init(
                project=self.project_name,
                name=self.run_name,
                config={
                    "architecture": "YOLOv8",
                    "dataset": "group_bees_2024",
                    "epochs": self.epochs,
                    "batch_size": self.batch_size,
                    "img_size": self.image_size,
                    "learning_rate": 0.01
                }
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize wandb: {str(e)}")
            raise

    def validate_paths(self):
        """Validate all necessary paths"""
        print("\nDEBUG: Checking paths...")
        print(f"Current working directory: {os.getcwd()}")
        
        # Check data.yaml
        print(f"\ndata.yaml path: {self.data_yaml_path}")
        print(f"data.yaml exists: {os.path.exists(self.data_yaml_path)}")
        
        # Get label paths correctly
        train_labels_path = self.train_path.replace('images', 'labels')
        val_labels_path = self.val_path.replace('images', 'labels')
        
        print(f"\nTrain images path: {self.train_path}")
        print(f"Train labels path: {train_labels_path}")
        print(f"Val images path: {self.val_path}")
        print(f"Val labels path: {val_labels_path}")
        
        # Check directories exist
        print(f"\nTrain images directory exists: {os.path.exists(self.train_path)}")
        print(f"Train labels directory exists: {os.path.exists(train_labels_path)}")
        print(f"Val images directory exists: {os.path.exists(self.val_path)}")
        print(f"Val labels directory exists: {os.path.exists(val_labels_path)}")
        
        if not all([os.path.exists(p) for p in [self.train_path, self.val_path, train_labels_path, val_labels_path]]):
            raise FileNotFoundError("One or more required directories not found!")
            
        # Count files
        train_images = len([f for f in os.listdir(self.train_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
        val_images = len([f for f in os.listdir(self.val_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
        train_labels = len([f for f in os.listdir(train_labels_path) if f.endswith('.txt')])
        val_labels = len([f for f in os.listdir(val_labels_path) if f.endswith('.txt')])
        
        print("\nFile counts:")
        print(f"Train images: {train_images}")
        print(f"Train labels: {train_labels}")
        print(f"Val images: {val_images}")
        print(f"Val labels: {val_labels}")
        
        if train_images != train_labels or val_images != val_labels:
            print("\nWARNING: Number of images and labels don't match!")

        # Validate yaml file content
        try:
            import yaml
            with open(self.data_yaml_path, 'r') as f:
                yaml_data = yaml.safe_load(f)
                required_keys = ['train', 'val', 'nc', 'names']
                if not all(key in yaml_data for key in required_keys):
                    raise ValueError(f"data.yaml must contain all of: {required_keys}")
        except Exception as e:
            raise ValueError(f"Error reading data.yaml: {str(e)}")

    def train(self):
        """Train the YOLO model"""
        try:
            # Get absolute paths
            data_yaml_dir = os.path.dirname(os.path.abspath(self.data_yaml_path))
            
            # Update data.yaml with absolute paths
            import yaml
            with open(self.data_yaml_path, 'r') as f:
                data_yaml = yaml.safe_load(f)
            
            # Create temporary data.yaml in the same directory as original data.yaml
            temp_yaml_path = os.path.join(data_yaml_dir, 'temp_data.yaml')
            
            # Convert relative paths to absolute paths
            if not os.path.isabs(data_yaml['train']):
                data_yaml['train'] = os.path.abspath(os.path.join(data_yaml_dir, data_yaml['train']))
            if not os.path.isabs(data_yaml['val']):
                data_yaml['val'] = os.path.abspath(os.path.join(data_yaml_dir, data_yaml['val']))
            
            # Update path to be the directory containing data.yaml
            data_yaml['path'] = data_yaml_dir
            
            with open(temp_yaml_path, 'w') as f:
                yaml.dump(data_yaml, f)
            
            model_path = f'yolov8{self.model_size}.pt'
            print(f"Loading model: {model_path}")
            
            # YOLO will automatically download the correct model
            self.model = YOLO(model_path)
            
            # Auto-detect device
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
            print(f"Using device: {device}")
            
            results = self.model.train(
                data=temp_yaml_path,  # Use the temporary yaml file
                epochs=self.epochs,
                imgsz=self.image_size,
                batch=self.batch_size,
                device=device,
                optimizer="auto",
                patience=50,
                save_period=10,
                workers=8,
                pretrained=True,
                verbose=True,
                seed=42,
                resume=False,
                project='runs/detect',  # Set explicit output directory
                name='train'  # This will auto-increment (train, train2, etc.)
            )
            
            # Clean up temporary file
            os.remove(temp_yaml_path)
            
            return results
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise

    def run_training(self):
        """Execute the full training pipeline"""
        try:
            self.init_wandb()
            self.validate_paths()
            results = self.train()
            return results
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise
        finally:
            wandb.finish()

def main():
    parser = argparse.ArgumentParser(description='Train YOLO model for bee detection')
    parser.add_argument('--project_name', type=str, required=True, help='Name of the W&B project')
    parser.add_argument('--run_name', type=str, required=True, help='Name of this specific training run')
    parser.add_argument('--data_yaml', type=str, required=True, help='Path to data.yaml file')
    parser.add_argument('--train_path', type=str, required=True, help='Path to training images directory')
    parser.add_argument('--val_path', type=str, required=True, help='Path to validation images directory')
    parser.add_argument('--model_size', type=str, default='n', choices=['n','s','m','l','x'], 
                       help='YOLOv8 model size: n(ano), s(mall), m(edium), l(arge), x(large)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('--image_size', type=int, default=640, help='Input image size')
    
    args = parser.parse_args()
    
    trainer = YOLOTrainer(
        project_name=args.project_name,
        run_name=args.run_name,
        data_yaml_path=args.data_yaml,
        train_path=args.train_path,
        val_path=args.val_path,
        model_size=args.model_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size
    )
    
    trainer.run_training()

if __name__ == "__main__":
    main()