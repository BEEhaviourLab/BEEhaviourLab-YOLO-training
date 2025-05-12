# Convert YOLOv11 model to NCNN format
import os
from ultralytics import YOLO

def convert_to_ncnn(model_path, output_name):
    """
    Convert YOLOv11 model to NCNN format
    
    Args:
        model_path (str): Path to the .pt model file
        output_name (str): Base name for the output files
    """
    try:
        print(f"Loading model from: {model_path}")
        model = YOLO(model_path)
        
        # Create output directory if it doesn't exist
        output_dir = "converted_models"
        os.makedirs(output_dir, exist_ok=True)
        
        # Export the model to NCNN format
        print("Converting to NCNN format...")
        output_path = os.path.join(output_dir, output_name)
        model.export(format="ncnn", imgsz=640)  # specify image size
        
        # The export creates several files:
        # - model.param (network structure)
        # - model.bin (weights)
        print(f"\nModel converted successfully!")
        print("Generated files:")
        print(f"- {output_name}.param")
        print(f"- {output_name}.bin")
        
    except Exception as e:
        print(f"Error during conversion: {str(e)}")

if __name__ == "__main__":
    # Path to your trained model
    model_path = "runs/detect/train5/weights/best.pt"
    output_name = "yolo11n_bee_detector"
    
    convert_to_ncnn(model_path, output_name)
