from ultralytics import YOLO
import cv2
import os
from pathlib import Path

def process_video(video_path, model_path, output_video_path, output_coords_path):
    """
    Process a video with the trained YOLO model and save coordinates
    
    Args:
        video_path (str): Path to input video
        model_path (str): Path to trained model weights
        output_video_path (str): Path for output video
        output_coords_path (str): Path for output coordinates file
    """
    # Load the model
    model = YOLO(model_path)
    
    # Open the video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (frame_width, frame_height))
    
    # Open coordinates file
    with open(output_coords_path, 'w') as f:
        f.write("frame_number class_id center_x center_y width height confidence\n")
    
    frame_number = 0
    
    # Process the video
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        # Run inference on the frame
        results = model(frame, conf=0.5)  # adjust confidence threshold as needed
        
        # Get the annotated frame
        annotated_frame = results[0].plot()
        
        # Write frame to video
        out.write(annotated_frame)
        
        # Save detection coordinates
        with open(output_coords_path, 'a') as f:
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Get normalized coordinates (YOLO format)
                    x, y, w, h = box.xywh[0].tolist()
                    x = x / frame_width
                    y = y / frame_height
                    w = w / frame_width
                    h = h / frame_height
                    
                    # Get class and confidence
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Write to file
                    f.write(f"{frame_number} {cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {conf:.6f}\n")
        
        frame_number += 1
        
        # Optional: Print progress every 100 frames
        if frame_number % 100 == 0:
            print(f"Processed {frame_number} frames")
    
    # Clean up
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Processing complete for {video_path.name}!")
    print(f"Saved to: {output_video_path}")
    print(f"Coordinates saved to: {output_coords_path}")

def process_all_videos(input_folder, output_folder, model_path):
    """
    Process all videos in the input folder
    """
    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Get all mp4 files in input folder
    input_videos = list(Path(input_folder).glob("*_trimmed.mp4"))
    
    if not input_videos:
        print(f"No trimmed videos found in {input_folder}")
        return
    
    print(f"Found {len(input_videos)} videos to process")
    
    # Process each video
    for video_path in input_videos:
        # Create output paths
        output_name = video_path.stem.replace("_trimmed", "_detected")
        output_video_path = Path(output_folder) / f"{output_name}.mp4"
        output_coords_path = Path(output_folder) / f"{output_name}.txt"
        
        print(f"\nProcessing: {video_path.name}")
        process_video(video_path, model_path, output_video_path, output_coords_path)

if __name__ == "__main__":
    # Paths
    input_folder = "videos/trimmed"
    output_folder = "videos/output"
    model_path = "runs/detect/train7/weights/best.pt"  # Your 100-epoch model
    
    process_all_videos(input_folder, output_folder, model_path)