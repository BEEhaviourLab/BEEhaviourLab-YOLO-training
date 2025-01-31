import cv2
import os
from pathlib import Path

def trim_video(input_path, output_path, start_time, end_time):
    """
    Trim a video using OpenCV
    
    Args:
        input_path (str): Path to input video
        output_path (str): Path to output video
        start_time (int): Start time in seconds
        end_time (int): End time in seconds
    """
    try:
        # Open the video
        cap = cv2.VideoCapture(str(input_path))
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate frame numbers from timestamps
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Set frame position to start_frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Read and write frames
        frame_count = 0
        while cap.isOpened() and frame_count < (end_frame - start_frame):
            ret, frame = cap.read()
            if not ret:
                break
                
            out.write(frame)
            frame_count += 1
            
            # Optional: Show progress
            if frame_count % fps == 0:  # Update every second
                print(f"Processing second {frame_count//fps} of {(end_frame-start_frame)//fps}")
        
        # Release everything
        cap.release()
        out.release()
        
        print(f"Video trimmed successfully: {output_path}")
        
    except Exception as e:
        print(f"Error trimming video: {e}")

def process_video_folder(input_folder, output_folder, start_time=10, end_time=30):
    """
    Process all videos in a folder
    
    Args:
        input_folder (str): Path to folder containing videos
        output_folder (str): Path to output folder
        start_time (int): Start time in seconds
        end_time (int): End time in seconds
    """
    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Get all video files
    video_files = [f for f in Path(input_folder).glob("*.mp4")]
    
    print(f"Found {len(video_files)} video files")
    
    # Process each video
    for video_file in video_files:
        # Create output filename
        output_name = video_file.stem + "_trimmed.mp4"
        output_path = Path(output_folder) / output_name
        
        print(f"\nProcessing: {video_file.name}")
        trim_video(video_file, output_path, start_time, end_time)

if __name__ == "__main__":
    # Set your paths
    videos_folder = "R:/Bee audio and video recordings/Group_data_training/raw_data_vids_images/videos"  # Your input folder
    output_folder = "videos/trimmed"  # Your output folder
    
    # Process all videos
    process_video_folder(videos_folder, output_folder)