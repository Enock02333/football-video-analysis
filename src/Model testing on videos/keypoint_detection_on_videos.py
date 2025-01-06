import os
import cv2
from ultralytics import YOLO
import torch

# Paths
video_path = "C:/Users/enock/Downloads/dataset/videos/train"
model_path = "C:/Users/enock/Downloads/dataset/runs/train/pitch_detection2/weights/best.pt"
output_path = "C:/Users/enock/Downloads/dataset/videos/annotated_pitch_keypoints"

# Create output directory if not exists
os.makedirs(output_path, exist_ok=True)

# Initialize the YOLO model
model = YOLO(model_path)

def process_video(video_file, output_file):
    cap = cv2.VideoCapture(video_file)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect keypoints
        results = model.predict(frame, imgsz=640, conf=0.25, device='cuda' if torch.cuda.is_available() else 'cpu')

        # Draw detected keypoints on the frame
        annotated_frame = results[0].plot()  # Visualize results on the frame
        
        out.write(annotated_frame)
        frame_count += 1
        print(f"Processed frame {frame_count}", end="\r")
    
    cap.release()
    out.release()
    print(f"\nFinished processing {video_file}. Saved to {output_file}")

def process_videos(video_folder, output_folder):
    for root, _, files in os.walk(video_folder):
        for file in files:
            if file.endswith(('.mp4', '.avi', '.mkv')):
                input_file = os.path.join(root, file)
                output_file = os.path.join(output_folder, f"annotated_{file}")
                print(f"Processing video: {input_file}")
                process_video(input_file, output_file)

if __name__ == "__main__":
    process_videos(video_path, output_path)
