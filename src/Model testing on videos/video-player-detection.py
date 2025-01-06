import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import supervision as sv

class FootballVideoAnalyzer:
    def __init__(self, player_model_path, output_path):
        """
        Initialize the Football Video Analyzer
        
        Args:
            player_model_path: Path to the YOLOv8 model for player detection
            output_path: Directory to save output videos
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.player_model = YOLO(player_model_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True, parents=True)
        
        # Initialize colors for different classes
        self.colors = {
            'team1': (0, 255, 0),    # Green for team 1
            'team2': (0, 0, 255),    # Red for team 2
            'referee': (255, 255, 0), # Yellow for referee
            'goalkeeper1': (0, 165, 255),  # Orange for goalkeeper 1
            'goalkeeper2': (255, 0, 255)   # Purple for goalkeeper 2
        }
        
    def process_video(self, video_path, output_name=None):
        """
        Process a video file for player detection and team classification
        
        Args:
            video_path: Path to input video
            output_name: Name for output video (optional)
        """
        video_path = Path(video_path)
        if output_name is None:
            output_name = f"analyzed_{video_path.name}"
            
        cap = cv2.VideoCapture(str(video_path))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        output_path = self.output_path / output_name
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )
        
        # Initialize tracker
        tracker = sv.ByteTrack()
        box_annotator = sv.BoxAnnotator()
        
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {video_path.name}")
        print(f"Total frames: {total_frames}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            if frame_count % 3 != 0:  # Process every 3rd frame for performance
                continue
                
            if frame_count % 30 == 0:  # Progress update every 30 frames
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}%")
                
            # Detect players
            results = self.player_model(frame)[0]
            detections = sv.Detections.from_yolov8(results)
            
            if len(detections) > 0:
                # Track players
                detections = tracker.update_with_detections(detections)
                
                # Classify teams (placeholder - you'll need to implement team classification)
                team_classifications = self.classify_teams(frame, detections)
                
                # Draw annotations
                labels = [
                    f"#{tracker_id} {team}"
                    for tracker_id, team in zip(detections.tracker_id, team_classifications)
                ]
                
                frame = box_annotator.annotate(
                    frame=frame,
                    detections=detections,
                    labels=labels
                )
            
            writer.write(frame)
            
        cap.release()
        writer.release()
        print(f"Completed processing {video_path.name}")
        
    def classify_teams(self, frame, detections):
        """
        Placeholder for team classification logic
        You'll need to implement this based on your jersey detection model
        
        Returns:
            List of team classifications for each detection
        """
        # Placeholder - replace with actual team classification logic
        return ['team1' for _ in range(len(detections))]
        
    def process_directory(self, input_dir):
        """
        Process all videos in a directory
        
        Args:
            input_dir: Directory containing input videos
        """
        input_path = Path(input_dir)
        video_extensions = ['.mp4', '.avi', '.mov']
        
        # Get all video files from both train and test folders
        video_files = []
        for ext in video_extensions:
            video_files.extend(input_path.rglob(f"*{ext}"))
            
        print(f"Found {len(video_files)} videos to process")
        
        # Process each video
        for video_file in video_files:
            try:
                relative_path = video_file.relative_to(input_path)
                output_name = relative_path.name
                self.process_video(video_file, output_name)
            except Exception as e:
                print(f"Error processing {video_file}: {str(e)}")

# Create and run the analyzer
def main():
    player_model_path = r"C:/Users/enock/Downloads/dataset/runs/train/football_detection/weights/best.pt"
    input_dir = r"C:/Users/enock/Downloads/dataset/videos/train"
    output_dir = r"C:/Users/enock/Downloads/dataset/videos/annotated_teams"
    
    analyzer = FootballVideoAnalyzer(player_model_path, output_dir)
    analyzer.process_directory(input_dir)

if __name__ == "__main__":
    main()
