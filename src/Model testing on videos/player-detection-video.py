import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from sklearn.cluster import KMeans

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
            'team1': (0, 255, 0),     # Green for team 1
            'team2': (0, 0, 255),     # Red for team 2
            'referee': (0, 255, 255)   # Yellow for referee
        }
        
        # Initialize team tracking
        self.team_memory = {}  # Dictionary to store team assignments
        
    def assign_teams(self, boxes, class_names):
        """
        Assign teams based on player positions using clustering
        """
        player_positions = []
        player_indices = []
        
        # Get positions of all players (excluding referees)
        for i, (box, class_name) in enumerate(zip(boxes, class_names)):
            if class_name == 'player':
                x1, y1, x2, y2 = box.xyxy[0].astype(int)
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                player_positions.append([center_x, center_y])
                player_indices.append(i)
                
        if len(player_positions) < 2:
            return ['unknown'] * len(boxes)
            
        # Use K-means clustering to separate players into two teams
        if len(player_positions) >= 2:
            kmeans = KMeans(n_clusters=2, n_init=10)
            clusters = kmeans.fit_predict(player_positions)
            
            # Assign teams
            team_assignments = ['unknown'] * len(boxes)
            for idx, cluster in zip(player_indices, clusters):
                team_assignments[idx] = f'team{cluster + 1}'
                
            # Assign referees
            for i, class_name in enumerate(class_names):
                if class_name == 'referee':
                    team_assignments[i] = 'referee'
                    
            return team_assignments
        return ['unknown'] * len(boxes)

    def draw_boxes(self, frame, results):
        """
        Draw bounding boxes and labels on the frame
        """
        annotated_frame = frame.copy()
        
        if not results:
            return annotated_frame
            
        for result in results:
            boxes = result.boxes.cpu().numpy()
            class_names = [result.names[int(box.cls[0])] for box in boxes]
            
            # Assign teams to players
            team_assignments = self.assign_teams(boxes, class_names)
            
            for i, (box, team) in enumerate(zip(boxes, team_assignments)):
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].astype(int)
                
                # Get confidence
                conf = box.conf[0]
                
                # Get color based on team assignment
                color = self.colors.get(team, (128, 128, 128))  # Gray for unknown
                
                # Draw box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{team} {conf:.2f}"
                (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(annotated_frame, (x1, y1 - label_height - 5), 
                            (x1 + label_width, y1), color, -1)
                cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                
        return annotated_frame
        
    def process_video(self, video_path, output_name=None):
        """
        Process a video file for player detection and team classification
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
            fps,  # Use original FPS
            (width, height)
        )
        
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {video_path.name}")
        print(f"Total frames: {total_frames}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            if frame_count % 30 == 0:  # Progress update every 30 frames
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}%")
                
            # Detect players
            results = self.player_model(frame)
            
            # Draw detections
            annotated_frame = self.draw_boxes(frame, results)
            
            writer.write(annotated_frame)
            
        cap.release()
        writer.release()
        print(f"Completed processing {video_path.name}")
        
    def process_directory(self, input_dir):
        """
        Process all videos in a directory
        """
        input_path = Path(input_dir)
        video_extensions = ['.mp4', '.avi', '.mov']
        
        video_files = []
        for ext in video_extensions:
            video_files.extend(input_path.rglob(f"*{ext}"))
            
        print(f"Found {len(video_files)} videos to process")
        
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
