import cv2
import torch
from ultralytics import YOLO
import numpy as np
import os

def load_models(player_detection_path, team_classification_path):
    """Load the detection and classification models."""
    detection_model = YOLO(player_detection_path)
    classification_model = YOLO(team_classification_path)
    return detection_model, classification_model

def classify_team(cropped_image, classification_model):
    """Classify a cropped player image into Team-A or Team-B."""
    results = classification_model.predict(source=cropped_image, imgsz=416, conf=0.5, device='cuda', half=True)
    if results and len(results[0].boxes):
        # Take the highest confidence classification
        class_id = int(results[0].boxes[0].cls[0])
        return classification_model.names[class_id]
    return "Unknown"

def process_videos(input_folder, output_folder, detection_model, classification_model):
    """Process all videos in the input folder to detect players and classify teams."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for video_name in os.listdir(input_folder):
        input_video_path = os.path.join(input_folder, video_name)
        output_video_path = os.path.join(output_folder, f"annotated_{video_name}")

        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {input_video_path}")
            continue

        # Video writer setup
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect players, referees, and goalkeepers
            results = detection_model.predict(source=frame, imgsz=416, conf=0.5, device='cuda', half=True, verbose=False)
            detections = results[0].boxes

            if detections is not None:
                for det in detections:
                    x1, y1, x2, y2 = map(int, det.xyxy[0])  # Bounding box coordinates
                    cls = int(det.cls[0])  # Class ID
                    conf = float(det.conf[0])  # Confidence score

                    if detection_model.names[cls] in ['football-players', 'goalkeeper', 'referee']:
                        # Crop the player region for classification
                        cropped_player = frame[y1:y2, x1:x2]
                        if cropped_player.size > 0:
                            # Classify the cropped player
                            team_label = classify_team(cropped_player, classification_model)

                            # Annotate the frame
                            color = (0, 255, 0) if team_label == 'Team-A' else (255, 0, 0)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(frame, f"{team_label} ({detection_model.names[cls]})", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Write the annotated frame to the output video
            out.write(frame)

        cap.release()
        out.release()
        print(f"Annotated video saved at {output_video_path}")

if __name__ == "__main__":
    # Paths to models
    player_detection_model_path = "C:/Users/enock/Downloads/dataset/runs/train/player_detection/weights/best.pt"
    team_classification_model_path = "C:/Users/enock/Downloads/dataset/runs/train/team_classification/weights/best.pt"

    # Input and output video folder paths
    input_folder = "C:/Users/enock/Downloads/dataset/videos/train"
    output_folder = "C:/Users/enock/Downloads/dataset/videos/annotated"

    # Load models
    detection_model, classification_model = load_models(player_detection_model_path, team_classification_model_path)

    # Process the videos
    process_videos(input_folder, output_folder, detection_model, classification_model)
