from ultralytics import YOLO
import os
import torch

torch.cuda.empty_cache()

if __name__ == '__main__':
    # Path to the dataset
    dataset_path = "C:/Users/enock/Downloads/dataset/football-pitch-keypoints-detection"

    # Path to data.yaml
    data_yaml = os.path.join(dataset_path, "data.yaml")

    # Initialize and train the YOLO model for keypoint detection
    model = YOLO("yolov8n-pose.pt")  # Pre-trained YOLOv8 model optimized for pose detection tasks

    # Training configuration
    model.train(
        data=data_yaml,                  # Path to data.yaml
        epochs=50,                       # Number of training epochs
        batch=4,                        # Batch size
        imgsz=512,                       # Image size
        task='pose',                     # Enable keypoint detection task
        project="runs/train",            # Directory to save training results
        name="pitch_keypoint_detection", # Name of the run
        device='cuda',                   # Use GPU for training
        half=True                        # Use half precision for faster training
    )

    # Evaluate the model on the validation set
    val_results = model.val(data=data_yaml)

    # Print validation and test results
    print("Validation Results:")
    print(val_results)
