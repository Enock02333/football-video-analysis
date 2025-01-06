import os
from ultralytics import YOLO
import torch

# Clear CUDA cache to free up memory
torch.cuda.empty_cache()

# Function to train the YOLO model
def train_team_classification_model():
    # Path to the dataset and its configuration
    dataset_path = "C:/Users/enock/Downloads/dataset/Football Teams Classification"
    data_yaml = os.path.join(dataset_path, "data.yaml")

    # Initialize the YOLO model
    model = YOLO("yolov8n.pt")  # Use YOLOv8n as the base model

    # Training configuration
    model.train(
        data=data_yaml,              # Path to data.yaml
        epochs=50,                   # Number of training epochs
        batch=4,                     # Batch size
        imgsz=416,                   # Image size
        device="cuda",              # Use GPU for training
        half=True,                   # Enable mixed precision for faster training
        project="runs/train",       # Directory to save training results
        name="team_classification", # Name of the training run
    )

    # Validate the trained model
    val_results = model.val(data=data_yaml)

    # Print validation results
    print("Validation Results:")
    print(val_results)

if __name__ == "__main__":
    train_team_classification_model()
