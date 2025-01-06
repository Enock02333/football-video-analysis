from ultralytics import YOLO
import os

import torch
torch.cuda.empty_cache()


if __name__ == '__main__':
    # Path to the dataset
    dataset_path = "C:/Users/enock/Downloads/dataset/Jersey Number detection"

    # Path to data.yaml
    data_yaml = os.path.join(dataset_path, "data.yaml")

    # Initialize and train the YOLO model
    model = YOLO("yolov5n.pt")  # Use YOLOv5s pre-trained weights

    # Training configuration
    model.train(
        data=data_yaml,          # Path to data.yaml
        epochs=10,               # Number of training epochs
        batch=4,                # Batch size
        imgsz=416,               # Image size
        project="runs/train",    # Directory to save training results
        name="jersey_number_detection",  # Name of the run
        device='cuda',
        half=True
    )

    # Evaluate the trained model on the validation set
    val_results = model.val(data=data_yaml)

    # Print validation results
    print("Validation Results:")
    print(val_results)
