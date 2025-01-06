from ultralytics import YOLO
import os
import torch

torch.cuda.empty_cache()

if __name__ == '__main__':
    # Path to the dataset
    dataset_path = "C:/Users/enock/Downloads/dataset/keyMoment"

    # Path to data.yaml
    data_yaml = os.path.join(dataset_path, "data.yaml")

    # Initialize and train the YOLO model
    model = YOLO("yolov8m.pt")  # Use YOLOv8m pre-trained weights for more complex tasks

    # Training configuration
    model.train(
        data=data_yaml,          # Path to data.yaml
        epochs=10,               # Number of training epochs
        batch=4,                 # Batch size
        imgsz=416,               # Image size
        project="runs/train",    # Directory to save training results
        name="key_moment_detection",  # Name of the run
        device='cuda',
        half=True
    )

    # Evaluate the model on the validation set
    val_results = model.val(data=data_yaml)

    # Print validation and test results
    print("Validation Results:")
    print(val_results)