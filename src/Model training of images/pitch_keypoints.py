import os
from ultralytics import YOLO
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

torch.cuda.empty_cache()

class CustomKeypointsDataset(Dataset):
    def __init__(self, dataset_path, keypoints=32, transform=None):
        self.dataset_path = dataset_path
        self.keypoints = keypoints
        self.transform = transform
        self.img_dir = os.path.join(dataset_path, "train", "images")
        self.label_dir = os.path.join(dataset_path, "train", "labels")
        self.image_files = [f for f in os.listdir(self.img_dir) if f.endswith('.png')]
        
    def normalize_labels_on_the_fly(self, label_path):
        """
        Normalize the labels by clipping the values to the range [0, 1] during training.
        Does not modify the original label file.
        """
        with open(label_path, "r") as file:
            lines = file.readlines()
        
        normalized_lines = []
        for line in lines:
            parts = line.strip().split()
            cls_id = parts[0]
            coords = list(map(float, parts[1:]))
            
            # Clip values to range [0, 1]
            normalized_coords = [min(max(c, 0.0), 1.0) for c in coords]
            normalized_lines.append(f"{cls_id} " + " ".join(map(str, normalized_coords)))
        
        return " ".join(normalized_lines)
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_file)
        label_path = os.path.join(self.label_dir, img_file.replace('.png', '.txt'))
        
        # Load image
        img = Image.open(img_path).convert("RGB")
        
        # Normalize labels
        normalized_label = self.normalize_labels_on_the_fly(label_path)
        
        # Apply any additional transforms (e.g., data augmentation) if provided
        if self.transform:
            img = self.transform(img)
        
        return img, normalized_label


def train_model(dataset_path, keypoints=32, epochs=20, batch_size=4, imgsz=416):
    # Initialize the model (YOLOv8 for keypoint detection)
    model = YOLO("yolov8n.pt")  # Using YOLOv8n for the keypoint detection task
    
    # Create the custom dataset
    dataset = CustomKeypointsDataset(dataset_path)
    
    # Create DataLoader for batch processing
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Training configuration
    model.train(
        data=os.path.join(dataset_path, "data.yaml"),  # Path to data.yaml
        epochs=epochs,           # Number of training epochs
        batch=batch_size,        # Batch size
        imgsz=imgsz,             # Image size
        task='train',            # Enable training task
        project="runs/train",    # Directory to save training results
        name="pitch_keypoints",  # Name of the run
        device='cuda',           # Training on GPU
        half=True,               # Enable mixed precision
        train_loader=train_loader  # Use the custom data loader
    )

    # Evaluate the model on the validation set
    val_results = model.val(data=os.path.join(dataset_path, "data.yaml"))

    # Print validation and test results
    print("Validation Results:")
    print(val_results)

if __name__ == '__main__':
    dataset_path = "C:/Users/enock/Downloads/dataset/football-pitch-keypoints-detection"
    train_model(dataset_path)
