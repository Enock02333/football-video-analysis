from ultralytics import YOLO
import torch
import yaml
import os

def train_yolo_keypoints():
    # Set the path to your dataset
    dataset_path = r'C:/Users/enock/Downloads/dataset/football-pitch-keypoints-detection'
    
    # Create a new YAML configuration file that's compatible with YOLOv8
    yolo_yaml = {
        'path': dataset_path,
        'train': os.path.join('train', 'images'),
        'val': os.path.join('train', 'images'),  # Using train as val since val isn't specified
        'test': '',
        
        # Dataset specific configurations
        'nc': 1,  # number of classes
        'names': ['pitch'],  # class names
        'kpt_shape': [32, 3],  # number of keypoints, number of dims (x,y,conf)
    }
    
    # Save the YAML configuration
    yaml_path = os.path.join(dataset_path, 'yolo_data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yolo_yaml, f)

    # Check available memory and set device
    if torch.cuda.is_available():
        # Get GPU memory info
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # Convert to GB
        print(f"Total GPU Memory: {gpu_memory:.2f} GB")
        
        # Set device-specific parameters
        if gpu_memory < 6:  # For GPUs with less than 6GB memory
            device = 0
            batch_size = 4
            image_size = 416
        else:
            device = 0
            batch_size = 16
            image_size = 640
    else:
        device = 'cpu'
        batch_size = 2
        image_size = 416

    # Initialize YOLO model
    model = YOLO('yolov8n-pose.pt')  # Load the smallest pose estimation model

    # Configure PyTorch to be memory efficient
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Set memory allocator settings to reduce fragmentation
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'

    try:
        # Train the model with memory-optimized parameters
        results = model.train(
            data=yaml_path,
            epochs=20,
            imgsz=image_size,
            batch=batch_size,
            name='football_pitch_keypoints',
            device=device,
            patience=50,  # Early stopping patience
            save=True,  # Save best model
            plots=True,  # Generate training plots
            cache=False,  # Disable caching to reduce memory usage
            amp=True,  # Enable automatic mixed precision for memory efficiency
            workers=4,  # Reduce number of workers
            resume=False  # Don't resume from last checkpoint
        )
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("WARNING: Out of memory error occurred. Trying with smaller batch size...")
            torch.cuda.empty_cache()
            # Retry with even smaller batch size
            results = model.train(
                data=yaml_path,
                epochs=100,
                imgsz=320,  # Further reduce image size
                batch=2,    # Minimal batch size
                name='football_pitch_keypoints',
                device=device,
                patience=50,
                save=True,
                plots=True,
                cache=False,
                amp=True,
                workers=2
            )
        else:
            raise e

def predict_keypoints(image_path):
    # Load the trained model
    model = YOLO('runs/pose/football_pitch_keypoints/weights/best.pt')
    
    # Perform prediction with memory-efficient settings
    results = model(image_path, device='cpu' if not torch.cuda.is_available() else 0)
    
    return results

def visualize_predictions(image_path):
    # Load and run prediction
    results = predict_keypoints(image_path)
    
    # Plot results
    results[0].plot()  # YOLOv8 has built-in visualization

if __name__ == '__main__':
    try:
        # Train the model
        train_yolo_keypoints()
        
        # Example of prediction and visualization
        # visualize_predictions('path_to_test_image.jpg')
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        if torch.cuda.is_available():
            print(f"Current GPU memory usage: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
            print(f"GPU memory cached: {torch.cuda.memory_reserved()/1024**3:.2f} GB")