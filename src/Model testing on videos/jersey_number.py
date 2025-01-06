import cv2
import os
from ultralytics import YOLO

def process_videos_in_folder(input_folder, output_folder, model_path):
    # Load the trained YOLO model
    model = YOLO(model_path)
    
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over all video files in the input folder
    for video_file in os.listdir(input_folder):
        video_path = os.path.join(input_folder, video_file)
        if not video_file.endswith(".mp4"):  # Process only .mp4 files
            continue
        
        # Open the video file
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            print(f"Error: Could not open video {video_path}")
            continue

        # Get video properties
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video.get(cv2.CAP_PROP_FPS))
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Prepare output video writer
        output_video_path = os.path.join(output_folder, f"annotated_{video_file}")
        output_video = cv2.VideoWriter(
            output_video_path, 
            cv2.VideoWriter_fourcc(*"mp4v"), 
            fps, 
            (width, height)
        )

        print(f"Processing {video_file}...")

        for frame_idx in range(frame_count):
            ret, frame = video.read()
            if not ret:
                break

            # Run detection on the frame
            results = model.predict(frame, conf=0.25, imgsz=416)

            # Annotate the frame with detected jersey numbers
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                confidence = box.conf[0]               # Confidence score
                class_id = int(box.cls[0])             # Class ID

                # Draw the bounding box and class ID on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
                cv2.putText(frame, f"Class: {class_id}, Conf: {confidence:.2f}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Write the annotated frame to the output video
            output_video.write(frame)

            # Display progress
            print(f"Processing frame {frame_idx + 1}/{frame_count} of {video_file}", end="\r")

        # Release resources for the video
        video.release()
        output_video.release()
        print(f"\nCompleted processing {video_file}. Output saved to {output_video_path}")

if __name__ == "__main__":
    # Define paths
    input_folder = "C:/Users/enock/Downloads/dataset/videos/train"  # Path to input video folder
    output_folder = "C:/Users/enock/Downloads/dataset/videos/annotated"  # Path to save annotated videos
    model_path = "C:/Users/enock/Downloads/dataset/runs/train/jersey_number_detection2/weights/best.pt"  # Path to YOLO model weights

    # Process videos
    process_videos_in_folder(input_folder, output_folder, model_path)
