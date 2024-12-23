# Football Video Analysis

A computer vision project aimed at analyzing football videos to detect and track players, classify teams, recognize jersey numbers, identify key events, and segment the football pitch. This repository demonstrates the integration of YOLO-based object detection models for comprehensive football analytics.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [Results](#results)
- [Future Work](#future-work)

---

## Overview

### Purpose

This project automates the analysis of football game videos using object detection and classification models. It encompasses the following key objectives:

- **Player Detection & Tracking:** Identify and track players across frames.
- **Team Classification:** Classify players into their respective teams.
- **Jersey Number Recognition:** Recognize jersey numbers of players.
- **Event Detection:** Detect specific actions like goals, tackles, or passes.
- **Pitch Detection and Segmentation:** Detect the football pitch and its key points.

### Key Features

- **Multi-Task YOLO Models:** Utilizes YOLOv8-based models fine-tuned for each task.
- **Video Analysis:** Annotates videos with detected objects and events.
- **Comprehensive Outputs:** Provides metrics, graphs, and annotated results.

---

## Dataset

### Structure

```plaintext
Data/
├── keyMoment/
├── football-player-detection/
├── Football-Teams-Classification/
├── football-pitch-keypoints-detection/
├── football-pitch-segmentation/
├── Jersey-number-detection/
```

Each subfolder contains:

- `data.yaml`: Configuration file.
- `train/`: Training data with `images/` and `labels/` subfolders.
- `valid/`: Validation data with `images/` and `labels/` subfolders.
- `test/`: Testing data with `images/` and `labels/` subfolders.

### Details

- **Annotations:** Includes bounding boxes and keypoints.
- **Format:** YOLO-compatible.
- **Classes:** Specific to each task (e.g., players, teams, jersey numbers).

---

## Project Structure

```plaintext
Football-Video-Analysis/
├── Data/                # Dataset folders
├── src/                 # Python scripts for training and testing
│   ├── model-training-on-images/
│   │   ├── key_moment_detection.py
│   │   ├── football_player_model.py
│   │   ├── team_classification.py
│   │   ├── pitch_keypoints.py
│   │   ├── pitch_segmentation.py
│   │   ├── jersey_number.py
│   ├── model-testing-on-videos/
│       ├── player_detection.py
│       ├── jersey_number.py
│       ├── keypoints_detection_on_videos.py
├── Model/               # Pretrained and Trained Models
│   ├── Pretrained-weights/
│   ├── Trained-Models/
│       ├── player_detection.pt
│       ├── team_classification.pt
│       ├── football_pitch_keypoints.pt
│       ├── pitch_detection.pt
│       ├── key_moment_detection.pt
│       ├── jersey_number_detection.pt
├── Results/             # Training and evaluation results
│   ├── Player_detection/
│   ├── Team_classification/
│   ├── Pitch_detection/
│   ├── Key_moment_detection/
│   ├── Jersey_number_detection/
│   ├── Football_pitch_keypoints/
├── Videos/              # Video data
│   ├── train/
│   ├── test/
│   ├── annotated/
```

---

## Setup and Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.8 or higher
- NVIDIA CUDA (for GPU acceleration)
- Git

### Installation Steps

1. **Clone this repository:**
   ```bash
   git clone https://github.com/Enock02333/football-video-analysis.git
   cd football-video-analysis
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv football-env
   source football-env/bin/activate  # Windows: football-env\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset:**
   - Place the dataset in the `Data/` folder, structured as described above.

---

## Training

Train the YOLO models for specific tasks using the provided scripts:

```bash
python src/model-training-on-images/<task_script>.py
```

### Key Parameters:

- `epochs`: Number of epochs (default: 100).
- `batch_size`: Training batch size (default: 8).
- `imgsz`: Image size (default: 416).
- `weights`: Path to pre-trained weights.

---

## Evaluation

Evaluate the performance of the trained models:

```bash
python src/model-testing-on-videos/<evaluation_script>.py
```

### Outputs:

- **Confusion Matrix**
- **Precision-Recall Curves**
- **Annotated Videos**

Results are saved in the `Results/` and `Videos/annotated/` folders.

---

## Inference

Perform inference on test videos:

```bash
python src/model-testing-on-videos/<inference_script>.py
```

The processed videos with annotations are saved in the `Videos/annotated/` folder.

---

## Results

- **Player Detection:** Achieved high accuracy in identifying and tracking players.
- **Team Classification:** Successfully classified players into teams based on jerseys.
- **Jersey Number Recognition:** Detected and recognized player jersey numbers.
- **Event Detection:** Identified key events like goals and tackles.
- **Pitch Detection and Segmentation:** Accurately segmented and identified pitch areas.

---

## Future Work

- Integrate real-time video analysis for live matches.
- Expand dataset for diverse leagues and environments.
- Improve model performance on edge cases (e.g., occlusions, low-light conditions).

