# Sign Language Recognition with ConvLSTM / LCRN

This repository presents an end-to-end pipeline for a word-level sign language recognition system built on the WLASL dataset. Using deep learning architectures (ConvLSTM and LRCN), the project covers every stage—from data preprocessing and frame extraction to data augmentation, model training, and evaluation.

---

## Table of Contents

- [Requirements](#requirements)
- [Dataset Preparation](#dataset-preparation)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [1. Preprocessing](#1-preprocessing)
  - [2. Data Augmentation](#2-data-augmentation)
  - [3. Training](#3-training)
  - [4. Evaluation](#4-evaluation)
- [Troubleshooting](#troubleshooting)
- [Citations](#citations)
- [License](#license)

---

## Requirements

- **Python:** 3.8 or higher
- **Key Libraries:**  
  ```bash
  pip install mediapipe opencv-python albumentations numpy scikit-learn matplotlib tqdm torch torchvision
  ```

---

## Dataset Preparation

1. **Download the WLASL Dataset:**  
   Visit the [WLASL GitHub repository](https://github.com/dxli94/WLASL) and download the following files:
   - `WLASL_v0.3.json` (metadata file)
   - `videos/` folder (contains raw videos named by their video IDs, e.g., `07502.mp4`)

2. **Organize the Files:**  
   Structure your project directory as follows:
   ```
   project_root/
   ├── videos/            # Raw video files (e.g., 07502.mp4)
   ├── data/
   │   ├── videos/        # (Optional) Additional organization under data/
   │   └── WLASL_v0.3.json # Metadata file
   ├── extracted_frames/  # Will contain preprocessed frames (organized by label and video ID)
   └── ...
   ```
   For quick experiments, you might also create an `example_videos/` folder with a few sample videos.

---

## Project Structure

The repository is organized in a modular way to help with reproducibility and experimentation:

```
project_root/
├── data/                      # Dataset files (raw videos, metadata)
├── extracted_frames/          # Frames extracted from videos, organized by label/video_id
├── models/                    # Model architectures (e.g., LRCN, ConvLSTM)
├── datasets/                  # Custom dataset implementations (e.g., VideoDataset)
├── utils/                     # Utility scripts for preprocessing and augmentation
│   ├── preprocess.py          # Frame extraction and temporal segmentation
│   └── augment.py             # Data augmentation routines
├── preprocess/                # Additional preprocessing modules
│   ├── main.py                # Entry-point for video processing
│   ├── extract_frames.py      # Functions to extract frames
│   └── video_dict.py          # Functions to create video-to-label mappings
├── train.py                   # Model training script
├── evaluation.py              # Model evaluation script
├── pipeline.ipynb             # Jupyter Notebook for interactive experiments
└── README.md                  # This documentation file
```

---

## Usage

Follow the steps below to run the complete pipeline.

### 1. Preprocessing

Extract frames from videos and perform temporal segmentation (using MediaPipe, if available).

#### Run Frame Extraction

```python
# utils/preprocess.py
import cv2
import os
import json
import mediapipe as mp

# Load video-to-label mapping from metadata
with open('data/WLASL_v0.3.json', 'r') as f:
    video_to_label = json.load(f)

# Process each video
for video_id, label in video_to_label.items():
    video_path = os.path.join('data/videos', f"{video_id}.mp4")
    output_dir = os.path.join('extracted_frames', label, video_id)
    os.makedirs(output_dir, exist_ok=True)
    
    # Use MediaPipe (or a custom function) to detect signing intervals
    start, end = detect_signing_intervals(video_path)
    extract_frames(video_path, output_dir, start, end)
```

Run the script using:
```bash
python utils/preprocess.py
```

---

### 2. Data Augmentation

Apply random augmentations (e.g., horizontal flips, rotations, brightness/contrast adjustments) to enhance the dataset variability.

#### Run Augmentation Script

```python
# utils/augment.py
import os
import numpy as np
import albumentations as A

augmenter = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.2),
])

# Process all augmented frames (assumes frames are stored as .npy files)
for label in os.listdir('extracted_frames'):
    label_dir = os.path.join('extracted_frames', label)
    for video_id in os.listdir(label_dir):
        frames_dir = os.path.join(label_dir, video_id)
        for frame_file in os.listdir(frames_dir):
            if frame_file.endswith('.npy'):
                frame_path = os.path.join(frames_dir, frame_file)
                frame = np.load(frame_path)
                augmented = augmenter(image=frame)
                np.save(os.path.join(frames_dir, f'aug_{frame_file}'), augmented['image'])
```

Execute the augmentation with:
```bash
python utils/augment.py
```

---

### 3. Training

Train the LRCN (or ConvLSTM) model using the preprocessed data. The training script uses the custom `VideoDataset` to load frames and their corresponding labels.

#### Training Script

```python
# train.py
import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from models.lrcn import LRCN
from datasets.video_dataset import VideoDataset

# Hyperparameters & configuration
root_dir = os.path.dirname(os.path.abspath(__file__))
data_root = os.path.join(root_dir, 'data', 'extracted_frames')
num_frames = 16           # Number of frames to sample per video
batch_size = 8
num_epochs = 10
learning_rate = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Label mapping: folder names represent labels
labels = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])
label2idx = {label: idx for idx, label in enumerate(labels)}
num_classes = len(label2idx)
print("Label mapping:", label2idx)

# Image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225]),
])

if __name__ == '__main__':
    # Create training dataset (80% split)
    train_dataset = VideoDataset(root_dir=data_root, label2idx=label2idx,
                                 num_frames=num_frames, transform=transform, split="train")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # Initialize the model, loss function, and optimizer
    model = LRCN(num_classes=num_classes, hidden_size=256, num_layers=1, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for videos, labels in train_loader:
            videos = videos.to(device)
            labels = labels.to(device)
    
            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item() * videos.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = correct / total
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
    
    # Save the trained model state
    torch.save(model.state_dict(), "lrcn_model.pth")
```

Run the training process with:
```bash
python train.py
```

---

### 4. Evaluation

Evaluate your trained model on the validation split of the dataset. This script computes the loss, accuracy, confusion matrix, and a detailed classification report.

#### Evaluation Script

```python
# evaluation.py
import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import transforms
from datasets.video_dataset import VideoDataset
from models.lrcn import LRCN
from sklearn.metrics import confusion_matrix, classification_report

# Configuration
data_root = 'data/extracted_frames'
num_frames = 16
batch_size = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Label mapping
labels = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])
label2idx = {label: idx for idx, label in enumerate(labels)}
idx2label = {idx: label for label, idx in label2idx.items()}
num_classes = len(label2idx)
print("Label mapping:", label2idx)

# Image transforms (must match training)
transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Create validation dataset (20% split)
val_dataset = VideoDataset(root_dir=data_root, label2idx=label2idx,
                           num_frames=num_frames, transform=transform, split="val")
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# Load the trained model
model = LRCN(num_classes=num_classes, hidden_size=256, num_layers=1, pretrained=True).to(device)
model.load_state_dict(torch.load("lrcn_model.pth", map_location=device))
model.eval()

criterion = nn.CrossEntropyLoss()

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for videos, labels in dataloader:
            videos = videos.to(device)
            labels = labels.to(device)
            
            outputs = model(videos)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * videos.size(0)
            
            _, preds = torch.max(outputs, 1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy, all_preds, all_labels

avg_loss, accuracy, all_preds, all_labels = evaluate(model, val_loader, criterion, device)
print(f"Validation Loss: {avg_loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")

cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:")
print(cm)

report = classification_report(all_labels, all_preds, target_names=labels)
print("Classification Report:")
print(report)
```

Run evaluation with:
```bash
python evaluation.py
```

---

## Troubleshooting

- **Permission Errors:**  
  Run scripts with administrator privileges if needed:
  ```bash
  sudo python train.py   # Linux/Mac
  ```

- **Missing or Corrupted Videos:**  
  If frames are missing or warnings appear during preprocessing, check the integrity of the video files using:
  ```bash
  ffmpeg -i <video_file.mp4>
  ```

- **MediaPipe Installation Issues:**  
  Try installing MediaPipe without cache:
  ```bash
  pip install mediapipe --no-cache-dir
  ```

---

## Citations

If you use this code in your research or projects, please cite the following resources:

- **WLASL Dataset:** [D. Li et al., 2020](https://arxiv.org/abs/2004.12355)
- **MediaPipe:** [Google Research](https://mediapipe.dev)
- **Albumentations:** [A. Buslaev et al., 2020](https://arxiv.org/abs/1809.06839)

---

## License

This project is licensed under the [MIT License](LICENSE).  
*(Replace or update the license section as needed.)*

---

**Key Features:**

- **End-to-End Pipeline:** From raw videos to a fully trained sign language recognition model.
- **Modular Codebase:** Separate components for preprocessing, augmentation, training, and evaluation.
- **Detailed Documentation:** Step-by-step instructions and troubleshooting tips.
- **State-of-the-Art Methods:** Utilizes deep learning architectures and advanced data augmentation.

Happy coding and best of luck with your project!