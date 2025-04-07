# American Sign Language Recognition System

This project implements a deep learning-based system for recognizing American Sign Language (ASL) gestures from video input. The system uses a Long-term Recurrent Convolutional Network (LRCN) architecture that combines CNN and LSTM to process both spatial and temporal features of sign language videos.

## Project Overview

The project consists of several key components:

### 1. Data Processing Pipeline
- **Data Extraction**: Processes raw ASL videos from the WLASL (Word Level American Sign Language) dataset
- **Frame Extraction**: Converts videos into sequences of frames
- **Data Augmentation**: Implements various augmentation techniques to improve model robustness:
  - Spatial augmentations (flips, brightness adjustments)
  - Temporal augmentations
  - Hand landmark augmentations (when using MediaPipe)

### 2. Model Architecture
The system uses an LRCN (Long-term Recurrent Convolutional Network) architecture that consists of:
- **CNN Backbone**: ResNet18 pretrained on ImageNet for spatial feature extraction
- **LSTM Layer**: Processes temporal information across frames
- **Classification Head**: Final fully connected layer for sign classification

### 3. Training Pipeline
- Supports training on both raw frames and MediaPipe-processed hand landmarks
- Implements dataset balancing and filtering to top 100 most common signs
- Uses cross-entropy loss and Adam optimizer
- Includes validation split for model evaluation

### 4. Evaluation System
- Comprehensive evaluation metrics including:
  - Confusion matrix
  - Classification report
  - Per-class accuracy
  - Overall accuracy and loss metrics

### 5. Web Interface
Two versions of the web interface are provided:
- **Standard Version**: Uses raw video frames
- **MediaPipe Version**: Incorporates hand landmark detection

Features:
- YouTube video input support
- Real-time hand landmark visualization
- Top-5 prediction display with confidence scores
- Progress bars for prediction visualization

## Project Structure
├── data/
│ ├── videos/ # Raw ASL videos
│ ├── extracted_frames/ # Processed video frames
│ └── WLASL_v0.3.json # Dataset annotations
├── models/
│ └── lrcn.py # LRCN model implementation
├── datasets/
│ └── video_dataset.py # Custom dataset class
├── preprocess/
│ ├── main.py # Main preprocessing pipeline
│ ├── extract_frames.py # Frame extraction utilities
│ ├── augment_data.py # Data augmentation functions
│ └── video_dict.py # Video-label mapping utilities
├── app.py # Web interface (standard version)
├── app2.py # Web interface (MediaPipe version)
├── train.py # Training script
└── evaluation.py # Model evaluation script

## Key Features

1. **Dual Processing Modes**:
   - Raw frame processing
   - MediaPipe hand landmark processing

2. **Data Augmentation**:
   - Spatial transformations
   - Temporal augmentations
   - Hand landmark augmentations

3. **Model Flexibility**:
   - Supports variable number of classes
   - Configurable architecture parameters
   - Pretrained backbone support

4. **Evaluation Tools**:
   - Comprehensive metrics
   - Visualization capabilities
   - Performance analysis

## Usage

1. **Data Preparation**:

```bash
python preprocess/main.py
```

2. **Training**:
```bash
python train.py
```

3. **Evaluation**:
```bash
python evaluation.py
```

4. **Web Interface**:
```bash
streamlit run app.py  # Standard version
streamlit run app2.py # MediaPipe version
```

## Requirements

- Python 3.7+
- PyTorch
- OpenCV
- MediaPipe
- Streamlit
- NumPy
- scikit-learn

## Future Improvements

1. Real-time video processing
2. Multi-hand gesture support
3. Continuous sign language recognition
4. Improved data augmentation techniques
5. Model architecture optimization


## Acknowledgments

- WLASL dataset
- MediaPipe team
- PyTorch team
