import streamlit as st
import torch
import cv2
import numpy as np
import os
import tempfile
import pytube
from torchvision import transforms
from PIL import Image
from models.lrcn import LRCN
import mediapipe as mp
import shutil

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
@st.cache_resource
def load_model():
    # Get labels from the extracted_frames directory
    data_root = 'data/extracted_frames'
    labels = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])
    label2idx = {label: idx for idx, label in enumerate(labels)}
    idx2label = {idx: label for label, idx in label2idx.items()}
    num_classes = len(label2idx)
    
    # Load model
    model = LRCN(num_classes=num_classes, hidden_size=256, num_layers=1, pretrained=True).to(device)
    
    try:
        # Try to load the 100-class model first
        model.load_state_dict(torch.load("lrcn_model_100c.pth", map_location=device))
        st.success("Loaded model trained on 100 classes")
    except:
        st.warning("No 100-class model found. Trying to load alternative model...")
        try:
            model.load_state_dict(torch.load("lrcn_model_100classes.pth", map_location=device))
            st.success("Loaded alternative 100-class model")
        except:
            st.error("Failed to load any model. Please ensure model files exist.")
    
    model.eval()
    return model, idx2label, label2idx

# Function to download YouTube video
def download_youtube_video(url):
    try:
        yt = pytube.YouTube(url)
        stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
        
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_file.close()
        
        # Download the video to the temporary file
        stream.download(filename=temp_file.name)
        
        return temp_file.name, yt.title
    except Exception as e:
        st.error(f"Error downloading video: {str(e)}")
        st.info("YouTube may have changed their API. Try uploading a video file directly instead.")
        return None, None

# Function to extract frames from a video (similar to extract_frames in preprocess/extract_frames.py)
def extract_frames(video_path, output_dir):
    """Extract frames from a video file and save them to the output directory"""
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Save frame
        frame_path = os.path.join(output_dir, f'frame_{frame_count:04d}.jpg')
        cv2.imwrite(frame_path, frame)
        frame_count += 1
        
    cap.release()
    return frame_count

# Function to load frames from directory (similar to VideoDataset)
def load_frames_from_directory(frames_dir, num_frames=16):
    """Load frames from directory and preprocess them for the model"""
    # Get all frame files
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.startswith('frame_')])
    
    # If we have more frames than needed, sample evenly
    if len(frame_files) > num_frames:
        indices = np.linspace(0, len(frame_files)-1, num_frames, dtype=int)
        frame_files = [frame_files[i] for i in indices]
    
    # If we have fewer frames than needed, duplicate the last frame
    while len(frame_files) < num_frames:
        frame_files.append(frame_files[-1] if frame_files else None)
    
    # Load and preprocess frames
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    frames = []
    for frame_file in frame_files:
        if frame_file:
            frame_path = os.path.join(frames_dir, frame_file)
            # Load image using PIL
            img = Image.open(frame_path).convert('RGB')
            # Apply transformations
            img_tensor = transform(img)
            frames.append(img_tensor)
        else:
            # Create a blank frame if needed
            blank = torch.zeros(3, 224, 224)
            frames.append(blank)
    
    # Stack frames into a tensor of shape [num_frames, channels, height, width]
    frames_tensor = torch.stack(frames)
    # Add batch dimension
    frames_tensor = frames_tensor.unsqueeze(0)
    
    return frames_tensor, frame_files

# Function to visualize hand landmarks
def visualize_hands(frames_dir, frame_files):
    """Visualize hand landmarks on frames"""
    frames_with_landmarks = []
    
    for frame_file in frame_files:
        frame_path = os.path.join(frames_dir, frame_file)
        frame = cv2.imread(frame_path)
        if frame is None:
            continue
            
        # Convert BGR to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = hands.process(frame_rgb)
        
        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame_rgb, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
        
        frames_with_landmarks.append(frame_rgb)
    
    return frames_with_landmarks

# Main Streamlit app
def main():
    st.title("ASL Sign Language Recognition with Mediapipe")
    st.write("Upload a video of American Sign Language to classify the sign")
    
    # Load model
    with st.spinner("Loading model..."):
        model, idx2label, label2idx = load_model()
    
    # Add tabs for different input methods
    tab1, tab2 = st.tabs(["YouTube URL", "Upload Video"])
    
    with tab1:
        # YouTube URL input
        youtube_url = st.text_input("Enter YouTube URL:")
        
        if st.button("Analyze YouTube Video"):
            if youtube_url:
                with st.spinner("Downloading YouTube video..."):
                    video_path, video_title = download_youtube_video(youtube_url)
                    
                    if video_path:
                        process_video(video_path, video_title, model, idx2label)
                        # Clean up the temporary file
                        os.unlink(video_path)
            else:
                st.warning("Please enter a YouTube URL")
    
    with tab2:
        # Direct video upload
        uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
        
        if uploaded_file is not None:
            # Save the uploaded file to a temporary location
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            temp_file.write(uploaded_file.read())
            temp_file.close()
            
            if st.button("Analyze Uploaded Video"):
                process_video(temp_file.name, uploaded_file.name, model, idx2label)
                # Clean up the temporary file
                os.unlink(temp_file.name)

# Function to process video (extracted to avoid code duplication)
def process_video(video_path, video_title, model, idx2label):
    st.success(f"Processing: {video_title}")
    
    # Create temporary directory for frames
    temp_frames_dir = tempfile.mkdtemp()
    
    try:
        # Extract frames (similar to main.py)
        with st.spinner("Extracting frames..."):
            num_frames = extract_frames(video_path, temp_frames_dir)
            
            if num_frames == 0:
                st.error("Could not extract frames from the video.")
                return
            
            st.success(f"Extracted {num_frames} frames")
        
        # Load and preprocess frames
        with st.spinner("Preprocessing frames..."):
            frames_tensor, frame_files = load_frames_from_directory(temp_frames_dir)
            
            # Visualize hands on sample frames
            frames_with_landmarks = visualize_hands(temp_frames_dir, frame_files[:8])  # Show first 8 frames
            
            # Display frames with landmarks
            st.write("Sample frames with hand landmarks:")
            cols = st.columns(4)
            for i, col in enumerate(cols):
                if i < len(frames_with_landmarks):
                    col.image(frames_with_landmarks[i], use_container_width=True)
        
        # Run inference
        with st.spinner("Classifying sign..."):
            with torch.no_grad():
                frames_tensor = frames_tensor.to(device)
                outputs = model(frames_tensor)
                
                # Get top 5 predictions
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                top5_prob, top5_indices = torch.topk(probabilities, 5)
                
                # Display results
                st.subheader("Classification Results:")
                
                results_container = st.container()
                with results_container:
                    for i in range(5):
                        idx = top5_indices[0][i].item()
                        prob = top5_prob[0][i].item() * 100
                        st.write(f"{idx2label[idx]}: {prob:.2f}%")
                        
                        # Create a progress bar for visualization
                        st.progress(prob / 100)
    
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_frames_dir, ignore_errors=True)

if __name__ == "__main__":
    main()