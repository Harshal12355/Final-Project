import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from models.lrcn import LRCN
import mediapipe as mp
from collections import deque
import time
import os 

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

# Load the model and labels
def load_model():
    # Get labels from the extracted_frames directory
    data_root = 'data/extracted_frames'
    labels = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])
    label2idx = {label: idx for idx, label in enumerate(labels)}
    idx2label = {idx: label for label, idx in label2idx.items()}
    num_classes = len(label2idx)
    
    # Load model
    model = LRCN(num_classes=num_classes, hidden_size=256, num_layers=1, pretrained=True).to(device)
    model.load_state_dict(torch.load("lrcn_model_100c_no_mediapipe.pth", 
                                   map_location=device,
                                   weights_only=True))
    model.eval()
    
    return model, idx2label

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225]),
])

def main():
    # Load model and labels
    model, idx2label = load_model()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Initialize frame buffer
    frame_buffer = deque(maxlen=16)  # Store 16 frames for classification
    
    # Initialize prediction variables
    last_prediction = None
    last_prediction_time = time.time()
    prediction_cooldown = 2.0  # Seconds between predictions
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Flip frame horizontally for a laterally correct view
        frame = cv2.flip(frame, 1)
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = hands.process(rgb_frame)
        
        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Add frame to buffer
        frame_buffer.append(frame)
        
        # Make prediction when buffer is full and cooldown has passed
        current_time = time.time()
        if len(frame_buffer) == 16 and (current_time - last_prediction_time) >= prediction_cooldown:
            # Prepare frames for model
            frames_tensor = []
            for f in frame_buffer:
                # Convert to PIL Image
                pil_image = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
                # Apply transforms
                tensor_image = transform(pil_image)
                frames_tensor.append(tensor_image)
            
            # Stack frames and add batch dimension
            frames_tensor = torch.stack(frames_tensor).unsqueeze(0).to(device)
            
            # Make prediction
            with torch.no_grad():
                outputs = model(frames_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                top5_prob, top5_indices = torch.topk(probabilities, 5)
                
                # Get top prediction
                top_idx = top5_indices[0][0].item()
                top_prob = top5_prob[0][0].item() * 100
                
                if top_prob > 50:  # Only show predictions with >50% confidence
                    last_prediction = (idx2label[top_idx], top_prob)
                    last_prediction_time = current_time
        
        # Display prediction
        if last_prediction and (current_time - last_prediction_time) < prediction_cooldown:
            label, prob = last_prediction
            cv2.putText(frame, f"{label}: {prob:.1f}%", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow('Sign Language Recognition', frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 