import cv2
import os

# Function to extract frames from a video
def extract_frames(video_path, output_dir):
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

def create_output_directories(base_output_dir, valid_video_to_label):
    # Create directories for each gloss
        for label in set(valid_video_to_label.values()):
            os.makedirs(os.path.join(base_output_dir, label), exist_ok=True)

def extract_frames_from_videos(videos_folder, base_output_dir, valid_video_to_label):
    # Process each video
    for video_id, label in valid_video_to_label.items():
        video_path = os.path.join(videos_folder, f"{video_id}.mp4")
        output_dir = os.path.join(base_output_dir, label, video_id)
        
        # Create directory for this specific video's frames
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Extracting frames from video {video_id} with label {label}")
        extract_frames(video_path, output_dir)
    
    
