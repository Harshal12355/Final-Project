import json
import os 

def create_video_dict(data):
    # Create a dictionary to map video IDs to labels
    video_to_label = {}

    for entry in data:
        label = entry['gloss']  # The word or label
        for instance in entry['instances']:
            video_id = instance['video_id']
            video_to_label[video_id] = label
    
    return video_to_label

def verify_video_files(path, video_to_label):
    # Path to the videos folder
    videos_folder = path

    # Verify that each video ID in the mapping has a corresponding video file
    for video_id in video_to_label.keys():
        video_path = os.path.join(videos_folder, f"{video_id}.mp4")
        if not os.path.exists(video_path):
            print(f"Warning: Video file for ID {video_id} does not exist.")
        else:
            print(f"Video file for ID {video_id} exists and is correctly mapped to label {video_to_label[video_id]}.")

    valid_video_to_label = {video_id: label for video_id, label in video_to_label.items() if os.path.exists(os.path.join(videos_folder, f"{video_id}.mp4"))}

    return valid_video_to_label
    