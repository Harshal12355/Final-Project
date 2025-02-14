import os 
import sys
import json 
import cv2
import numpy as np
from extract_frames import extract_frames, extract_frames_from_videos, create_output_directories
from video_dict import create_video_dict, verify_video_files

def main():
    # Check if data directory exists, create if not
    # if not os.path.exists('data'):
    #     print('Error: data directory not found')
    #     return
    # Get the absolute path to the root directory (one level up from current directory)
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Load the json file
    json_file = os.path.join(root_dir, 'data', 'WLASL_v0.3.json')
    with open(json_file) as f:
        data = json.load(f)

    video_to_label = create_video_dict(data)
    
    videos_folder = os.path.join(root_dir, 'data', 'videos')

    valid_video_to_label = verify_video_files(videos_folder, video_to_label)

    # Base directory to store extracted frames
    base_output_dir = os.path.join(root_dir, 'data', 'extracted_frames')

    create_output_directories(base_output_dir, valid_video_to_label)

    extract_frames_from_videos(videos_folder, base_output_dir, valid_video_to_label)

if __name__ == "__main__":
    main()

