import os 
import sys
import json 
import cv2
import numpy as np
from extract_frames import extract_frames, extract_frames_from_videos, create_output_directories
from video_dict import create_video_dict, verify_video_files
from augment_data_no_mediapipe import analyze_class_distribution, balance_dataset, filter_to_top_categories

def main():
    # Get the absolute path to the root directory (one level up from current directory)
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Load the json file
    json_file = os.path.join(root_dir, 'data', 'WLASL_v0.3.json')
    with open(json_file) as f:
        data = json.load(f)

    video_to_label = create_video_dict(data)
    
    videos_folder = os.path.join(root_dir, 'data', 'videos')

    valid_video_to_label = verify_video_files(videos_folder, video_to_label)

    # Base directory to store extracted frames - use a different directory for non-mediapipe version
    base_output_dir = os.path.join(root_dir, 'data', 'extracted_frames_no_mediapipe')

    create_output_directories(base_output_dir, valid_video_to_label)

    extract_frames_from_videos(videos_folder, base_output_dir, valid_video_to_label)
    
    # Analyze class distribution
    print("Analyzing class distribution...")
    distribution = analyze_class_distribution(base_output_dir)
    
    # Print some statistics
    counts = list(distribution.values())
    print(f"Total categories: {len(counts)}")
    print(f"Samples per category - Min: {min(counts)}, Max: {max(counts)}, Mean: {np.mean(counts):.2f}, Median: {np.median(counts)}")
    
    # Automatically filter to top 100 categories
    print("Filtering to top 100 categories...")
    top_categories = filter_to_top_categories(base_output_dir, 100)
    print(f"Dataset filtered to top 100 categories")
    
    # Option to balance the dataset
    balance = input("Do you want to balance the dataset? (y/n): ").lower() == 'y'
    if balance:
        # You can specify a target count or use None for median
        balance_dataset(base_output_dir)
        print("Dataset balanced through augmentation (without MediaPipe)")

if __name__ == "__main__":
    main() 