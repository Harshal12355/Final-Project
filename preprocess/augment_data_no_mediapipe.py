import cv2
import os
import numpy as np
import random
import shutil
from tqdm import tqdm

def analyze_class_distribution(base_dir):
    """
    Analyze the distribution of samples across categories
    
    Args:
        base_dir: Base directory containing class folders
        
    Returns:
        Dictionary mapping category names to sample counts
    """
    distribution = {}
    
    for category in os.listdir(base_dir):
        category_path = os.path.join(base_dir, category)
        if os.path.isdir(category_path):
            # Count video folders in this category
            video_count = len(os.listdir(category_path))
            distribution[category] = video_count
    
    return distribution

def select_top_n_categories(distribution, n=100):
    """
    Select the top N categories with most samples
    
    Args:
        distribution: Dictionary mapping categories to sample counts
        n: Number of top categories to select
        
    Returns:
        List of category names for the top N categories
    """
    sorted_categories = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
    return [category for category, count in sorted_categories[:n]]

def apply_spatial_augmentation(frame):
    """
    Apply various spatial augmentations to a single frame
    
    Args:
        frame: Input image frame
        
    Returns:
        List of augmented frames
    """
    augmented_frames = []
    
    # Horizontal flip (important for sign language as it simulates viewing from different angles)
    flip_frame = cv2.flip(frame, 1)
    augmented_frames.append(flip_frame)
    
    # Brightness adjustment (simulates different lighting conditions)
    brightness_factor = random.uniform(0.8, 1.2)
    brightness_frame = cv2.convertScaleAbs(frame, alpha=brightness_factor, beta=0)
    augmented_frames.append(brightness_frame)
    
    # Small rotation (±15 degrees) - simulates camera angle variations
    angle = random.uniform(-15, 15)
    h, w = frame.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    rotated_frame = cv2.warpAffine(frame, M, (w, h))
    augmented_frames.append(rotated_frame)
    
    # Zoom in slightly (simulates different distances)
    h, w = frame.shape[:2]
    crop_percentage = random.uniform(0.9, 0.95)
    crop_h, crop_w = int(h * crop_percentage), int(w * crop_percentage)
    start_h = random.randint(0, h - crop_h)
    start_w = random.randint(0, w - crop_w)
    zoomed_frame = frame[start_h:start_h+crop_h, start_w:start_w+crop_w]
    zoomed_frame = cv2.resize(zoomed_frame, (w, h))
    augmented_frames.append(zoomed_frame)
    
    # Add slight blur (simulates motion or focus issues)
    blur_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    augmented_frames.append(blur_frame)
    
    # Add slight noise
    noise = np.random.normal(0, 10, frame.shape).astype(np.uint8)
    noise_frame = cv2.add(frame, noise)
    augmented_frames.append(noise_frame)
    
    return augmented_frames

def augment_video_frames(video_dir, output_dir, augmentation_factor=6):
    """
    Augment frames from a video directory
    
    Args:
        video_dir: Directory containing original video frames
        output_dir: Directory to save augmented frames
        augmentation_factor: Number of augmentations to create
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of frame files
    frame_files = sorted([f for f in os.listdir(video_dir) if f.startswith('frame_')])
    
    for frame_file in frame_files:
        # Read the original frame
        frame_path = os.path.join(video_dir, frame_file)
        frame = cv2.imread(frame_path)
        
        if frame is None:
            print(f"Warning: Could not read frame {frame_path}")
            continue
        
        # Save the original frame
        original_output_path = os.path.join(output_dir, frame_file)
        cv2.imwrite(original_output_path, frame)
        
        # Apply augmentations
        augmented_frames = apply_spatial_augmentation(frame)
        
        # Save augmented frames
        for i, aug_frame in enumerate(augmented_frames):
            aug_frame_file = f"aug{i+1}_" + frame_file
            aug_path = os.path.join(output_dir, aug_frame_file)
            cv2.imwrite(aug_path, aug_frame)

def balance_dataset(base_dir, target_count=None, max_augmentation_factor=5):
    """
    Balance the dataset by augmenting minority classes
    
    Args:
        base_dir: Base directory containing class folders
        target_count: Target number of samples per class (if None, uses median)
        max_augmentation_factor: Maximum number of augmented versions per original
    """
    distribution = analyze_class_distribution(base_dir)
    
    # If target_count not provided, use 1.5x the median to ensure adequate samples
    if target_count is None:
        counts = list(distribution.values())
        median_count = np.median(counts)
        target_count = max(int(median_count * 1.5), 20)  # At least 20 samples per class
    
    print(f"Balancing dataset to target {target_count} samples per class")
    
    for category, count in tqdm(distribution.items()):
        if count < target_count:
            category_dir = os.path.join(base_dir, category)
            videos = os.listdir(category_dir)
            
            # Calculate how many augmentations needed
            augmentations_needed = target_count - count
            
            # Keep augmenting until we reach the target
            augmentation_round = 0
            while augmentations_needed > 0 and augmentation_round < max_augmentation_factor:
                # How many videos to augment in this round
                videos_to_augment = min(len(videos), augmentations_needed)
                
                # Select random videos to augment
                videos_for_aug = random.sample(videos, videos_to_augment)
                
                for video_id in videos_for_aug:
                    video_dir = os.path.join(category_dir, video_id)
                    
                    # Create augmented version with a unique identifier
                    aug_video_id = f"{video_id}_aug{augmentation_round}"
                    aug_video_dir = os.path.join(category_dir, aug_video_id)
                    
                    if not os.path.exists(aug_video_dir):
                        # Apply augmentations
                        augment_video_frames(video_dir, aug_video_dir)
                        
                        # Update count of augmentations needed
                        augmentations_needed -= 1
                        
                        # Add the new augmented video to our list of videos
                        videos.append(aug_video_id)
                        
                        if augmentations_needed <= 0:
                            break
                
                augmentation_round += 1
            
            # If we still need more samples, duplicate existing ones
            if augmentations_needed > 0:
                print(f"Warning: Could not create enough augmentations for {category}. Using duplication.")
                all_videos = os.listdir(category_dir)
                for i in range(augmentations_needed):
                    source_video = random.choice(all_videos)
                    source_path = os.path.join(category_dir, source_video)
                    dest_video = f"{source_video}_copy{i}"
                    dest_path = os.path.join(category_dir, dest_video)
                    shutil.copytree(source_path, dest_path)

def filter_to_top_categories(base_dir, n=100):
    """
    Filter dataset to keep only top N categories
    
    Args:
        base_dir: Base directory containing class folders
        n: Number of top categories to keep
        
    Returns:
        List of category names that were kept
    """
    distribution = analyze_class_distribution(base_dir)
    top_categories = select_top_n_categories(distribution, n)
    
    # Create a backup directory
    backup_dir = base_dir + "_full_backup"
    if not os.path.exists(backup_dir):
        print(f"Creating backup of full dataset at {backup_dir}")
        shutil.copytree(base_dir, backup_dir)
    
    # Remove categories not in the top N
    for category in os.listdir(base_dir):
        if category not in top_categories and os.path.isdir(os.path.join(base_dir, category)):
            shutil.rmtree(os.path.join(base_dir, category))
    
    print(f"Dataset filtered to top {n} categories")
    return top_categories 