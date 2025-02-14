from PIL import Image
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split

class VideoDataset(Dataset):
    """
    This dataset expects that the videos are saved as folders of JPEG frames.
    The directory structure is assumed to be:
      root_dir/
          gloss_1/
              video_id_1/
                  frame001.jpg
                  frame002.jpg
                  ...
              video_id_2/
                  ...
          gloss_2/
              ...
    
    Args:
      root_dir (str): Path to the root directory.
      label2idx (dict): Mapping from gloss (str) to integer label.
      num_frames (int): Number of frames to sample per video.
      transform (callable, optional): Transformations to apply on the images.
      split (str, optional): "train" or "val". If provided splits the data (80% training, 20% validation).
    """
    def __init__(self, root_dir, label2idx, num_frames=16, transform=None, split=None):
        self.root_dir = root_dir
        self.label2idx = label2idx
        self.num_frames = num_frames
        self.transform = transform
        self.videos = []  # list of dictionaries: { 'frames': list_of_frame_paths, 'label': int }
        
        # Walk through the directory structure and collect video data
        for label in os.listdir(root_dir): 
            label_path = os.path.join(root_dir, label)
            if os.path.isdir(label_path):
                for video in os.listdir(label_path):
                    video_path = os.path.join(label_path, video)
                    if os.path.isdir(video_path):
                        # Get all JPEG image file paths and sort them to maintain temporal order
                        frame_files = sorted([
                            os.path.join(video_path, f)
                            for f in os.listdir(video_path)
                            if f.lower().endswith(('.jpeg', '.jpg'))
                        ])
                        # Ensure there are at least num_frames available (skip shorter videos)
                        if len(frame_files) < self.num_frames:
                            continue
                        self.videos.append({
                            'frames': frame_files,
                            'label': self.label2idx[label]
                        })
        
        # If split is specified, partition the videos list into train and validation sets.
        if split is not None:
            train_videos, val_videos = train_test_split(self.videos, test_size=0.2, random_state=42)
            if split == "train":
                self.videos = train_videos
            elif split == "val":
                self.videos = val_videos

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_info = self.videos[idx]
        frames = video_info['frames']
        
        # Uniformly sample exactly num_frames from the total frames
        total_frames = len(frames)
        interval = total_frames // self.num_frames
        selected_frames = [frames[i * interval] for i in range(self.num_frames)]
        
        video_tensor = []
        for frame_path in selected_frames:
            image = Image.open(frame_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            video_tensor.append(image)
        
        # video_tensor shape: (num_frames, C, H, W)
        video_tensor = torch.stack(video_tensor)
        label = video_info['label']
        return video_tensor, label