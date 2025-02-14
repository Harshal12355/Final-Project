import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from models.lrcn import LRCN
from datasets.video_dataset import VideoDataset

# -------------------------
# Hyperparameters & Settings
# -------------------------
root_dir = os.path.dirname(os.path.abspath(__file__))
data_root = os.path.join(root_dir, 'data', 'extracted_frames')
num_frames = 16           # Number of frames to sample per video.
batch_size = 8
num_epochs = 10
learning_rate = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Creating Label Mapping
# -------------------------
# Each subfolder of data_root represents a gloss.
labels = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])
label2idx = {label: idx for idx, label in enumerate(labels)}
num_classes = len(label2idx)
print("Label mapping:", label2idx)

# -------------------------
# Define Image Transforms
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225]),
])

if __name__ == '__main__':
    # -------------------------
    # Create the Training Dataset & DataLoader (80% of data)
    # -------------------------
    train_dataset = VideoDataset(root_dir=data_root, label2idx=label2idx,
                                 num_frames=num_frames, transform=transform, split="train")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # -------------------------
    # Instantiate the Model, Loss, and Optimizer
    # -------------------------
    model = LRCN(num_classes=num_classes, hidden_size=256, num_layers=1, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # -------------------------
    # Training Loop
    # -------------------------
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
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
    
    # Save the trained model's state dictionary.
    torch.save(model.state_dict(), "lrcn_model.pth")