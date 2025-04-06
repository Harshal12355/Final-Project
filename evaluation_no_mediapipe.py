import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import transforms
from datasets.video_dataset import VideoDataset
from models.lrcn import LRCN
from sklearn.metrics import confusion_matrix, classification_report

# -------------------------
# Settings and Hyperparameters
# -------------------------
data_root = 'data/extracted_frames_no_mediapipe'
num_frames = 16
batch_size = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Label Mapping
# -------------------------
labels = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])
label2idx = {label: idx for idx, label in enumerate(labels)}
# Create reverse mapping for reporting.
idx2label = {idx: label for label, idx in label2idx.items()}
num_classes = len(label2idx)
print(f"Number of classes: {num_classes}")
print("Label mapping:", label2idx)

# -------------------------
# Define Transforms (Same as Training)
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# -------------------------
# Create the Validation Dataset & DataLoader (20% of data)
# -------------------------
val_dataset = VideoDataset(root_dir=data_root, label2idx=label2idx,
                           num_frames=num_frames, transform=transform, split="val")
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# -------------------------
# Load the Trained Model
# -------------------------
model = LRCN(num_classes=num_classes, hidden_size=256, num_layers=1, pretrained=True).to(device)

try:
    model.load_state_dict(torch.load("lrcn_model_100c_no_mediapipe.pth", map_location=device))
    print("Loaded model trained on 100 classes (no MediaPipe)")
except:
    print("No model found for 100 classes without MediaPipe.")

model.eval()  # Set the model to evaluation mode.

criterion = nn.CrossEntropyLoss()

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for videos, labels in dataloader:
            videos = videos.to(device)
            labels = labels.to(device)
            
            outputs = model(videos)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * videos.size(0)
            
            _, preds = torch.max(outputs, 1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy, all_preds, all_labels

print("Starting evaluation...")
avg_loss, accuracy, all_preds, all_labels = evaluate(model, val_loader, criterion, device)
print(f"Validation Loss: {avg_loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")

cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:")
print(cm)

# Print classification report with class names
report = classification_report(all_labels, all_preds, target_names=labels)
print("Classification Report:")
print(report) 