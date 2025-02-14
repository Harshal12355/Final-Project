import torch
import torch.nn as nn
import torchvision.models as models

class LRCN(nn.Module):
    """
    LRCN model that first extracts spatial features from each frame using a CNN
    (here, a pretrained ResNet18) and then models the temporal information with an LSTM.
    
    Args:
      num_classes (int): Number of output classes (unique glosses).
      hidden_size (int): Hidden layer size of the LSTM.
      num_layers (int): Number of LSTM layers.
      pretrained (bool): Whether to use pre-trained weights for the CNN.
    """
    def __init__(self, num_classes, hidden_size=256, num_layers=1, pretrained=True):
        super(LRCN, self).__init__()
        # Use ResNet18 as the CNN feature extractor
        resnet = models.resnet18(pretrained=pretrained)
        # Remove the final fully connected layer.
        modules = list(resnet.children())[:-1]  
        self.cnn = nn.Sequential(*modules)
        self.feature_size = resnet.fc.in_features  # typically 512 for resnet18
        
        # LSTM to process sequential features.
        self.lstm = nn.LSTM(input_size=self.feature_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        """
        Input:
          x: A tensor of shape (batch, seq_len, C, H, W)
        Output:
          out: A tensor of shape (batch, num_classes) with raw class scores.
        """
        batch_size, seq_len, C, H, W = x.size()
        # Merge batch and sequence dimensions; process all frames through the CNN at once.
        x = x.view(batch_size * seq_len, C, H, W)
        features = self.cnn(x)  # shape: (batch * seq_len, feature_size, 1, 1)
        features = features.view(batch_size, seq_len, self.feature_size)
        
        # Pass the sequence of feature vectors through the LSTM.
        lstm_out, _ = self.lstm(features)
        # Use the last time step's output for classification.
        out = self.fc(lstm_out[:, -1, :])
        return out 