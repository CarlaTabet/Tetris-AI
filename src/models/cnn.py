import torch
import torch.nn as nn

'''Defines the CNN for encoding the board state into feature vectors.'''

class TetrisCNN(nn.Module):
    def __init__(self):
        super(TetrisCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 20 * 10, 128)
        self.fc2 = nn.Linear(128, 64)  # Feature vector output

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(-1, 64 * 20 * 10)  # Flatten for the fully connected layer
        x = torch.relu(self.fc1(x))
        return self.fc2(x)  # Output the feature vector
