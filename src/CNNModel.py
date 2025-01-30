import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, num_points=10):
        super(CNNModel, self).__init__()
        self.num_points = num_points

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Mean pooling layer
        self.mean_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 16 * 16, 512)  
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 2 * num_points)

    def forward(self, x):
        # Pass through convolutional layers
        x = F.relu(self.conv1(x))  # First conv
        x = F.relu(self.conv2(x))  # Second conv

        # Apply mean pooling at the end of convolutional layers
        x = self.mean_pool(x)

        # Flatten the feature map for linear layers
        x = x.view(x.size(0), -1)

        # Fully connected layers with ReLU activations
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        # Reshape the final output to (batch_size, num_points, 2)
        return x.view(x.size(0), self.num_points, 2)

