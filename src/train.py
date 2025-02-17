import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
import h5py
import numpy as np
from CNNModel import CNNModel

# define dataset class to handle loading and preprocessing data from an hdf5 file
class rcDataset(Dataset):
    def __init__(self, h5_file, transform=None):
        self.h5_file = h5_file
        self.transform = transform

        # load data from the hdf5 file
        with h5py.File(h5_file, 'r') as file:
            self.images = file['images'][:]  # assuming images are stored in 'images'
            self.labels = file['paths'][:]  # assuming labels are stored in 'labels'

    def __len__(self):
        # return the number of samples in the dataset
        return len(self.labels)

    def __getitem__(self, idx):
        # retrieve an image and its corresponding label by index
        image = self.images[idx]
        label = self.labels[idx]

        # apply transformations (for example: normalization, augmentation) if specified
        if self.transform:
            image = self.transform(image)

        return image, label

# data preparation
h5_file = 'paths.h5'  # add the actual path to hdf5 file (I don't have the dataset)

# define transformations for preprocessing the data
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,)),  
    # transforms.RandomHorizontalFlip(),  # Randomly flip images
    # transforms.RandomRotation(10),  # Rotate images slightly
    # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Small translations
])

# create dataset and split into training and validation sets
dataset = rcDataset(h5_file, transform=transform)
train_size = int(0.8 * len(dataset))  # 80% of the data for training
val_size = len(dataset) - train_size  # remaining 20% for validation
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# create data loaders for batching and shuffling data
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)  # training data loader
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)  # validation data loader

# check if a GPU is available and set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# define the training loop
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        model.train()  # set the model to training mode
        train_loss = 0.0

        # iterate over training data batches
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)  # move data to the selected device

            optimizer.zero_grad()  # clear gradients from the previous step
            outputs = model(images)  # forward pass: compute predictions
            loss = criterion(outputs, labels)  # compute loss
            loss.backward()  # backward pass: compute gradients
            optimizer.step()  # update model parameters

            train_loss += loss.item()  # accumulate training loss

        # validation phase
        model.eval()  # set the model to evaluation mode
        val_loss = 0.0

        with torch.no_grad():  # disable gradient computation for validation
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)  # foward pass: compute predictions
                    # print(outputs)
                    loss = criterion(outputs, labels)  # compute loss
                    val_loss += loss.item()  # accumulate validation loss

            # print training and validation statistics for the current epoch
        print(f"Epoch {epoch+1}/{epochs}, "
            f"Train Loss: {train_loss/len(train_loader):.4f}, "
            f"Val Loss: {val_loss/len(val_loader):.4f}, ")

# define the model, loss function, and optimizer
# Omar or Justin, replace `YourModel` with the actual model class name (maybe call is CNNmodel)
model = CNNModel()
criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # Adam optimizer with learning rate 0.001

# move the model to the selected device
model.to(device)

# start the training process
train_model(model, train_loader, val_loader, criterion, optimizer, epochs=20)

# save the trained model to a file
torch.save(model.state_dict(), 'model.pth')  # save the model weights