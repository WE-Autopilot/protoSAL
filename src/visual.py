import torch
import h5py as hp
import numpy as np
from CNNModel import CNNModel
from dataset_generator import visualize_path

# Load model
model = CNNModel()
state_dict = torch.load("model.pth")
model.load_state_dict(state_dict)

# Load dataset
f = hp.File("paths.h5")
imgs = f["images"][:]
paths = f["paths"][:]

# Reshape images to match model input
imgs = imgs.reshape(100, 1, 32, 32)

# Convert to PyTorch tensors
imgs_tensor = torch.tensor(imgs, dtype=torch.float32)
paths_tensor = torch.tensor(paths, dtype=torch.float32)

# Get predictions
y = model(imgs_tensor)

# Compute mean absolute error
mean_abs_error = (paths_tensor - y).abs().mean()
print("Mean Absolute Error:", mean_abs_error.item())

# Process first prediction
y1 = y[0].reshape(4, 2)
for i in range(1, 4):
    y1[i] += y1[i - 1]

# Visualize path
visualize_path(torch.tensor(imgs[10, 0]), y1.detach())
