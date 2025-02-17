import torch
import h5py as hp
import numpy as np
from CNNModel import CNNModel
from dataset_generator import visualize_path

# Load model
model = CNNModel(num_points=4)
state_dict = torch.load("model.pth")
model.load_state_dict(state_dict)

# Load dataset
f = hp.File("paths.h5")
imgs = f["images"][:1000]
paths = f["paths"][:1000]

# Reshape images to match model input
imgs = np.expand_dims(imgs, 1)

# Convert to PyTorch tensors
imgs_tensor = torch.tensor(imgs, dtype=torch.float32)
paths_tensor = torch.tensor(paths, dtype=torch.float32)

# Get predictions
y = model(imgs_tensor)

# Compute mean absolute error
mean_abs_error = (paths_tensor - y).abs().mean()
print("Mean Absolute Error:", mean_abs_error.item())

while 1:
    i = int(input("id: "))

    # Process first prediction
    y1 = torch.cat((torch.tensor([0, 0]), y[i])).reshape(-1, 2)
    y1 = torch.cumsum(y1, axis=0)

    # Visualize path
    visualize_path(torch.tensor(imgs[i, 0]), y1.detach())
