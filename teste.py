import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt # para plots
import torch as tc
import pandas as pd 
import torch.nn as nn
import os
import matplotlib.colors as mcolors
from PIL import Image

device = tc.device("cuda" if tc.cuda.is_available() else "cpu")

def plot_input(img):
    image_np = img[0].numpy().transpose((1, 2, 0))
    
    # Plot each channel separately
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    for i in range(3):
        axs[i].imshow(image_np[:, :, i], cmap='gray')  # Use cmap='gray' for grayscale images
        axs[i].set_title(f'Channel {i}')

    plt.show()

def plot_output(recon_img):
    img2 = recon_img[0].detach().numpy()
    image_np2 = img2.transpose((1, 2, 0))
    # Plot each channel separately
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    for i in range(3):
        axs[i].imshow(image_np2[:, :, i], cmap='gray')  # Use cmap='gray' for grayscale images
        axs[i].set_title(f'Channel {i}')

    plt.show()

# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, frames_matrices):
        self.frames_matrices = frames_matrices

    def __len__(self):
        return len(self.frames_matrices)

    def __getitem__(self, idx):
        image = self.frames_matrices[idx]
        # Assuming image is already normalized between 0 and 1
        image = tc.tensor(image, dtype=tc.float32)
        return image

# Define the autoencoder model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.device = device
        self.encoder = nn.Sequential(
            #nn.Linear(input_shape, input_shape),
            #nn.ReLU(),
            nn.Conv2d(3, 77, 8, stride=2, padding=1),  # b, 16, 76, 76
            nn.MaxPool2d(kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(77, 30, 5, stride=2, padding=1),  # b, 8, 38, 38
            nn.ReLU(),
            nn.Conv2d(30, 15, 5, stride=2, padding=1),  # b, 8, 38, 38
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Flatten(),  # Flatten the feature maps
            nn.Linear(15, 180),  # Dense layer with 50 neurons
            nn.ReLU(),
            nn.Linear(180, 90),  # Dense layer with 50 neurons
            nn.ReLU(),
            nn.Linear(90, 45),  # Dense layer with 50 neurons
            nn.ReLU(),
            nn.Linear(45, 18),  # Dense layer with 5 neurons
            nn.Tanh(),
            nn.Linear(18,1),
            nn.Tanh()
        )
        self.middlelayer = nn.Sequential(
            nn.Linear(1,1),
            nn.Sigmoid()
        )
        self.rnn = nn.LSTM(1, 5, 100)
        self.decoder = nn.Sequential(
            nn.Linear(5, 5),  # Dense layer with 5 neurons
            nn.Tanh(),
            nn.Linear(5, 18),  # Dense layer with 5 neurons
            nn.Tanh(),
            nn.Linear(18, 45),  # Dense layer with 50 neurons
            nn.ReLU(),
            nn.Linear(45, 90),  # Dense layer with 50 neurons
            nn.ReLU(),
            nn.Linear(90, 180),  # Dense layer with 50 neurons
            nn.ReLU(),
            nn.Linear(180, 16),  # Dense layer with 50 neurons
            nn.ReLU(),
            nn.Flatten(),  # Flatten the feature maps
            nn.Linear(16, 15 * 7 * 7),  # Dense layer with 50 neurons
            nn.ReLU(),
            #nn.Unflatten(1, (15, 27, 27)),  # Reshape back to feature maps
            nn.Unflatten(1, (15, 7, 7)),  # Reshape back to feature maps
            nn.ReLU(),
            nn.ConvTranspose2d(15, 30, 5, stride=2, padding=1, output_padding=1),  # b, 30, 77, 77
            nn.ReLU(),
            nn.ConvTranspose2d(30, 77, 8, stride=2, padding=1, output_padding=1),  # b, 77, 154, 154
            #nn.MaxPool2d(kernel_size=4, stride=3),
            nn.ReLU(),
            nn.ConvTranspose2d(77, 3, 3, stride=2, padding=1, output_padding=1),  # b, 3, 311, 311
            nn.ReLU(),
            #nn.Flatten(),
            #nn.Linear(7700,77*77*3),
            #nn.Unflatten(1, (input_shape[2],input_shape[1],input_shape[1]))
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.middlelayer(x)
        x = x.unsqueeze(0)
        x, _ = self.rnn(x)
        x = x.squeeze(0)
        x = self.decoder(x)
        return x

class WeightedMSELoss(nn.Module):
    def __init__(self, weight=None):
        super(WeightedMSELoss, self).__init__()
        self.weight = weight  # Weight tensor for each channel

    def forward(self, input, target):
        # Compute squared error
        squared_error = (input - target) ** 2
        
        # Apply weights to specific channels (e.g., red and blue)
        if self.weight is not None:
            squared_error[:, 0, :, :] *= self.weight[0]  # Red channel
            squared_error[:, 1, :, :] *= self.weight[1]  # Green channel            
            squared_error[:, 2, :, :] *= self.weight[2]  # Blue channel
        
        # Compute mean squared error
        loss = tc.mean(squared_error)
        
        return loss


# CODE


# variaveis ctrl C
N=1000
frames_matrices = []
# Directory to save images
output_directory = 'pendulum_images'

for i in range(N):
    image_path = os.path.join(output_directory, f'frame_{i:04d}.png')
    image = Image.open(image_path)
    matrix = np.array(image)
    rgb_matrix = matrix[2:76,2:76,:3] # 77 to set image size 
    # Append the matrix for the current frame to the list
    frames_matrices.append(rgb_matrix)    

frames_matrices = np.array(frames_matrices)
frames_matrices_tensor = tc.tensor(frames_matrices, dtype=tc.float32)


# Hyperparameters
batch_size = 32
learning_rate = 1e-3
num_epochs = 400

# Example: frames_matrices = np.random.rand(10000, 387, 385, 3)

# Create DataLoader
dataset = CustomDataset(frames_matrices)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Initialize the autoencoder model and loss function
model = Autoencoder()
# Define custom weights for red and blue channels
custom_weights = tc.tensor([2.0, 1, 2], dtype=tc.float32)

# Create an instance of the weighted MSE loss function
weighted_mse_loss = WeightedMSELoss(weight=custom_weights)

#criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
print(f'device is : {device}')

LOSS = []
# Training loop
for epoch in range(num_epochs):
    i=0
    for data in dataloader:
        img = data.permute(0, 3, 1, 2)  # Change the order of dimensions for Conv2d input
        if i <batch_size:
            recon_img = model(img)
            #loss = criterion(recon_img, img)
            # Calculate loss
            loss = weighted_mse_loss(recon_img, img)
            i+=1
        else: 
            recon_img = model(last_img)
            loss = weighted_mse_loss(recon_img, img)
        '''
        recon_img = model(img)
        loss = weighted_mse_loss(recon_img, img)
        '''
        last_img = img
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        LOSS.append(loss.cpu().detach().numpy())

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    if ((epoch)%399) ==0:    
        plt.plot(np.log(LOSS))
        plt.yscale('log')
        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        plt.show()    
        plot_input(img)
        plot_output(recon_img)

# Save the trained model
tc.save(model.state_dict(), 'autoencoder.pth')
