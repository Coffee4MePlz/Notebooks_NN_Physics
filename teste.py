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
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 18, 10, stride=2, padding=1),  # b, 16, 76, 76
            nn.MaxPool2d(kernel_size=10, stride=2),
            nn.ReLU(True),
            nn.Conv2d(18, 14, 4, stride=2, padding=1),  # b, 8, 38, 38
            nn.MaxPool2d(kernel_size=4, stride=2),
            nn.ReLU(True),
            nn.Conv2d(14, 4, 3, stride=2, padding=1),  # b, 8, 38, 38
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(4, 4, 3, stride=2, padding=1),  # b, 8, 38, 38
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(True)
        )
        self.middlelayer = nn.Sequential(
            nn.Flatten(),  # Flatten the feature maps
            nn.Linear( 16, 32),  # Dense layer with 50 neurons
            nn.ReLU(True),
            nn.Linear(32, 20),  # Dense layer with 50 neurons
            nn.ReLU(True),
            nn.Linear(20, 32),  # Dense layer with 50 neurons
            nn.ReLU(True),
            nn.Linear(32, 5),  # Dense layer with 5 neurons
            nn.Tanh(),
        )
        self.rnn = nn.LSTM(5, 20, 10)
        #self.rnn2 = nn.LSTM(25, 5, 13)
        self.middlelayer2 = nn.Sequential(
            nn.Linear(20,32),  # Dense layer with 50 neurons # MIDDLELAYER
            nn.Tanh(),
            #nn.Linear(5,32),  # Dense layer with 50 neurons # MIDDLELAYER
            #nn.Tanh(),
            nn.Linear(32, 20),  # Dense layer with 50 neurons
            nn.ReLU(True),
            nn.Linear(20, 32),  # Dense layer with 50 neurons
            nn.ReLU(True),
            nn.Linear(32,3 * 8 * 8),  # Dense layer with 50 neurons
            nn.Unflatten(1, (3, 8,8))  # Reshape back to feature maps
        )
        self.decoder = nn.Sequential(
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(3, 16, 3, stride=2, padding=1, output_padding=1),  # b, 16, 76, 76
            nn.ReLU(True),
            #nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(16, 13, 8, stride=2, padding=1, output_padding=1),  # b, 3, 151, 151
            nn.ReLU(True),
            nn.ConvTranspose2d(13, 13, 4, stride=2, padding=1, output_padding=1),  # b, 3, 151, 151
            nn.ReLU(True),
            nn.ConvTranspose2d(13, 11, 3, stride=2, padding=1, output_padding=1),  # b, 3, 151, 151
            nn.ReLU(True),
            nn.ConvTranspose2d(11, 3, 2, stride=2, padding=1, output_padding=1),  # b, 3, 151, 151
            #nn.ReLU(True),
            #nn.ConvTranspose2d(3, 3, 2, stride=2, padding=1, output_padding=1),
            #nn.ReLU(True)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.middlelayer(x)
        x = x.unsqueeze(0)
        x, _ = self.rnn(x)
        #x, _ = self.rnn2(x)
        x = x.squeeze(0)
        x = self.middlelayer2(x)
        x = self.decoder(x)
        return x


# variaveis ctrl C
N=1000
frames_matrices = []
# Directory to save images
output_directory = 'pendulum_images'

for i in range(N):
    image_path = os.path.join(output_directory, f'frame_{i:04d}.png')
    image = Image.open(image_path)
    matrix = np.array(image)
    rgb_matrix = matrix[45:600,45:600,:3] # 376 to set image size 
    # Append the matrix for the current frame to the list
    frames_matrices.append(rgb_matrix)    

frames_matrices = np.array(frames_matrices)
frames_matrices_tensor = tc.tensor(frames_matrices, dtype=tc.float32)


# Hyperparameters
batch_size = 32
learning_rate = 1e-1
num_epochs = 200

# Example: frames_matrices = np.random.rand(10000, 387, 385, 3)

# Create DataLoader
dataset = CustomDataset(frames_matrices)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Initialize the autoencoder model and loss function
model = Autoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

LOSS = []
# Training loop
for epoch in range(num_epochs):
    for data in dataloader:
        img = data.permute(0, 3, 1, 2)  # Change the order of dimensions for Conv2d input
        recon_img = model(img)
        loss = criterion(recon_img, img)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        LOSS.append(loss.cpu().detach().numpy())

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    if ((epoch)%5) ==0:    
        plt.plot(np.log(LOSS))
        plt.yscale('log')
        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        plt.show()    
        plot_input(img)
        plot_output(recon_img)

# Save the trained model
tc.save(model.state_dict(), 'autoencoder.pth')
