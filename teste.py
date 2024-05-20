import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt # para plots
import torch as tc
import pandas as pd 
import torch.nn as nn
import os
import matplotlib.colors as mcolors
from PIL import Image
from datasetgen import *
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import StepLR



device = tc.device("cuda" if tc.cuda.is_available() else "cpu")

def plot_input(img):
    image_np = img[0].numpy().transpose((1, 2, 0))
    #image_np = img[0,0].numpy().reshape(77,77)
    plt.imshow(image_np, cmap='gray') 
    '''
    # Plot each channel separately
    #fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    
    for i in range(1):
        axs[i].imshow(image_np[:, :, i], cmap='gray')  # Use cmap='gray' for grayscale images
        axs[i].set_title(f'Channel {i}')
    #'''
    plt.show()

def plot_output(recon_img):
    img2 = recon_img[0].detach().numpy()
    #image_np2 = img2[0].reshape(77,77)

    image_np2 = img2.transpose((1, 2, 0))
    plt.imshow(image_np2, cmap='gray')
    '''
    # Plot each channel separately
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    for i in range(1):
        axs[i].imshow(image_np2[:, :, i], cmap='gray')  # Use cmap='gray' for grayscale images
        axs[i].set_title(f'Channel {i}')
    '''
    plt.show()

# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, frames_matrices, transform=None):
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
            nn.Conv2d(1, 5, 9, stride=2, padding=1),  # b, 16, 76, 76
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Sigmoid(),
            #nn.ReLU(),
            nn.Conv2d(5, 13, 5, stride=2, padding=1),  # b, 8, 38, 38
            nn.Sigmoid(),#nn.ReLU(),
            #nn.Conv2d(30, 15, 2, stride=2, padding=1),  # b, 8, 38, 38
            #nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.Sigmoid(),#nn.ReLU(),
            nn.Conv2d(13, 6, 2, stride=2, padding=1),  # b, 8, 38, 38
            nn.MaxPool2d(kernel_size=1, stride=2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Flatten(),  # Flatten the feature maps
            #nn.Linear(16, 180),  # Dense layer with 50 neurons
            #nn.ReLU(),
            nn.Linear(54, 90),  # Dense layer with 50 neurons
            nn.ReLU(),
            nn.Linear(90, 45),  # Dense layer with 50 neurons
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(45, 16),  # Dense layer with 5 neurons
            nn.ReLU(),
            nn.Linear(16,15),
            nn.ReLU(),
        )
        self.middlelayer = nn.Sequential(
            nn.Linear(15,15),
            nn.ReLU(),#
            #nn.Sigmoid(),
        )
        #self.rnn = nn.LSTM(1, 5, 100)
        self.decoder = nn.Sequential(
            nn.Linear(15, 15),  # Dense layer with 5 neurons
            nn.ReLU(),
            nn.Linear(15, 16),  # Dense layer with 50 neurons
            nn.ReLU(),
            nn.Linear(16, 45),  # Dense layer with 50 neurons
            nn.ReLU(),
            nn.Linear(45, 90),  # Dense layer with 50 neurons
            nn.ReLU(),
            #nn.Linear(90, 180),  # Dense layer with 50 neurons
            #nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(90, 16),  # Dense layer with 50 neurons
            nn.ReLU(),
            nn.Flatten(),  # Flatten the feature maps
            nn.Linear(16, 6 *20* 20),  # Dense layer with 50 neurons
            nn.ReLU(),
            #nn.Unflatten(1, (15, 27, 27)),  # Reshape back to feature maps
            nn.Unflatten(1, (6, 20, 20)),  # Reshape back to feature maps
            nn.ReLU(),
            nn.ConvTranspose2d(6, 13, 4, stride=2, padding=1),  # b, 8, 38, 38
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Sigmoid(),#nn.ReLU(),
            #nn.ConvTranspose2d(20, 30, 2, stride=2, padding=1, output_padding=1),  # b, 30, 77, 77
            #nn.Sigmoid(),#nn.ReLU(),
            nn.Dropout(0.5),
            nn.ConvTranspose2d(13, 5, 4, stride=2, padding=1, output_padding=1),  # b, 77, 154, 154
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Sigmoid(),#nn.ReLU(),
            nn.ConvTranspose2d(5, 1, 3, stride=2, padding=1, output_padding=0),  # b, 3, 311, 311
            nn.Sigmoid(),#nn.ReLU(),
            #nn.Flatten(),
            #nn.Linear(7700,77*77*3),
            #nn.Unflatten(1, (input_shape[2],input_shape[1],input_shape[1]))
        )
        # Manually initialize weights and biases
        #nn.init.normal_(self.encoder[0].weight, mean=0.2, std=0.05)
        #nn.init.constant_(self.encoder[0].bias, 0)  # Initialize biases to zero
        #nn.init.normal_(self.decoder[0].weight, mean=0.2, std=0.05)
        #nn.init.constant_(self.decoder[0].bias, 0)  # Initialize biases to zero
        
    def forward(self, x):
        x = self.encoder(x)
        #for k in range(0,1):
        x = self.middlelayer(x)
        #x = x.unsqueeze(0)
        #x, _ = self.rnn(x)
        #x = x.squeeze(0)
        x = self.decoder(x)
        return x


# autoencoder without CNN
class DenseAutoencoder(nn.Module):
    def __init__(self,input_size, latent_dim):
        super(DenseAutoencoder, self).__init__()
        self.device = device
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 180),
            nn.ReLU(),
            nn.Linear(180, 190),  # Dense layer with 50 neurons
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(190, 145),  # Dense layer with 50 neurons
            nn.ReLU(),            
            nn.Linear(145, 160),  # Dense layer with 5 neurons
            nn.ReLU(),
            #nn.Dropout(0.4),
            nn.Linear(160,latent_dim),
            nn.ReLU(),
        )
        self.middlelayer = nn.Sequential(
            nn.Linear(latent_dim,latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 160),  # Dense layer with 50 neurons
            nn.ReLU(),
            nn.Linear(160, 145),  # Dense layer with 50 neurons
            nn.ReLU(),
            #nn.Dropout(0.4),            
            nn.Linear(145, 190),  # Dense layer with 50 neurons
            nn.ReLU(),
            nn.Linear(190, 180),  # Dense layer with 50 neurons
            nn.ReLU(),
            #nn.Dropout(0.5),            
            nn.Linear(180, input_size),  # Dense layer with 50 neurons
            nn.Sigmoid()
        )
        # Manually initialize weights and biases
        #nn.init.normal_(self.encoder[0].weight, mean=0.2, std=0.05)
        #nn.init.constant_(self.encoder[0].bias, 0)  # Initialize biases to zero
        #nn.init.normal_(self.decoder[0].weight, mean=0.2, std=0.05)
        #nn.init.constant_(self.decoder[0].bias, 0)  # Initialize biases to zero
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.middlelayer(x)
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
            # squared_error[:, 1, :, :] *= self.weight[1]  # Green channel            
            #squared_error[:, 2, :, :] *= self.weight[2]  # Blue channel
        
        # Compute mean squared error
        loss = tc.mean(squared_error)
        
        return loss
    
def treatdata(N):
    scaler = StandardScaler()

# Fit the scaler to the data and transform the data
    frames_matrices = []
    for i in range(N):
        image_path = os.path.join(output_directory, f'frame_{i:04d}.png')
        image = Image.open(image_path)
        matrix = np.array(image)
        rgb_matrix = matrix[:,:,1:2]
        reshaped_matrix = rgb_matrix.reshape(-1, 1)  # Reshape to [77*77, 4]

        if np.where(reshaped_matrix<255)[0].size >0:
            normalized_data = scaler.fit_transform(reshaped_matrix)
            rgb_matrix = normalized_data.reshape(77, 77, 1)
            #rgb_matrix = normalized_data.reshape(77* 77, 1)
            # Append the matrix for the current frame to the list
            frames_matrices.append(rgb_matrix)    

    frames_matrices = np.array(frames_matrices)
    frames_matrices_tensor = tc.tensor(frames_matrices, dtype=tc.float32)

    return frames_matrices


# CODE

# variaveis ctrl C
N= 101#101 #801
frames_matrices = []
# Directory to save images
output_directory = 'pendulum_images'

#generate_batch_set(N,np.pi/5,0.0, l=1)
frames_matrices = treatdata(N)

# Hyperparameters
batch_size = 1 #50
learning_rate = 1e-3 # 1e-4
num_epochs = 20000

# Example: frames_matrices = np.random.rand(10000, 77, 77, 3)

# Create DataLoader
dataset = CustomDataset(frames_matrices)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize the autoencoder model and loss function
model = Autoencoder()
input_size = 77*77
latent_dim = 3
#model = DenseAutoencoder(input_size, latent_dim)

# Define custom weights for red and blue channels
custom_weights = tc.tensor([1., 0., 0.], dtype=tc.float32)

# Create an instance of the weighted MSE loss function
criterion = nn.MSELoss()
weighted_mse_loss = WeightedMSELoss(weight=custom_weights)

#criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  
scheduler = StepLR(optimizer, step_size=5000, gamma=0.8  )
print(f'device is : {device}')

LOSS = []
# Training loop
for epoch in range(num_epochs):
    i=0
    '''
    angle = np.pi/10*(1+np.random.randn())
    omega_0 = 0.2*np.random.randn()
    l = abs(4*np.random.randn())
    generate_batch_set(N,angle,omega_0,l)
    frames_matrices = treatdata(N)
    dataset = CustomDataset(frames_matrices)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    #'''
    for data in dataloader:
        #img = data.permute(0, 2, 1)  # Change the order of dimensions for Conv2d input
        img = data.permute(0, 3, 1, 2)  # Change the order of dimensions for Conv2d input
        #if i <1:#batch_size:
        recon_img = model(img)
            #loss = 0
        loss = criterion(recon_img, img) - 15*1e-3*tc.norm(recon_img)

            # Calculate loss
            #loss = weighted_mse_loss(recon_img, img)
        #    i+=1
        #else: 
        #    recon_img = model(last_img)
            #    loss = weighted_mse_loss(recon_img, img)
            #recon_img = model(img)
            #loss = weighted_mse_loss(recon_img, img)
        #    loss = criterion(recon_img, img)

        #last_img = img 
    
        optimizer.zero_grad()
        scheduler.step()
        loss.backward()
        optimizer.step()
        LOSS.append(loss.cpu().detach().numpy())

        #last_img = img  
        '''
        recon_img = model(img)
        loss = weighted_mse_loss(recon_img, img)
        last_img = img  
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        '''

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.8f}  Learning Rate: {scheduler.get_last_lr()}')
    if ((epoch+1)%((num_epochs/10))) ==0:    
        #plt.plot(np.log(LOSS))
        #plt.yscale('log')
        plt.plot(LOSS)
        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        plt.show()    
        plot_input(img)
        #plt.plot(recon_img)
        plot_output(recon_img)

# Save the trained model
tc.save(model.state_dict(), 'autoencoder.pth')
