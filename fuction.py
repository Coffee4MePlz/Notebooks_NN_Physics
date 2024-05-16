from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms

import matplotlib.pyplot as plt
import numpy as np
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
import os


class SIN(nn.Module):
    def __init__(self): 
        super(SIN, self).__init__() 
    def forward(self, x):
        return tc.sin(x)

class CustomImageDataset(Dataset):
    def __init__(self, root_dir):
        """
        Args:
            root_dir (string): Diretório com todas as imagens.
        """
        self.root_dir = root_dir
        self.image_files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path)
        image = np.array(image)
        image = image / image.max()  # Normaliza a imagem para o intervalo [0, 1]
        image = 1-image
        image = image
        image = tc.tensor(image, dtype=tc.float32)
        image = image.unsqueeze(0)  # Adiciona uma dimensão de canal se for uma imagem em escala de cinza
        return image

class SimpleAutoencoder(nn.Module):
    def __init__(self, neck):
        super(SimpleAutoencoder, self).__init__()
        
        # Encoder CNN
        self.conv1 = nn.Conv2d(1, 5, kernel_size=4, stride=1, padding=1) 
        self.conv2 = nn.Conv2d(5, 10, kernel_size=4, stride=1, padding=1)  
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1) 
        
        self.encode3 = nn.Linear(10*40*40, 40*40)  # Encode
        self.encode2 = nn.Linear(40*40, 40)  # Encode
        self.encode1 = nn.Linear(40, neck)  # Encode
        self.decoder1 = nn.Linear(neck, 40)  # Decoder
        self.decoder2 = nn.Linear(40, 40*40)  # Decoder
        self.decoder3 = nn.Linear(40*40, 10*40*40)  # Decoder
        
        # Decoder CNN
        self.t_conv1 = nn.ConvTranspose2d(10, 5, kernel_size=4, stride=2, padding=1) 
        self.t_conv2 = nn.ConvTranspose2d(5, 1, kernel_size=4, stride=2, padding=1) 
        
        self.act = SIN()

    def forward(self, x):
        # Encoder CNN
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        # Encoder Autoencoder
        x = x.view(-1, 10*40*40)  # Flatten before fully connected layers
        x = self.act(self.encode3(x))
        x = self.act(self.encode2(x))
        y = self.act(self.encode1(x))

        # Decoder Autoencoder
        x = self.act(self.decoder1(y))
        x = self.act(self.decoder2(x))
        x = self.act(self.decoder3(x))

        # Reshape for decoder
        x = x.view(-1, 10, 40, 40)  # Reshape output to match the input of the first transposed conv layer

        # Decoder
        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))
        
        return x, y

        
class Trainer:
    def __init__(self, model, dataset, batch_size=15, lr=0.001, step_size=500, gamma=0.9,device='cpu',pltrue=True):
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.criterion = nn.MSELoss()
        self.optimizer = tc.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        self.losses = []
        self.dev = device
        self.pltrue = pltrue

    def train(self, epochs):
        self.model.to(self.dev)
        
        self.model.train()
        for epoch in range(epochs):
            for i, data in enumerate(self.dataloader, 0):
                data = data.to(self.dev)  # Move data to GPU
                self.optimizer.zero_grad()
                outputs, encoded = self.model(data)
                loss = self.criterion(outputs, data)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.losses.append(loss.item())
            #if epoch %(epochs/10) == 0:
            #    print(f'Epoch [{epoch+1}/{epochs}] , Loss: {loss.item():.4f}')
        print('Treinamento concluído')
        self.plot_losses(self.pltrue)

    def plot_losses(self,condi=True):
        if condi ==True:
            plt.plot(self.losses)
            plt.yscale("log")
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.title("Training Loss")
            plt.show()
