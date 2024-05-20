import numpy as np
import matplotlib.pyplot as plt # para plots
import torch as tc
import pandas as pd
import torch.nn as nn
import os
import matplotlib.colors as mcolors
from PIL import Image



def create_colormap():
    colors = ['blue', 'red']
    cmap = mcolors.LinearSegmentedColormap.from_list('blue_to_red', colors)
    return cmap

def EDO_pendulo(t, y):
    theta, d_theta_dt = y
    dd_theta_dt = - (g / L) * np.sin(theta)
    return [d_theta_dt, dd_theta_dt]

def generate_data_set(omega = 1, theta = 0, time_parameters =[] ):
    t_0,t_end,dt = time_parameters    
    # Método de Euler para resolver a EDO
    t_values = np.arange(t_0, t_end, dt)

    theta_values = []
    omega_values = []
    for t in t_values:
        theta_values.append(theta)
        omega_values.append(omega)
        
        # Método de Euler
        dtheta_dt, domega_dt = EDO_pendulo(t, [theta, omega])
        theta += dtheta_dt * dt
        omega += domega_dt * dt

    # Converter o deslocamento angular para coordenadas x-y
    x_values = L * np.sin(theta_values)
    y_values = -L * np.cos(theta_values)
    return [x_values,y_values]

# Function to save dataset as images
def save_dataset_as_images(data_x, data_y, directory,trailsize = 10,cmap=0):
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    # Determine the number of frames based on the length of one of the data arrays
    num_frames = len(data_x)
    x_min, x_max = -1.3,1.3 #min(data_x), max(data_x)
    y_min, y_max = -2, 0.85, #min(data_y), max(data_y)
    # Loop through each frame in the dataset
    for i in range(num_frames):
        # Create a new figure
        plt.figure(figsize=(1,1))  # Adjust size as needed

        if trailsize == 0:
            trail_x = data_x[i:i+1]
            trail_y = data_y[i:i+1]
        else:
            if i>trailsize:
                trail_x = data_x[i-trailsize:i]
                trail_y = data_y[i-trailsize:i]
            else:
                trail_x = data_x[0:i]
                trail_y = data_y[0:i]

        # Plot the points
        plt.scatter(trail_x, trail_y, s=5, c=np.arange(len(trail_x)), cmap=cmap)  # Adjust size as needed

        #plt.scatter(data_x[0:i], data_y[0:i], s=5, c='black')  # Adjust size and color as needed
        
        # Set plot limits if necessary
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        
        # Turn off axis
        plt.axis('off')
        
        # Save the figure as an image
        plt.savefig(os.path.join(directory, f'frame_{i:04d}.png'), bbox_inches='tight', pad_inches=0)
        
        # Close the figure to release memory
        plt.close()


def generate_batch_set(N=100,phi=np.pi/20,omega_0=0.0, l=1):
    global g, L#, phi, omega_0
    g = 9.81  # m/s^2, aceleração devido à gravidade
    L = 1.0   # m, comprimento do pêndulo
    #phi = np.pi / 2  # Ângulo inicial em radianos
    #omega_0 = 0.0        # Velocidade angular inicial

    # Parâmetros de tempo
    #N=100 # Numero de pontos
    t_0 = 0.0            # Tempo inicial
    t_end = 3.0         # Tempo final
    dt = t_end/N            # Tamanho do passo de tempo
    time_parameters = [t_0,t_end,dt]

    # Condições iniciais
    theta = phi 
    omega = omega_0

    data = generate_data_set(omega, theta, time_parameters)

    data_x = data[0]  
    data_y = data[1]  

    # Directory to save images
    output_directory = 'pendulum_images'

    # Create a colormap from blue to red
    cmap = create_colormap()
    trailsize=0
    # Save dataset as images
    save_dataset_as_images(data_x, data_y, output_directory, trailsize,cmap)

    #print(f"Dataset saved as images in '{output_directory}' directory.")
    return