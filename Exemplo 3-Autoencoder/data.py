import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
# Parâmetros do pêndulo
L = 1.0  # comprimento do pêndulo em metrose
g = 9.81  # aceleração devido à gravidade em m/s^2
theta0 = np.radians(20)  # ângulo inicial em radianos
omega = np.sqrt(g / L)  # Frequência angular

# Configuração inicial da plotagem
fig, ax = plt.subplots(figsize=(2, 2), dpi=100) 
ax.set_xlim((-0.5 * L, 0.5 * L))
ax.set_ylim((-1.1 * L, 0.05 * L))
ax.set_aspect('equal')
ax.axis('off')  # desativar os eixos
line, = ax.plot([], [], lw=2, color='black')
bob, = ax.plot([], [], 'o', color='black', markersize=10)
# Ajustar os limites do eixo


# Tempo total e número de frames
duration = 2.1  # segundos
frames = 100  # número de frames

# Função para obter coordenadas x, y do pêndulo
def get_coords(t):
    theta = theta0 * np.cos(omega * t)
    x = L * np.sin(theta)
    y = -L * np.cos(theta)
    return x, y
# Função de animação
# Coordenadas do corte (esquerda, superior, direita, inferior)
crop_rectangle = (20, 20, 180, 180)
for i in range(frames):
  t = i* duration / frames
  x, y = get_coords(t)
  line.set_data([0, x], [0, y])
  bob.set_data(x, y)
  fig.savefig(f'Notebooks_NN_Physics/data_pendulo/img_{i:03d}.png')
  # Carregar a imagem
  image = Image.open(f'Notebooks_NN_Physics/data_pendulo/img_{i:03d}.png')
  # Converter a imagem para escala de cinza
  image = image.convert('L')  # 'L' representa o modo escala de cinza
  image = image.crop(crop_rectangle)
  # Salvar a imagem em escala de cinza
  image.save(f'Notebooks_NN_Physics/data_pendulo/img_{i:03d}.png')
plt.close(fig)
