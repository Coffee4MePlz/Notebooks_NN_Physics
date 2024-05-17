
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Importar o módulo 3D

time = np.linspace(0, 2.1, 100)
latente = np.loadtxt("Notebooks_NN_Physics\lantentespace.txt")

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')  # Criar um subplot 3D

ax.plot(latente[:, 0], latente[:, 1], time[:], c='b', marker='o', linestyle='-', markersize=5, linewidth=1, alpha=1.0)
scatter = ax.scatter(latente[:, 0], latente[:, 1], time[:], c=time[:], cmap='jet', s=80, alpha=1.0, edgecolors='k')

# Adicionar colorbar
cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
cbar.set_label('Tempo (s)', rotation=270, labelpad=15)

# Definir título e labels
ax.set_title("Espaço Latente", fontsize=14)
ax.set_xlabel(r"$z_1$", fontsize=12)
ax.set_ylabel(r"$z_2$", fontsize=12)
ax.set_zlabel(r"$Tempo$", fontsize=12)

# Adicionar grid
ax.grid(True)

# Mostrar o plot
plt.show()
