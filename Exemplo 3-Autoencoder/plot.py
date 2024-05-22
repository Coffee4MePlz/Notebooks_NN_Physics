
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Importar o módulo 3D

time = np.linspace(0, 2.1, 100)
latente = np.loadtxt("Notebooks_NN_Physics\Exemplo 3-Autoencoder\lantentespace.txt")


fig = plt.figure(figsize=(15, 7))

# Subplot para o gráfico 2D
ax2 = fig.add_subplot(122)

scatter = ax2.scatter(latente[:, 0], latente[:, 1], c=time, cmap='jet', s=80, alpha=1.0, edgecolors='k')

# Adicionar colorbar
cbar = fig.colorbar(scatter, ax=ax2)
cbar.set_label('Tempo (s)', rotation=270, labelpad=15)

# Definir título e labels
ax2.set_title("Espaço Latente 2D", fontsize=14)
ax2.set_xlabel(r"$z_1$", fontsize=12)
ax2.set_ylabel(r"$z_2$", fontsize=12)

# Adicionar grid
ax2.grid(True)

# Subplot para o gráfico 3D
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot(latente[:, 0], latente[:, 1], time[:], c='b', marker='o', linestyle='-', markersize=5, linewidth=1, alpha=1.0)
scatter = ax1.scatter(latente[:, 0], latente[:, 1], time[:], c=time[:], cmap='jet', s=80, alpha=1.0, edgecolors='k')

# Adicionar colorbar ao gráfico 3D
##cbar = fig.colorbar(scatter, ax=ax2, pad=0.1)
#cbar.set_label('Tempo (s)', rotation=270, labelpad=15)

# Definir título e labels do gráfico 3D
ax1.set_title("Espaço Latente 3D", fontsize=14)
ax1.set_xlabel(r"$z_1$", fontsize=12)
ax1.set_ylabel(r"$z_2$", fontsize=12)
ax1.set_zlabel(r"$Tempo$", fontsize=12)
# Definir os ticks dos eixos
ax1.set_xticks([-1,0,1])  # 5 valores no eixo X
ax1.set_yticks([-1,0,1])   # 5 valores no eixo Y



# Adicionar grid ao gráfico 3D
ax1.grid(True)

# Ajustar layout para evitar sobreposição
plt.tight_layout()

# Mostrar o plot
plt.show()
