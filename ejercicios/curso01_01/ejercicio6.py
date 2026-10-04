import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

u = np.array([1, 2, 3])
v = np.array([1, 1, 1])
proj = np.array([2, 2, 2])          # proyección de u sobre v
perp = u - proj                       # componente perpendicular = [-1, 0, 1]

# Vector u (rojo)
ax.quiver(0, 0, 0, *u, color='red', arrow_length_ratio=0.12, linewidth=2)
# Vector v (azul)
ax.quiver(0, 0, 0, *v, color='blue', arrow_length_ratio=0.12, linewidth=2)
# Proyección [2,2,2] (verde)
ax.quiver(0, 0, 0, *proj, color='green', arrow_length_ratio=0.12, linewidth=2)
# Componente perpendicular: desde la punta de la proyección hasta la punta de u (naranja)
ax.quiver(*proj, *perp, color='orange', arrow_length_ratio=0.15, linewidth=2, linestyle='dashed')

# Marcadores de los puntos clave
ax.scatter(*u, color='red', s=30)
ax.scatter(*proj, color='green', s=30)

# Etiquetas
# ax.text(*u, '  u=[1,2,3]', color='red')
# ax.text(*proj, '  proj=[2,2,2]', color='green')
ax.text(u[0], u[1], u[2], '  u=[1,2,3]', color='red')
ax.text(proj[0], proj[1], proj[2], '  proj=[2,2,2]', color='green')
ax.text(v[0]*1.05, v[1]*1.05, v[2]*1.05, '  v=[1,1,1]', color='blue')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Proyección ortogonal de u sobre v')
ax.view_init(elev=25, azim=-60)
plt.grid(True)
plt.show()