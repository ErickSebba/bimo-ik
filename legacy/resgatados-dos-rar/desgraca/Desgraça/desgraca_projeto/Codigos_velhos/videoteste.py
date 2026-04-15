import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
line, = ax.plot([], [], [], "o-", lw=2)

qs = gerar_trajetoria_simples(passos=60)
pontos = np.array([cinematica_direta(bimo, q) for q in qs])

ax.set_xlim(-400, 400)
ax.set_ylim(-400, 400)
ax.set_zlim(0, 400)

def update(i):
    p = pontos[:i+1]
    line.set_data(p[:, 0], p[:, 1])
    line.set_3d_properties(p[:, 2])
    return line,

ani = FuncAnimation(fig, update, frames=len(pontos), interval=50, blit=False)
ani.save("movimento.gif", writer=PillowWriter(fps=20))
plt.show()