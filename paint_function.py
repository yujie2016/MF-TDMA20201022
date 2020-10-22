import numpy as np
from matplotlib import pyplot as plt

x = np.linspace(0, 10 * np.pi, 1000, endpoint=True)
y = np.sin(x)
z = np.cos(x)

plt.plot(x, y)
# plt.plot(x, z)

plt.show()
