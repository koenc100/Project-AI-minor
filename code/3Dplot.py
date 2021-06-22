# 3D plot

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import math

# make 3D figure
fig = plt.figure(figsize=(10,10))

# create 3 axises
ax = fig.add_subplot(111, projection='3d')

X0 = np.linspace(0, 10, 100)
Y0 = np.linspace(0, 10, 100)

# make a grid, in order to plot every point
X, Y = np.meshgrid(X0, Y0)

Z =  np.sin(X) + np.sqrt(Y)

# make plot
actual_plot = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)

# add colorbar
fig.colorbar(actual_plot, shrink=0.5, aspect=5)

# Label X and Y

ax.set_xlabel('Oversampling ratio')
ax.set_ylabel('Class weight ratio')
ax.set_zlabel('Performance in percentage')

plt.show()
