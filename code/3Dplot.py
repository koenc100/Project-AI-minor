# 3D plot

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# make figure
fig = plt.figure()

# create 3 axises
ax = fig.add_subplot(111, projection='3d')

#The X and Y input values
# This would be class-weight ratio and oversampling ratio
X0 = np.array([2, 3, 4, 2.5, 1.5, 1])
Y0 = np.array([1, 2, 3, 4, 3, 2])

# make a grid, in order to plot every point
X, Y = np.meshgrid(X0, Y0)

# This would be the outcome of NN with certain values
Z = (X + Y)

actual_plot = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
fig.colorbar(actual_plot, shrink=0.5, aspect=5)

ax.set_xlabel('Class weight ratio')
ax.set_ylabel('Oversampling ratio')
ax.set_zlabel('Performance in percentage')

plt.show()
