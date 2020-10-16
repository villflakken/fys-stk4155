import os
import sys
import pretty_errors  # available with pip via `pip install pretty_errors`
import numpy as np
import math_functions as mf
import matplotlib.pyplot as plt
import sklearn.linear_model as skl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
sys.path.insert(3, '.')  # To keep my linter silenced

def plot_FrankeFunction(x_mg, y_mg, z_mg):
    """ 
    Visualize the Franke function in 3D.
    Takes x, y, and z values as meshgrids.
    """
    # Initialize plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x_mg, y_mg, z_mg, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    # Customizing the z axis.
    ax.set_zlim(-0.1, 1.4)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    pass


def plot_MSE_comps(mse_, ytilde_arrays, legend_list):
    """
    Plots an MSE curve between two or more.
    Both args should be iterables.
    """
    # Declarations
    num_arrays = len(ytilde_arrays)

    # Creating plotted object
    fig = plt.figure()
    ax = fig.gca(projection='2d')

    mse_



    # for i in np.arange(num_arrays):
    #     # 
    #     graph = ax.plot(, input_arrays[i])
    #     graph.xlabel("Polynomial complexity")
    #     graph.ylabel("Prediction Error")
    
    
    pass


if __name__ == "__main__":
    pass