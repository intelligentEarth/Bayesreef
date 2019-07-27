import os
import math
import time
import random
import csv
import numpy as np
from numpy import inf
from pyReefCore.model import Model
from cycler import cycler

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from matplotlib.cm import viridis
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D

from scipy.spatial import cKDTree
from scipy import stats
from scipy.stats import multivariate_normal

import plotly
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.graph_objs import *
plotly.offline.init_notebook_mode()
from plotly.offline.offline import _plot_html


def plotFunctions(X, X_title, Y, Y_title, Z):

        font = 10
        width = 1

        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
        ax.set_title('Joint Likelihood', fontsize=  font+2)#, y=1.02)

        ax1 = fig.add_subplot(211, projection = '3d')

                
        surf = ax1.plot_surface(X,Y,Z, cmap=cm.viridis, linewidth=0, antialiased=False)
        ax1.set_xlabel('%s' % Y_title)
        ax1.set_ylabel('%s' % X_title)
        ax1.set_zlabel('Log likelihood')
        ax1.set_zlim(Z.min(), Z.max())
        ax1.zaxis.set_major_locator(LinearLocator(10))
        # ax1.zaxis.set_major_formatter(FormatStrFormatter('%.05f'))
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.tight_layout(pad=2.0)
        plt.savefig('plot.png', bbox_inches='tight', dpi=300, transparent=False)
        plt.show()


        # surf = go.Surface(
        #     x=X, 
        #     y=Y, 
        #     z=Z,
        #     colorscale='Viridis'
        #     # mode='markers',
        #     # marker=dict(
        #     #     size=12, 
        #     #     color=Z, 
        #     #     colorscale='Viridis',
        #     #     opacity=0.8,
        #     #     showscale=True
        #     #     )
        #     )
        # data = [surf]
        # layout = go.Layout(
        #     title='%s' % self.description,
        #     autosize=True,
        #     width=1000,
        #     height=1000,
        #     scene=Scene(
        #         xaxis=XAxis(
        #             title='%s' % self.var2_title,
        #             nticks=10,
        #             gridcolor='rgb(255, 255, 255)',
        #             gridwidth=2,
        #             zerolinecolor='rgb(255, 255, 255)',
        #             zerolinewidth=2
        #             ),
        #         yaxis=YAxis(
        #             title='%s' % self.var1_title,
        #             nticks=10,
        #             gridcolor='rgb(255, 255, 255)',
        #             gridwidth=2,
        #             zerolinecolor='rgb(255, 255, 255)',
        #             zerolinewidth=2
        #             ),
        #         zaxis=ZAxis(
        #             title='Log likelihood'
        #             ),
        #         bgcolor="rgb(244, 244, 248)"
        #         )
        #     )

        # fig = go.Figure(data=data, layout=layout)
        # graph=plotly.offline.plot(fig, 
        #     auto_open=False, 
        #     output_type='file',
        #     filename= '%s/3d-likelihood-surface.html' % (self.filename),
        #     validate=False
        #     )
def main():

    v1 = 'Malthusian parameter'
    v1_title = r'$\varepsilon$'
    v1_min, v1_max = 0., 0.15

    v2 = 'Sub-/Super-diagonal'
    v2_title = r'$\alpha_s$'
    v2_min, v2_max = -0.15, 0.

    X = np.loadtxt('X.txt')
    Y = np.loadtxt('Y.txt')
    Z = np.loadtxt('Z.txt')

    plotFunctions(X, v1_title, Y, v2_title, Z)

if __name__ == "__main__": main()
