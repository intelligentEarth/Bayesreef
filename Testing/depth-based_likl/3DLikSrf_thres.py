# !/usr/bin/python
# BayesReef: a MCMC random walk method applied to pyReef-Core
# Authors: Jodie Pall and Danial Azam (2017)
# Adapted from: [Chandra_ICONIP2017] R. Chandra, L. Azizi, S. Cripps, 'Bayesian neural learning via Langevin dynamicsfor chaotic time series prediction', ICONIP 2017.
# (to be addeded on https://www.researchgate.net/profile/Rohitash_Chandra)

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


cmap=plt.cm.Set2
c = cycler('color', cmap(np.linspace(0,1,8)) )
plt.rcParams["axes.prop_cycle"] = c

class MCMC():
    def __init__(self, filename, xmlinput, simtime, samples, communities, sedsim, sedlim, flowsim, flowlim, vis,
        gt_depths, gt_vec_d, gt_prop_d, v1_min, v1_max, v2_min, v2_max, assemblage, description,
        v1, v1_title, v2, v2_title):

        self.font = 10
        self.width = 1
        self.d_sedprop = float(np.count_nonzero(gt_prop_d[:,communities]))/gt_prop_d.shape[0]
        
        self.filename = filename
        self.input = xmlinput
        self.simtime = simtime
        self.samples = samples
        self.communities = communities
        self.sedsim = sedsim
        self.sedlim = sedlim
        self.flowlim = flowlim
        self.flowsim = flowsim
        self.vis = vis   
        
        self.gt_depths = gt_depths
        self.gt_prop_d = gt_prop_d
        self.gt_vec_d = gt_vec_d
        
        self.true_sed = []
        self.true_flow = []
        self.true_m = 0.08
        self.true_ax = -0.01
        self.true_ay = -0.03
        
        self.assemblage = assemblage
        self.description = description
        self.var1= v1
        self.var2 = v2
        self.v1_min = v1_min
        self.v1_max = v1_max
        self.v2_min = v2_min
        self.v2_max = v2_max
        self.var1_title = v1_title
        self.var2_title = v2_title


    def runModel(self, reef, input_vector):
        reef.convertVector(self.communities, input_vector, self.sedsim, self.flowsim) #model.py
        self.true_sed, self.true_flow = reef.load_xml(self.input, self.sedsim, self.flowsim)
        # if self.vis[0] == True:
            # reef.core.initialSetting(size=(8,2.5), size2=(8,3.5)) # View initial parameters
        reef.run_to_time(self.simtime,showtime=100.)
        # if self.vis[1] == True:
        #     reef.plot.drawCore(lwidth = 3, colsed=self.colors, coltime = self.colors2, size=(9,8), font=8, dpi=300)
        sim_output_d = reef.plot.convertDepthStructure(self.communities, self.gt_depths) #modelPlot.py
        # predicted_core = reef.convert_core(self.communities, output_core, self.gt_depths) #model.py
        # return predicted_core
        return sim_output_d

    def convertCoreFormat(self, core):
        vec = np.zeros(core.shape[0])
        for n in range(len(vec)):
            if not all(v == 0 for v in core[n,:]):
                idx = np.argmax(core[n,:])# get index,
                vec[n] = idx+1 # +1 so that zero is preserved as 'none'
            else:
                vec[n] = 5.
        return vec

    def diffScore(self, sim_data,synth_data,intervals):
        maxprop = np.zeros((intervals,sim_data.shape[1]))
        for n in range(intervals):
            idx_synth = np.argmax(synth_data[n,:])
            idx_sim = np.argmax(sim_data[n,:])
            if ((sim_data[n,self.communities] != 1.) and (idx_synth == idx_sim)): #where sediment !=1 and max proportions are equal:
                maxprop[n,idx_synth] = 1
        diff = (1- float(np.count_nonzero(maxprop))/intervals)*100
        return diff

    def rmse(self, sim, obs):
        # where there is 1 in the sed column, count
        sed = np.count_nonzero(sim[:,self.communities])
        p_sedprop = (float(sed)/sim.shape[0])
        sedprop = np.absolute((self.d_sedprop - p_sedprop)*0.5)
        rmse =(np.sqrt(((sim - obs) ** 2).mean()))*0.5
        return rmse + sedprop

    def likelihoodWithProps(self, reef, gt_prop_d, input_v):
        sim_prop_d = self.runModel(reef, input_v)
        sim_prop_d = sim_prop_d.T
        intervals = sim_prop_d.shape[0]
        # # Uncomment if noisy synthetic data is required.
        # self.NoiseToData(intervals,sim_prop_t5)
        log_core = np.log(sim_prop_d+0.0001)
        log_core[log_core == -inf] = 0
        z = log_core * gt_prop_d
        likelihood = np.sum(z)
        diff = self.diffScore(sim_prop_d,gt_prop_d, intervals)
        rmse = self.rmse(sim_prop_d, gt_prop_d)
        # sim_vec_t = self.convertCoreFormat(sim_prop_t5)
        # sim_vec_d = self.convertCoreFormat(sim_prop_d.T)
        return [likelihood, diff, rmse, sim_prop_d]

    # def plotFunctions(X, X_title, Y, Y_title, Z):
    def plotFunctions(self, fname, v1, v2, likelihood):
        font = self.font
        width = self.width

        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
        ax.set_title('Joint Likelihood', fontsize=  font+2)#, y=1.02)

        ax1 = fig.add_subplot(211, projection = '3d')

        X = v2
        Y = v1
        # R = X/Y
        X, Y = np.meshgrid(X,Y)
        Z = likelihood
        np.savetxt('%s/X.txt' % fname, X)
        np.savetxt('%s/Y.txt' % fname, Y)
        np.savetxt('%s/Z.txt' % fname, Z)

        surf = ax1.plot_surface(X,Y,Z, cmap=cm.viridis, linewidth=0, antialiased=False)
        ax1.set_xlabel('\n%s' % self.var2_title, fontsize=font+2, linespacing=2.0)
        ax1.set_ylabel('\n%s' % self.var1_title, fontsize=font+2, linespacing=2.0)
        ax1.set_zlabel('\nLog likelihood', fontsize=font+2, linespacing=1.5)
        ax1.set_zlim(Z.min(), Z.max())
        ax1.zaxis.set_major_locator(LinearLocator(10))
        # ax1.zaxis.set_major_formatter(FormatStrFormatter('%.05f'))
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.tight_layout(pad=2.0)
        plt.savefig('%s/plot.png' % (fname), bbox_inches='tight', dpi=300, transparent=False)
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

    def save_params(self, v1, v2, likl, diff):    
        ### SAVE RECORD OF ACCEPTED PARAMETERS ###  
        if not os.path.isfile(('%s/data.csv' % (self.filename))):
            with file(('%s/data.csv' % (self.filename)),'wb') as outfile:
                writer = csv.writer(outfile, delimiter=',')
                titles = ["v1", "v2", "likl", "diff"]
                writer.writerow(titles)
                data = [v1, v2, likl, diff]
                writer.writerow(data)
        else:
            with file(('%s/data.csv' % (self.filename)),'ab') as outfile:
                writer = csv.writer(outfile, delimiter=',')
                data = [v1, v2, likl, diff]
                writer.writerow(data)
               
    def likelihood_surface(self):
    	samples = self.samples
        assemblage = self.assemblage
        dimension=samples*samples
        # Declare pyReef-Core and initialize
        reef = Model()
        
        # Define fixed parameters
        sed1=[0.0009, 0.0015, 0.0023]
        sed2=[0.0015, 0.0017, 0.0024]
        sed3=[0.0016, 0.0028, 0.0027]
        sed4=[0.0017, 0.0031, 0.0043]
        flow1=[0.055, 0.008 ,0.]
        flow2=[0.082, 0.051, 0.]
        flow3=[0.259, 0.172, 0.058] 
        flow4=[0.288, 0.185, 0.066] 
        m = self.true_m
        ax = self.true_ax
        ay = self.true_ay

        #Define min/max of parameter of interest
        v1_p1 = self.v1_min
        v1_p2 = self.v1_max
        v2_p1 = self.v2_min
        v2_p2 = self.v2_max 
        # Set number and value of iterates
        s_v1 = np.linspace(v1_p1, v1_p2, num=samples, endpoint=True)
        s_v2 = np.linspace(v2_p1, v2_p2, num=samples, endpoint=True)
        print 's_v1', s_v1
        print 's_v2', s_v2
        # Create storage for data
        pos_likl = np.full((s_v1.shape[0],s_v2.shape[0]), np.nan)
        pos_v1 = np.full(dimension, np.nan)
        pos_v2 = np.full(dimension, np.nan)
        pos_diff = np.full(dimension, np.nan)
        pos_rmse = np.full(dimension, np.nan)
        
        # S_star, cpts_star, ca_props_star = self.modelOutputParameters(self.gt_prop_t,self.gt_vec_t,self.gt_timelay)

        start = time.time()
        i = 0
        for a in range(s_v1.shape[0]):
            for b in np.arange(a,s_v2.shape[0]):
                print 'sample: ', i
                # print 'Variable 1: ', s_v1[a], 'Variable 2: ', s_v2[b]
                # # Update parameters
                # p_v1 = s_v1[a]
                # p_v2 = s_v2[b]

                # # USER DEFINED: Substitute generated variables into proposal vector 
                # m = p_v1
                # ay = p_v2
                
                # # Proposal to be passed to runModel
                # v_proposal = np.concatenate((sed1,sed2,sed3,sed4,flow1,flow2,flow3,flow4))
                # v_proposal = np.append(v_proposal,(ax,ay,m))
                # # [likelihood, diff, rmse, pred_data] = self.likelihoodWithDependence(reef, v_proposal, S_star, cpts_star, ca_props_star)
                # [likelihood, diff, rmse, pred_data] = self.likelihoodWithProps(reef, self.gt_prop_d, v_proposal)
                # print 'Likelihood:', likelihood, 'and difference score:', diff
                
                # pos_v1[i] = p_v1
                # pos_v2[i] = p_v2
                # pos_likl[a,b] = likelihood
                # pos_diff[i] = diff
                # pos_rmse[i] = rmse
                # self.save_params(pos_v1[i], pos_v2[i], pos_likl[a,b], pos_diff[i], pos_rmse[i])
                # # self.saveCore(reef,i)
                # i += 1
               
                if b >= a:
                    print '\n Variable 1: ', s_v1[a], 'Variable 2: ', s_v2[b]
                    # Update parameters
                    p_v1 = s_v1[a]
                    p_v2 = s_v2[b]

                    # Substitute generated variables into proposal vector 
                    flow2[assemblage-1] = p_v1
                    flow3[assemblage-1] = p_v2
                    
                    # Proposal to be passed to runModel
                    v_proposal = np.concatenate((sed1,sed2,sed3,sed4,flow1,flow2,flow3,flow4))
                    v_proposal = np.append(v_proposal,(ax,ay,m))
                    [likelihood, diff, rmse, pred_data] = self.likelihoodWithProps(reef, self.gt_prop_d, v_proposal)
                    print 'Likelihood:', likelihood, 'and difference score:', diff

                    pos_diff[i] = diff
                    pos_v1[i] = p_v1
                    pos_v2[i] = p_v2
                    pos_likl[a,b] = likelihood
                    self.save_params(pos_v1[i], pos_v2[i], pos_likl[a,b], pos_diff[i])
                i += 1
        pos_likl[pos_likl == -inf] = -528.696341847866#-178.965042307943
        pos_likl[np.isnan(pos_likl)] = -528.696341847866#-178.965042307943
        end = time.time()
        total_time = end - start
        self.plotFunctions(self.filename, s_v1, s_v2, pos_likl)
        print 'Counter:', i, '\nTime elapsed:', total_time, '\npos_likl.shape:', pos_likl.shape 
        
        return (pos_v1, pos_v2, pos_likl)


def main():
    random.seed(time.time())
    #    Set all input parameters    #

    # USER DEFINED: parameter names and plot titles.
    samples= 100
    titles = ['Shallow', 'Mod-deep', 'Deep']
    assemblage= 2
    sedlim = [0., 0.005]
    flowlim = [0.,0.3]

    sed1=[0.0009, 0.0015, 0.0023]
    sed2=[0.0015, 0.0017, 0.0024]
    sed3=[0.0016, 0.0028, 0.0027]
    sed4=[0.0017, 0.0031, 0.0043]
    flow1=[0.055, 0.008 ,0.]
    flow2=[0.082, 0.051, 0.]
    flow3=[0.259, 0.172, 0.058] 
    flow4=[0.288, 0.185, 0.066] 

    # v1 = 'Flow 1'
    # v1_title = r'$f_{flow}^1$'
    # v1_min, v1_max = flowlim[0], flow3[assemblage-1]

    # v2 = 'Flow 2'
    # v2_title = r'$f_{flow}^2$'
    # v2_min, v2_max = flowlim[0], flow3[assemblage-1]

    v1 = 'Flow 2'
    v1_title = r'$f_{flow}^2$'
    v1_min, v1_max = flow1[assemblage-1], flow4[assemblage-1]
    
    v2 = 'Flow 3'
    v2_title = r'$f_{flow}^3$'
    v2_min, v2_max = flow1[assemblage-1], flow4[assemblage-1]

    # v1 = 'Flow 3'
    # v1_title = r'$f_{flow}^3$'
    # v1_min, v1_max = flow2[assemblage-1], flowlim[1]

    # v2 = 'Flow 4'
    # v2_title = r'$f_{flow}^4$'
    # v2_min, v2_max = flow2[assemblage-1], flowlim[1]


    description = '3D likelihood surface, %s & %s' % (v1, v2)
    description2 = 'self.likelihoodWithProps'
    nCommunities = 3
    simtime = 8500
    xmlinput = 'input_synth.xml'
    datafile = 'data/synth_core_vec_d_08.txt'
    gt_depths, gt_vec_d = np.genfromtxt(datafile, usecols=(0, 1), unpack = True) 
    gt_prop_d = np.loadtxt('data/synth_core_prop_d_08.txt', usecols=(1,2,3,4))
    vis = [False, False]
    sedsim, flowsim = True, True
    sedlim = [0., 0.005]
    flowlim = [0.,0.3]

    run_nb = 0
    path_name = 'results-3d-thres'
    while os.path.exists('%s_%s' % (path_name, run_nb)):
        run_nb+=1
    if not os.path.exists('%s_%s' % (path_name, run_nb)):
        os.makedirs('%s_%s' % (path_name, run_nb))
    filename = ('%s_%s' % (path_name, run_nb))

    #    Save File of Run Description   #
    if not os.path.isfile(('%s/description.txt' % (filename))):
        with file(('%s/description.txt' % (filename)),'w') as outfile:
            outfile.write('Filename : {0}'.format(os.path.basename(__file__)))
            outfile.write('\nTest Description: ')
            outfile.write(description)
            outfile.write(description2)
            outfile.write('\nSamples: {0}'.format(samples))

    mcmc = MCMC(filename, xmlinput, simtime, samples, nCommunities, sedsim, sedlim, flowsim, flowlim, vis,
        gt_depths,gt_vec_d, gt_prop_d, v1_min, v1_max, v2_min, v2_max, assemblage, description,
        v1, v1_title, v2, v2_title)

    [pos_v1, pos_v2, pos_likl] = mcmc.likelihood_surface()

    print 'Successfully sampled'
    print 'Saved in %s' % filename
    print 'Finished producing Likelihood Surface'
if __name__ == "__main__": main()
