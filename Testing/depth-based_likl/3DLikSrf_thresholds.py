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
from numpy import inf
import numpy as np
from pyReefCore.model import Model
from cycler import cycler

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
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
    def __init__(self, simtime, samples, communities, core_data, core_depths,data_vec, timestep,filename, 
        xmlinput, sedsim, sedlim, flowsim, flowlim, assemblage, vis, description,
        v1, v1_title, v2, v2_title):
        self.filename = filename
        self.input = xmlinput
        self.communities = communities
        self.samples = samples       
        self.core_data = core_data
        self.core_depths = core_depths
        self.data_vec = data_vec
        self.timestep = timestep
        self.vis = vis
        self.sedsim = sedsim
        self.flowsim = flowsim
        self.sedlim = sedlim
        self.flowlim = flowlim
        self.simtime = simtime
        self.assemblage = assemblage
        self.font = 10
        self.width = 1
        self.d_sedprop = float(np.count_nonzero(core_data[:,self.communities]))/core_data.shape[0]
        self.initial_sed = []
        self.initial_flow = []
        self.true_m = 0.086
        self.true_ax = -0.01
        self.true_ay = -0.03
        self.description = description
        self.var1= v1
        self.var2= v2
        self.var1_title = v1_title
        self.var2_title = v2_title

    def run_Model(self, reef, input_vector):
        reef.convert_vector(self.communities, input_vector, self.sedsim, self.flowsim) #model.py
        self.initial_sed, self.initial_flow = reef.load_xml(self.input, self.sedsim, self.flowsim)
        if self.vis[0] == True:
            reef.core.initialSetting(size=(8,2.5), size2=(8,3.5)) # View initial parameters
        reef.run_to_time(self.simtime,showtime=100.)
        if self.vis[1] == True:
            from matplotlib.cm import terrain, plasma
            nbcolors = len(reef.core.coralH)+10
            colors = terrain(np.linspace(0, 1.8, nbcolors))
            nbcolors = len(reef.core.layTime)+3
            colors2 = plasma(np.linspace(0, 1, nbcolors))
            reef.plot.drawCore(lwidth = 3, colsed=colors, coltime = colors2, size=(9,8), font=8, dpi=300)
        output_core = reef.plot.core_timetodepth(self.communities, self.core_depths) #modelPlot.py
        # predicted_core = reef.convert_core(self.communities, output_core, self.core_depths) #model.py
        # return predicted_core
        return output_core

    def plotFunctions(self, fname, v1, v2, likelihood):

        font = self.font
        width = self.width
        X = v1
        Y = v2
        # R = X/Y
        X, Y = np.meshgrid(X, Y)
        Z = likelihood

        surf = go.Surface(
            x=X, 
            y=Y, 
            z=Z,
            colorscale='Viridis'
            # mode='markers',
            # marker=dict(
            #     size=12, 
            #     color=Z, 
            #     colorscale='Viridis',
            #     opacity=0.8,
            #     showscale=True
            #     )
            )
        data = [surf]
        layout = go.Layout(
            title='%s' % self.description,
            autosize=True,
            width=1000,
            height=1000,
            scene=Scene(
                xaxis=XAxis(
                    title='%s' % self.var1_title,
                    nticks=10,
                    gridcolor='rgb(255, 255, 255)',
                    gridwidth=2,
                    zerolinecolor='rgb(255, 255, 255)',
                    zerolinewidth=2
                    ),
                yaxis=YAxis(
                    title='%s' % self.var2_title,
                    nticks=10,
                    gridcolor='rgb(255, 255, 255)',
                    gridwidth=2,
                    zerolinecolor='rgb(255, 255, 255)',
                    zerolinewidth=2
                    ),
                zaxis=ZAxis(
                    title='Likelihood'
                    ),
                bgcolor="rgb(244, 244, 248)"
                )
            )

        fig = go.Figure(data=data, layout=layout)
        graph=plotly.offline.plot(fig, 
            auto_open=False, 
            output_type='file',
            filename= '%s/3d-likelihood-surface.html' % (self.filename),
            validate=False
            )

        # current_cmap = cm.coolwarm
        # current_cmap.set_bad(color='red')
        # print 'X shape ', X.shape, 'Y shape ', Y.shape, 'Z shape ', Z.shape

        # fig = plt.figure(figsize=(15,15))
        # ax = fig.add_subplot(111)
        # ax.spines['top'].set_color('none')
        # ax.spines['bottom'].set_color('none')
        # ax.spines['left'].set_color('none')
        # ax.spines['right'].set_color('none')
        # ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
        # ax.set_title('Likelihood', fontsize=  font+2)#, y=1.02)
        
        # ax1 = fig.add_subplot(211, projection = '3d')
        # ax1.set_facecolor('#f2f2f3')

        

        # surf = ax1.plot_surface(X,Y,Z, cmap = current_cmap, linewidth= 0, antialiased = False)
        # # ax1.set_zlim(Z.min(), Z.max())
        # ax1.zaxis.set_major_locator(LinearLocator(10))
        # ax1.zaxis.set_major_formatter(FormatStrFormatter('%.05f'))
        # ax1.set_xlabel(r"$\dot{\Theta}$ 1")
        # ax1.set_ylabel(r"$\dot{\Theta}$ 2")
        # ax1.set_zlabel('Likelihood')


        # Add a color bar which maps values to colors.

        # ax2 = fig.add_subplot(212)
        # surf2 = ax2.plot_surface(X,Y,Z2, cmap = cm.coolwarm, linewidth= 0, antialiased = False)
        # ax2.set_zlim(Z.min(), Z.max())
        # ax2.zaxis.set_major_locator(LinearLocator(10))
        # ax2.zaxis.set_major_formatter(FormatStrFormatter('%.05f'))

        # fig.colorbar(surf, shrink=0.5, aspect=5)
        # plt.savefig('%s/plot.png'% (fname), bbox_inches='tight', dpi=300, transparent=False)
        # plt.show()

    def save_params(self, v1, v2, likl, diff):    
    	### SAVE RECORD OF ACCEPTED PARAMETERS ###  
        if not os.path.isfile(('%s/data.csv' % (self.filename))):
            with file(('%s/data.csv' % (self.filename)),'wb') as outfile:
                writer = csv.writer(outfile, delimiter=',')
                data = [v1, v2, likl, diff]
                writer.writerow(data)
        else:
            with file(('%s/data.csv' % (self.filename)),'ab') as outfile:
                writer = csv.writer(outfile, delimiter=',')
                data = [v1, v2, likl, diff]
                writer.writerow(data)

    def diff_score(self, pred_core_w_noise,core_data, intervals):
        maxprop = np.zeros((intervals,self.communities+1))
        for n in range(intervals):
            idx_data = np.argmax(core_data[n,:])
            idx_model = np.argmax(pred_core_w_noise[n,:])
            if ((pred_core_w_noise[n,self.communities] != 1.) and (idx_data == idx_model)): #where sediment !=1 and max proportions are equal:
                maxprop[n,idx_data] = 1
        same= np.count_nonzero(maxprop)
        same = float(same)/intervals
        diff = 1-same
        print 'Difference:', diff
        return diff*100

    def rmse(self, sim, obs):
        # where there is 1 in the sed column, count
        sed = np.count_nonzero(sim[:,self.communities])
        p_sedprop = (float(sed)/sim.shape[0])
        sedprop = np.absolute((self.d_sedprop - p_sedprop)*0.5)
        rmse =(np.sqrt(((sim - obs) ** 2).mean()))*0.5
        
        return rmse + sedprop

    def likelihood_func(self, reef, core_data, input_v):
        pred_core = self.run_Model(reef, input_v)
        pred_core = pred_core.T
        pred_core_w_noise = np.zeros((pred_core.shape[0], pred_core.shape[1]))
        intervals = pred_core.shape[0]
        for n in range(intervals):
           pred_core_w_noise[n,:] = np.random.multinomial(1000,pred_core[n],size=1)
        pred_core_w_noise = pred_core_w_noise/1000
        z = np.zeros((intervals,self.communities+1))  
        z = pred_core_w_noise * core_data
        loss = np.log(z)
        loss[loss == -inf] = 0
        loss = np.sum(loss)
        diff = self.diff_score(pred_core_w_noise,core_data, intervals)

        return [loss, pred_core_w_noise, diff]

    def likelihood_func_old(self, reef, core_data, input_v):
        pred_core = self.run_Model(reef, input_v)
        pred_core = pred_core.T
        intervals = pred_core.shape[0]
        z = np.zeros((intervals,self.communities+1))    
        for n in range(intervals):
            idx_data = np.argmax(core_data[n,:])
            idx_model = np.argmax(pred_core[n,:])
            if ((pred_core[n,self.communities] != 1.) and (idx_data == idx_model)): #where sediment !=1 and max proportions are equal:
                z[n,idx_data] = 1
        diff = self.diff_score(z,intervals)
        # rmse = self.rmse(pred_core, core_data)
        
        z = z + 0.1
        z = z/(1+(1+self.communities)*0.1)
        loss = np.log(z)
        # print 'sum of loss:', np.sum(loss)        
        return [np.sum(loss), pred_core, diff]
               
    def likelihood_surface(self):
    	samples = self.samples
    	assemblage = self.assemblage

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
        m = 0.1
        ax = -0.01
        ay = -0.03

        dimension = int(math.sqrt(samples))
        print 'dimension:', dimension
        # firstpoint = flow_lower#sed1[assemblage-1]
        # lastpoint =  flow_upper #sed4[assemblage-1]

        # USER DEFINED: max and min range
        v1_p1 = flow1[assemblage-1]
        v1_p2 = flow4[assemblage-1]
        v2_p1 = flow1[assemblage-1]
        v2_p2 = flow4[assemblage-1]

        # s_v1 = np.linspace(firstpoint, lastpoint, num=dimension, endpoint=False)
        # s_v2 = np.linspace(firstpoint, lastpoint, num=dimension, endpoint=False)
        s_v1 = np.linspace(v1_p1, v1_p2, num=dimension, endpoint=True)
        s_v2 = np.linspace(v2_p1, v2_p2, num=dimension, endpoint=True)
        print 's_v1', s_v1
        print 's_v2', s_v2


        # Create storage for data
        dimx = s_v1.shape[0]
        dimy = s_v2.shape[0]
        pos_likl = np.zeros((dimx,dimy))
        pos_v1 = np.zeros(samples) 
        pos_v2 = np.zeros(samples)
        pos_diff = np.zeros(samples)

        start = time.time()
        i = 0
        for a in range(len(s_v1)):
            for b in range(len(s_v2)):
                # print 'sample: ', i
                # print 'Variable 1: ', s_v1[a], 'Variable 2: ', s_v2[b]
                
                # # Update parameters
                # p_v1 = s_v1[a]
                # p_v2 = s_v2[b]

                # # Substitute generated variables into proposal vector 
                # m = p_v1
                # # ax = p_v1
                # ay = p_v2
                
                # # Proposal to be passed to runModel
                # v_proposal = np.concatenate((sed1,sed2,sed3,sed4,flow1,flow2,flow3,flow4))
                # v_proposal = np.append(v_proposal,(ax,ay,m))

                # [likelihood, pred_data, diff] = self.likelihood_func(reef, self.core_data, v_proposal)
                # print 'Likelihood:', likelihood, 'and difference score:', diff

                # pos_diff[i] = diff
                # pos_v1[i] = p_v1
                # pos_v2[i] = p_v2
                # pos_likl[a,b] = likelihood
                # self.save_params(pos_v1[i], pos_v2[i], pos_likl[a,b], pos_diff[i])
                # i += 1
                
                if b < a:
                    #set pos_sed 2, pos_v2, pos_likl and pos_diff to NaN because can't run.
                    pos_diff[i] = np.nan
                    pos_v1[i] = np.nan
                    pos_v2[i] = np.nan
                    pos_likl[a,b] = np.nan
                    self.save_params(pos_v1[i], pos_v2[i], pos_likl[a,b], pos_diff[i])

                else:
                    print '\n Variable 1: ', s_v1[a], 'Variable 2: ', s_v2[b]

                    # Update parameters
                    p_v1 = s_v1[a]
                    p_v2 = s_v2[b]


                    # Substitute generated variables into proposal vector 
                    flow2[assemblage-1] = p_v1
                    flow3[assemblage-1] = p_v2
                    # print 'sed2', sed2
                    # print 'sed3', sed3
                    
                    # Proposal to be passed to runModel
                    v_proposal = np.concatenate((sed1,sed2,sed3,sed4,flow1,flow2,flow3,flow4))
                    v_proposal = np.append(v_proposal,(ax,ay,m))

                    [likelihood, pred_data, diff] = self.likelihood_func(reef, self.core_data, v_proposal)
                    print 'Likelihood:', likelihood, 'and difference score:', diff

                    pos_diff[i] = diff
                    pos_v1[i] = p_v1
                    pos_v2[i] = p_v2
                    pos_likl[a,b] = likelihood
                    self.save_params(pos_v1[i], pos_v2[i], pos_likl[a,b], pos_diff[i])
                i += 1

        self.plotFunctions(self.filename, s_v1, s_v2, pos_likl)
        end = time.time()
        total_time = end - start
        print 'Counter:', i, '\nTime elapsed:', total_time, '\npos_likl.shape:', pos_likl.shape 
        
        return (pos_v1, pos_v2, pos_likl)

#####################################################################

def main():
    random.seed(time.time())

    #    Set all input parameters    #
    samples= 22500 #input('Enter number of samples: ')
    assemblage= 3
    v1 = 'Flow 2'
    v2 = 'Flow 3'
    v1_title = '2'
    v2_title = '3'
    super_title = 'Flow velocity threshold'
    # super_title = 'Sediment input threshold'
    description = '%s likelihood surface: %s assemblage (\n %s & %s)' % (super_title, assemblage, v1, v2)


    nCommunities = 3
    simtime = 8500
    timestep = np.arange(0,simtime+1,50)
    xmlinput = 'input_synth_new.xml'
    datafile = 'data/synth_core_vec_1.txt'
    core_depths, data_vec = np.genfromtxt(datafile, usecols=(0, 1), unpack = True) 
    core_data = np.loadtxt('data/synth_core_prop_1.txt', usecols=(1,2,3,4))
    vis = [False, False] # first for initialisation, second for cores
    sedsim, flowsim = True, True
    sedlim = [0., 0.005]
    flowlim = [0.,0.3]
    run_nb = 0
    while os.path.exists('results_3dsurf_%s' % (run_nb)):
        run_nb+=1
    if not os.path.exists('results_3dsurf_%s' % (run_nb)):
        os.makedirs('results_3dsurf_%s' % (run_nb))
    filename = ('results_3dsurf_%s' % (run_nb))

    #    Save File of Run Description   #
    if not os.path.isfile(('%s/description.txt' % (filename))):
        with file(('%s/description.txt' % (filename)),'w') as outfile:
            outfile.write('Filename : {0}'.format(os.path.basename(__file__)))
            outfile.write('\nTest Description: ')
            outfile.write(description)
            

    mcmc = MCMC(simtime, samples, nCommunities, core_data, core_depths, data_vec, timestep, 
        filename, xmlinput, sedsim, sedlim, flowsim,flowlim, 
        assemblage, vis, description, v1, v1_title, v2, v2_title)

    [pos_v1, pos_v2, pos_likl] = mcmc.likelihood_surface()

    print 'Successfully sampled'
    
    print 'Finished producing Likelihood Surface'
if __name__ == "__main__": main()
