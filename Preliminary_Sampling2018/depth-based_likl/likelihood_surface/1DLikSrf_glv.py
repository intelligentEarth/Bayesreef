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
        xmlinput, sedsim, sedlim, flowsim, flowlim, min_v, max_v, assemblage, vis, description,
        v1, v1_title):
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
        self.min_v = min_v
        self.max_v = max_v
        self.simtime = simtime
        self.assemblage = assemblage
        self.font = 10
        self.width = 1
        self.d_sedprop = float(np.count_nonzero(core_data[:,self.communities]))/core_data.shape[0]
        self.initial_sed = []
        self.initial_flow = []
        self.true_m = 0.1
        self.true_ax = -0.01
        self.true_ay = -0.03
        self.description = description
        self.var1= v1
        self.var1_title = v1_title

    def run_Model(self, reef, input_vector):
        reef.convertVector(self.communities, input_vector, self.sedsim, self.flowsim) #model.py
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
        output_core = reef.plot.convertDepthStructure(self.communities, self.core_depths) #modelPlot.py
        # predicted_core = reef.convert_core(self.communities, output_core, self.core_depths) #model.py
        # return predicted_core
        return output_core

    def plotFunctions(self, fname, v1, likelihood, diff, rmse):

        font = self.font
        width = self.width
        X = v1
        Y = likelihood
        print 'X shape ', X.shape, 'Y shape ', Y.shape
        fig = plt.figure(figsize=(6,4))
        ax1 = fig.add_subplot(111)
        ax1.set_title('%s' % self.description, fontsize= font+2)#, y=1.02)
        ax1.set_facecolor('#f2f2f3')
        ax1.set_xlabel('%s' % self.var1_title)
        ax1.set_ylabel('Likelihood')
        ax1.plot(X,Y)
        ax1.set_xlim(X.min(), X.max())
        fig.tight_layout()
        plt.savefig('%s/1dsurf.png'% (fname), bbox_inches='tight', dpi=300, transparent=False)
        plt.clf()

        fig = plt.figure(figsize=(6,4))
        ax= fig.add_subplot(111)
        ax.set_facecolor('#f2f2f3')
        plt.plot(v1,diff)
        plt.title("Difference score evolution", size=self.font+2)
        plt.ylabel("Difference", size=self.font+1)
        plt.xlabel('%s' % self.var1_title)
        plt.xlim(X.min(), X.max())
        plt.savefig('%s/diff_evol.png' % (self.filename), bbox_inches='tight',dpi=300,transparent=False)
        fig.tight_layout()
        plt.clf()

        fig = plt.figure(figsize=(6,4))
        ax= fig.add_subplot(111)
        ax.set_facecolor('#f2f2f3')
        plt.plot(v1,rmse)
        plt.title("RMSE evolution", size=self.font+2)
        plt.ylabel("RMSE", size=self.font+1)
        plt.xlabel('%s' % self.var1_title)
        plt.xlim(X.min(), X.max())
        plt.savefig('%s/rmse_evol.png' % (self.filename), bbox_inches='tight',dpi=300,transparent=False)
        fig.tight_layout()
        plt.clf()

    def save_params(self, var1, likl, diff, rmse):    
        ### SAVE RECORD OF ACCEPTED PARAMETERS ###  
        if not os.path.isfile(('%s/data.csv' % (self.filename))):
            with file(('%s/data.csv' % (self.filename)),'wb') as outfile:
                writer = csv.writer(outfile, delimiter=',')
                writer.writerow(["Var", "Likl", "Diff", "RMSE"])
                data = [var1, likl, diff, rmse]
                writer.writerow(data)
        else:
            with file(('%s/data.csv' % (self.filename)),'ab') as outfile:
                writer = csv.writer(outfile, delimiter=',')
                data = [var1, likl, diff, rmse]
                writer.writerow(data)

    def diff_score(self, sim_data,synth_data,intervals):
        maxprop = np.zeros((intervals,self.communities+1))
        for n in range(intervals):
            idx_synth = np.argmax(synth_data[n,:])
            idx_sim = np.argmax(sim_data[n,:])
            if ((sim_data[n,self.communities] != 1.) and (idx_synth == idx_sim)): #where sediment !=1 and max proportions are equal:
                maxprop[n,idx_synth] = 1
        same= np.count_nonzero(maxprop)
        same = float(same)/intervals
        diff = 1-same
        return diff*100

    def rmse(self, sim, obs):
        # where there is 1 in the sed column, count
        sed = np.count_nonzero(sim[:,self.communities])
        p_sedprop = (float(sed)/sim.shape[0])
        sedprop = np.absolute((self.d_sedprop - p_sedprop)*0.5)
        rmse =(np.sqrt(((sim - obs) ** 2).mean()))*0.5
        return rmse + sedprop
    
    def NoiseToData(self,intervals,sim_data):
        # Function to add noise to simulated data to create synthetic core data with noise.
        synth_data = self.core_data #np.full((sim_data.shape[0],sim_data.shape[1]),100)
        list_cutpoints = []
        for n in range(1,intervals):
            a = np.argmax(sim_data[n])
            b = np.argmax(sim_data[n-1])
            if not (a == b):
                list_cutpoints = np.append(list_cutpoints, n)
                print 'list', list_cutpoints
        for idx, val in enumerate(list_cutpoints):
            print 'val', val
            for x in range(int(val-2),int(val+3)):
                print 'x',x
                synth_data[x,:] = np.random.multinomial(1, sim_data[x], size=1)
        print 'synth_data', synth_data

        with file('%s/data_probstic.txt' % (self.filename),'wb') as outfile:
            for h in range(intervals):
                rev = -1-h
                outfile.write('{0}\t'.format(self.core_depths[h]))
                for c in range(self.communities+1):
                    data_str = str(synth_data[h,c])
                    outfile.write('{0}\t'.format(data_str))
                outfile.write('\n')
        return

    def probabilisticLikelihood(self, reef, core_data, input_v):
        sim_propn = self.run_Model(reef, input_v)
        T_sim_propn = sim_propn.T
        intervals = T_sim_propn.shape[0]
        
        # # Uncomment if noisy synthetic data is required.
        # self.NoiseToData(intervals,T_sim_propn)
        
        log_core = np.log(T_sim_propn)
        log_core[log_core == -inf] = 0
        z = log_core * core_data
        likelihood = np.sum(z)
        diff = self.diff_score(T_sim_propn,core_data, intervals)
        rmse = self.rmse(T_sim_propn, core_data)
        return [likelihood, sim_propn, diff, rmse]

    # def noiseToDataLikelihood(self, reef, core_data, input_v):
    #     pred_core = self.run_Model(reef, input_v)
    #     pred_core = pred_core.T
    #     pred_core_w_noise = np.zeros((pred_core.shape[0], pred_core.shape[1]))
    #     intervals = pred_core.shape[0]
    #     for n in range(intervals):
    #        pred_core_w_noise[n,:] = np.random.multinomial(1,pred_core[n],size=1)
    #     pred_core_w_noise = pred_core_w_noise/1
    #     z = np.zeros((intervals,self.communities+1))  
    #     z = pred_core_w_noise * core_data
    #     loss = np.log(z)
    #     loss[loss == -inf] = 0
    #     loss = np.sum(loss)
    #     diff = self.diff_score(pred_core_w_noise,core_data, intervals)
    #     loss = np.exp(loss)
    #     return [loss, pred_core_w_noise, diff]

    def deterministicLikelihood(self, reef, core_data, input_v):
        sim_data = self.run_Model(reef, input_v)
        sim_data = sim_data.T
        intervals = sim_data.shape[0]
        z = np.zeros((intervals,self.communities+1))    
        for n in range(intervals):
            idx_data = np.argmax(core_data[n,:])
            idx_model = np.argmax(sim_data[n,:])
            if ((sim_data[n,self.communities] != 1.) and (idx_data == idx_model)): #where sediment !=1 and max proportions are equal:
                z[n,idx_data] = 1
        same = np.count_nonzero(z)
        same = float(same)/intervals
        diff = 1-same
        rmse = self.rmse(sim_data, core_data)
        z = z + 0.1
        z = z/(1+(1+self.communities)*0.1)
        l1 = np.log(z)
        l2 = l1
        likelihood = np.exp(l2)
        return [np.sum(likelihood), sim_data, diff, rmse]
               
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
        m = self.true_m
        ax = self.true_ax
        ay = self.true_ay

        #Define min/max of parameter of interest 
        p1_v1 = self.min_v
        p2_v1 = self.max_v
        # Set number and value of iterates
        s_v1 = np.linspace(p1_v1, p2_v1, num=samples, endpoint=True)
        print 's_v1', s_v1
        # Create storage for data
        dimx = s_v1.shape[0]
        pos_likl = np.zeros(dimx)
        pos_v1 = np.zeros(samples) 
        pos_diff = np.zeros(samples)
        pos_rmse = np.zeros(samples)
        

        start = time.time()
        i = 0
        for a in range(len(s_v1)):
            print 'sample: ', i
            print '%s:' % self.var1, s_v1[a]
            p_v1 = s_v1[a] # Update parameters


            # USER DEFINED: Substitute generated variables into proposal vector 

            # ay = p_v1
            
            # Proposal to be passed to runModel
            v_proposal = np.concatenate((sed1,sed2,sed3,sed4,flow1,flow2,flow3,flow4))
            v_proposal = np.append(v_proposal,(ax,ay,m))
            [likelihood, pred_data, diff, rmse] = self.probabilisticLikelihood(reef, self.core_data, v_proposal)
            print 'Likelihood:', likelihood, 'and difference score:', diff

            pos_v1[i] = p_v1
            pos_likl[i] = likelihood
            pos_diff[i] = diff
            pos_rmse[i] = rmse
            self.save_params(pos_v1[i], pos_likl[i], pos_diff[i], pos_rmse[i])
            i += 1

        end = time.time()
        total_time = end - start
        self.plotFunctions(self.filename, s_v1, pos_likl,pos_diff, pos_rmse)
        print 'Counter:', i, '\nTime elapsed:', total_time, '\npos_likl.shape:', pos_likl.shape 
        
        return (pos_v1, pos_likl)

def main():
    random.seed(time.time())

    #    Set all input parameters    #

    # USER DEFINED: parameter names and plot titles.
    samples= 100
    assemblage= 2

    # v1 = 'Malthusian Parameter'
    # v1_title = r'$\varepsilon$'
    # min_v =0.01
    # max_v = 0.15
    
    # v1 = 'Main diagonal'
    # v1_title = r'$\alpha_m$'
    # min_v =-0.15
    # max_v = 0

    v1 = 'Super-/sub-diagonals'
    v1_title = r'$\alpha_s$'# r'$\varepsilon$'#r'$\alpha_m$'
    min_v =-0.15
    max_v = 0
    description = '1D likelihood surface, %s' % v1

    nCommunities = 3
    simtime = 8500
    timestep = np.arange(0,simtime+1,50)
    xmlinput = 'input_synth.xml'
    datafile = 'data/synth_core_vec.txt'
    core_depths, data_vec = np.genfromtxt(datafile, usecols=(0, 1), unpack = True) 
    core_data = np.loadtxt('data/synth_core_prop.txt', usecols=(1,2,3,4))
    vis = [False, False] # first for initialisation, second for cores
    sedsim, flowsim = True, True
    sedlim = [0., 0.005]
    flowlim = [0.,0.3]
    
    run_nb = 0
    while os.path.exists('1dsurf_glv_%s' % (run_nb)):
        run_nb+=1
    if not os.path.exists('1dsurf_glv_%s' % (run_nb)):
        os.makedirs('1dsurf_glv_%s' % (run_nb))
    filename = ('1dsurf_glv_%s' % (run_nb))

    #    Save File of Run Description   #
    if not os.path.isfile(('%s/description.txt' % (filename))):
        with file(('%s/description.txt' % (filename)),'w') as outfile:
            outfile.write('Filename : {0}'.format(os.path.basename(__file__)))
            outfile.write('\nTest Description: ')
            outfile.write(description)
            outfile.write('\nSamples: {0}'.format(samples))
            

    mcmc = MCMC(simtime, samples, nCommunities, core_data, core_depths, data_vec, timestep,  filename, xmlinput, 
                sedsim, sedlim, flowsim,flowlim, min_v, max_v, assemblage, vis, description,v1, v1_title)
    [pos_v1, pos_likl] = mcmc.likelihood_surface()

    print 'Successfully sampled'
    
    print 'Finished producing Likelihood Surface'
if __name__ == "__main__": main()
