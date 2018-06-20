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
# from matplotlib import cm
from matplotlib.cm import terrain, plasma, Set2
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
        gt_depths, gt_timelay, gt_vec_t, gt_prop_t, min_v, max_v, assemblage, description,v1, v1_title):
        
        self.font = 12
        self.width = 1
        self.colors = terrain(np.linspace(0, 1.8, 14)) #len(reef.core.coralH)+10))
        self.colors2 = plasma(np.linspace(0, 1, 174)) #len(reef.core.layTime)+3))
        self.d_sedprop = float(np.count_nonzero(gt_prop_t[:,communities]))/gt_prop_t.shape[0]
        
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
        self.gt_timelay = gt_timelay
        self.gt_prop_t = gt_prop_t
        self.gt_vec_t = gt_vec_t
        
        self.true_sed = []
        self.true_flow = []
        self.true_m = 0.08
        self.true_ax = -0.01
        self.true_ay = -0.03
        self.min_v = min_v
        self.max_v = max_v
        
        self.assemblage = assemblage
        self.description = description
        self.var1= v1
        self.var1_title = v1_title


    def runModel(self, reef, input_vector):
        reef.convertVector(self.communities, input_vector, self.sedsim, self.flowsim) #model.py
        self.true_sed, self.true_flow = reef.load_xml(self.input, self.sedsim, self.flowsim)
        # if self.vis[0] == True:
            # reef.core.initialSetting(size=(8,2.5), size2=(8,3.5)) # View initial parameters
        reef.run_to_time(self.simtime,showtime=100.)
        # if self.vis[1] == True:
        #     reef.plot.drawCore(lwidth = 3, colsed=self.colors, coltime = self.colors2, size=(9,8), font=8, dpi=300)
        sim_output, sim_timelay = reef.plot.convertTimeStructure() #modelPlot.py
        return sim_output, sim_timelay
    
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

    def modelOutputParameters(self, prop_t, vec_t, timelay):

        n = timelay.size # no. of data points in gt output #171
        s = 1
        cpts = np.zeros(n)# (171,)
        cpts[0] = timelay[0]
        ca_props = np.zeros((n, prop_t.shape[1]))# (171,5)
        
        for i in range(1,n):
            if vec_t[i] != vec_t[i-1]:
                cpts[s] = timelay[i-1]
                ca_props[s-1] = prop_t[i-1,:]
                s += 1
            if i == n-1:
                cpts[s] = timelay[i]
                ca_props[s-1] = prop_t[i,:]
        S = s
        cpts = np.trim_zeros(cpts, 'b') # append a zero on the end afterwards
        ca_props = ca_props[0:S,:]
        return S, cpts, ca_props

    def noGrowthColumn(self, sim_prop):
        # Creates additional binary column that takes a value of 1 where there is no growth, otherwise 0.
        v_nogrowth = np.zeros((sim_prop.shape[0],1))
        for a in range(sim_prop.shape[0]):
            if np.amax(sim_prop[a,:]) == 0.:
                v_nogrowth[a,:] = 1.
        sim_prop = np.append(sim_prop,v_nogrowth,axis=1)
        return sim_prop

    def likelihoodWithDependence(self,reef, input_v, S_star, cpts_star, ca_props_star):
        """
        (1) compute the number of segments (S)
        (2) compute the location of the cutpoints (xi) 
        (3) find the proportion in the segment (ca_props)
        """
        sim_prop_t, sim_timelay = self.runModel(reef, input_v)
        # sim_vec_d = self.convertCoreFormat(sim_prop_d.T)
        sim_vec_t = self.convertCoreFormat(sim_prop_t)
        sim_prop_t5 = self.noGrowthColumn(sim_prop_t)

        # Counting segments, recording location of cutpoints and associated cagal assemblage proportions
        print 'S_star',S_star, '\n cpts_star',cpts_star,'\n ca_props_star props', ca_props_star
        S, cpts, ca_props = self.modelOutputParameters(sim_prop_t5,sim_vec_t,sim_timelay)
        print 's',S, '\n cpts',cpts,'\n ca props', ca_props
        # First reject if number of segments in sim != S_star
        if S != S_star:
            likelihood=0
            diff = 100
            rmse = 100
            return [likelihood, diff, rmse, sim_prop_t5]
        # Likelihood for cutpoints conditional on S_star
        likl_cpts_star = np.zeros(S_star)
        for j in range(S_star):
            if j == 0:
                distance = cpts[j+1]-cpts[j]
            else:
                distance = min((cpts[j+1]-cpts[j]),(cpts[j]-cpts[j-1]))
            likl_cpts_star[j] = stats.norm.pdf(cpts_star[j],cpts[j],float(distance)/2.)
        likl_cpts_star = np.ma.masked_invalid(likl_cpts_star)
        print 'likl_cpts_star:',likl_cpts_star
        like_all_cpts_star = np.prod(likl_cpts_star) #correct product after removing NaNs
        print 'like_all_cpts_star:',like_all_cpts_star
        # Multinomial likelihood - a product of the no. of segments
        likl_ca_prop= np.zeros((S_star,5))
        for k in range(S_star):
            likl_ca_prop[k,:] = np.random.multinomial(1,ca_props[k,:],size=1)
        print 'likl_ca_prop', likl_ca_prop
        likl_ca_prop = (likl_ca_prop*100)+1
        like_all_coral = np.prod(likl_ca_prop)
        total_likelihood = like_all_cpts_star*like_all_coral
        
        diff = self.diffScore(sim_prop_t5,self.gt_prop_t, sim_timelay.size)
        rmse= self.rmse(sim_prop_t5, self.gt_prop_t)
        return [total_likelihood, diff, rmse, sim_prop_t]


    def likelihoodWithProps(self, reef, gt_prop_t, input_v):
        sim_prop_t, sim_timelay = self.runModel(reef, input_v)
        sim_prop_t5 = self.noGrowthColumn(sim_prop_t)
        intervals = sim_prop_t5.shape[0]
        # # Uncomment if noisy synthetic data is required.
        # self.NoiseToData(intervals,sim_prop_t5)
        log_core = np.log(sim_prop_t5+0.0001)
        log_core[log_core == -inf] = 0
        z = log_core * gt_prop_t
        likelihood = np.sum(z)
        diff = self.diffScore(sim_prop_t5,gt_prop_t, intervals)
        rmse = self.rmse(sim_prop_t5, gt_prop_t)
        # sim_vec_t = self.convertCoreFormat(sim_prop_t5)
        # sim_vec_d = self.convertCoreFormat(sim_prop_d.T)
        return [likelihood, diff, rmse, sim_prop_t5]
           
    def likelihoodWithDominance(self, reef, gt_prop_t, input_v):
        sim_data_t, sim_timelay = self.runModel(reef, input_v)
        sim_data_t5 = self.noGrowthColumn(sim_data_t)
        intervals = sim_data_t5.shape[0]
        z = np.zeros((intervals,sim_data_t5.shape[1]))    
        for n in range(intervals):
            idx_data = np.argmax(gt_prop_t[n,:])
            idx_model = np.argmax(sim_data_t5[n,:])
            if ((sim_data_t5[n,self.communities] != 1.) and (idx_data == idx_model)): #where sediment !=1 and max proportions are equal:
                z[n,idx_data] = 1
        diff = 1. - (float(np.count_nonzero(z))/intervals)# Difference score calculation
        z = z + 0.1
        z = z/(1+(1+self.communities)*0.1)
        log_z = np.log(z)
        likelihood = np.sum(log_z)
        rmse = self.rmse(sim_data_t5, gt_prop_t)
        # sim_vec_t = self.convertCoreFormat(sim_data_t5)
        # sim_vec_d = self.convertCoreFormat(sim_data_d.T)
        return [likelihood, diff, rmse, sim_data_t5]
               
    def plotFunctions(self, fname, v1, likelihood, diff, rmse):

        font = self.font
        width = self.width
        X = v1
        Y = likelihood
        print 'X shape: ', X.shape, 'Y shape: ', Y.shape
        fig = plt.figure(figsize=(6,3))
        ax1 = fig.add_subplot(111)
        ax1.set_title('%s' % self.description, fontsize= font+1)#, y=1.02)
        ax1.set_facecolor('#f2f2f3')
        ax1.set_xlabel('%s' % self.var1_title)
        ax1.set_ylabel('Likelihood')
        ax1.plot(X,Y)
        ax1.set_xlim(X.min(), X.max())
        ax1.set_ylim(Y.max()-10,0)
        fig.tight_layout()
        plt.savefig('%s/1dsurf.png'% (fname), bbox_inches='tight', dpi=300, transparent=False)
        plt.clf()

        fig = plt.figure(figsize=(6,3))
        ax= fig.add_subplot(111)
        ax.set_facecolor('#f2f2f3')
        plt.plot(v1,diff)
        plt.title("Difference score evolution", size=self.font+1)
        plt.ylabel("Difference", size=self.font+1)
        plt.xlabel('%s' % self.var1_title)
        plt.xlim(X.min(), X.max())
        plt.ylim(Y.max()-10,0)
        plt.savefig('%s/diff_evol.png' % (self.filename), bbox_inches='tight',dpi=300,transparent=False)
        fig.tight_layout()
        plt.clf()

        fig = plt.figure(figsize=(6,3))
        ax= fig.add_subplot(111)
        ax.set_facecolor('#f2f2f3')
        plt.plot(v1,rmse)
        plt.title("RMSE evolution", size=self.font+1)
        plt.ylabel("RMSE", size=self.font+1)
        plt.xlabel('%s' % self.var1_title)
        plt.xlim(X.min(), X.max())
        plt.ylim(Y.max()-10,0)
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

    def saveCore(self,reef,naccept):
        path = '%s/%s' % (self.filename, naccept)
        if not os.path.exists(path):
            os.makedirs(path)
        
        #     Initial settings     #
        reef.core.initialSetting(size=(8,2.5), size2=(8,4.5), dpi=300, fname='%s/a_thres_%s_' % (path, naccept))        
        #      Community population evolution    #
        reef.plot.speciesDepth(colors=self.colors, size=(8,4), font=8, dpi=300, fname =('%s/b_popd_%s.png' % (path,naccept)))
        reef.plot.speciesTime(colors=self.colors, size=(8,4), font=8, dpi=300,fname=('%s/c_popt_%s.png' % (path,naccept)))
        reef.plot.accomodationTime(size=(8,4), font=8, dpi=300, fname =('%s/d_acct_%s.pdf' % (path,naccept)))
        
        #      Draw core      #
        reef.plot.drawCore(lwidth = 3, colsed=self.colors, coltime = self.colors2, size=(9,8), font=8, dpi=300, 
                           figname=('%s/e_core_%s' % (path, naccept)), filename=('%s/core_%s.csv' % (path, naccept)), sep='\t')
        return

    def NoiseToData(self,intervals,sim_data):
        # Function to add noise to simulated data to create synthetic core data with noise.
        synth_data = self.gt_prop_t #np.full((sim_data.shape[0],sim_data.shape[1]),100)
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
                outfile.write('{0}\t'.format(self.gt_depths[h]))
                for c in range(self.communities+1):
                    data_str = str(synth_data[h,c])
                    outfile.write('{0}\t'.format(data_str))
                outfile.write('\n')
        return

    def createData(self,sim_data, sim_timelay):
        synth_data = np.zeros((sim_data.shape))
        with file('%s/data_timestructure.txt' % (self.filename),'wb') as outfile:
            for h in range(sim_timelay.shape[0]):
                rev = -1-h
                outfile.write('{0}\t'.format(sim_timelay[h]))
                for c in range(self.communities+1):
                    data_str = str(sim_data[h,c])
                    outfile.write('{0}\t'.format(data_str))
                outfile.write('\n')
        return

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
        print 'Vector of all iterates:', s_v1
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
            flow4[assemblage-1] = p_v1
            
            # Proposal to be passed to runModel
            v_proposal = np.concatenate((sed1,sed2,sed3,sed4,flow1,flow2,flow3,flow4))
            v_proposal = np.append(v_proposal,(ax,ay,m))
            # S_star, cpts_star, ca_props_star = self.modelOutputParameters(self.gt_prop_t,self.gt_vec_t,self.gt_timelay)
            # [likelihood, diff, rmse, pred_data] = self.likelihoodWithDependence(reef, v_proposal, S_star, cpts_star, ca_props_star)
            [likelihood, diff, rmse, pred_data] = self.likelihoodWithProps(reef, self.gt_prop_t, v_proposal)
            print 'Likelihood:', likelihood, 'and difference score:', diff
            pos_v1[i] = p_v1
            pos_likl[i] = likelihood
            pos_diff[i] = diff
            pos_rmse[i] = rmse
            self.save_params(pos_v1[i], pos_likl[i], pos_diff[i], pos_rmse[i])
            # self.saveCore(reef,i)
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
    samples= 200
    assemblage= 1

    title = ['Shallow', 'Mod-deep', 'Deep'] 
    sed1=[0.0009, 0.0015, 0.0023]
    sed2=[0.0015, 0.0017, 0.0024]
    sed3=[0.0016, 0.0028, 0.0027]
    sed4=[0.0017, 0.0031, 0.0043]
    flow1=[0.055, 0.008 ,0.]
    flow2=[0.082, 0.051, 0.]
    flow3=[0.259, 0.172, 0.058] 
    flow4=[0.288, 0.185, 0.066]
    sedlim = [0., 0.003]
    flowlim = [0.,0.3]

    v1_title = r'$f_{flow}^4$'
    v1 = 'Hydrodynamic energy threshold:\n %s assemblage (%s)' % (title[assemblage-1],v1_title)
    
    # min_v = flowlim[0]
    # max_v = flow2[assemblage-1]

    # min_v = flow1[assemblage-1]
    # max_v = flow3[assemblage-1]

    # min_v = flow2[assemblage-1]
    # max_v = flow4[assemblage-1]

    min_v = flow3[assemblage-1]
    max_v = flowlim[1]

    # v1_title = r'$f_{sed}^4$'
    # v1 = 'Sediment exposure threshold:\n %s assemblage (%s)' % (title[assemblage-1], v1_title)
    
    # min_v = sedlim[0]
    # max_v = sed2[assemblage-1]

    # min_v = sed1[assemblage-1]
    # max_v = sed3[assemblage-1]

    # min_v = sed2[assemblage-1]
    # max_v = sed4[assemblage-1]

    # min_v = sed3[assemblage-1]
    # max_v = sedlim[1]

    description = 'Marginal likelihood: %s assemblage, %s' % (title[assemblage-1],v1_title)
    description2 = 'self.likelihoodWithProps'
    nCommunities = 3
    simtime = 8500
    xmlinput = 'input_synth.xml'
    gt_depths = np.genfromtxt('data/synthdata_d_vec_08.txt', usecols=(0), unpack=True)
    synth_data = 'data/synthdata_t_prop_08_1.txt'
    gt_prop_t = np.loadtxt(synth_data, usecols=(1,2,3,4,5))    
    synth_vec = 'data/synthdata_t_vec_08_1.txt'
    gt_timelay, gt_vec_t = np.genfromtxt(synth_vec, usecols=(0, 1), unpack = True) 
    gt_timelay = gt_timelay[::-1]
    vis = [False, False]
    sedsim, flowsim = True, True
    sedlim = [0., 0.005]
    flowlim = [0.,0.3]
    
    run_nb = 0
    path_name = 'results-1d-thres'
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
        gt_depths, gt_timelay,gt_vec_t, gt_prop_t, min_v, max_v, assemblage, description,v1, v1_title)

    [pos_v1, pos_likl] = mcmc.likelihood_surface()

    print 'Successfully sampled'
    
    print 'Finished producing Likelihood Surface'
if __name__ == "__main__": main()
