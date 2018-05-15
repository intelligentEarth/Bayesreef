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
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from pyReefCore.model import Model
from pyReefCore import (plotResults, saveParameters)
import fnmatch
import matplotlib as mpl
from cycler import cycler
from scipy import stats 
from matplotlib.cm import terrain, plasma, Set2

cmap=plt.cm.Set2
c = cycler('color', cmap(np.linspace(0,1,8)) )
plt.rcParams["axes.prop_cycle"] = c

class MCMC():
    def __init__(self, filename, xmlinput, simtime, samples, nCommunities, sedsim, sedlim, flowsim, flowlim, vis,
        data_depths, data_vec_d, data_times, data_vec_t, core_data, true_pop, true_timeCarb,
        min_m, max_m, true_m, step_m, min_a, max_a, true_ax, true_ay,step_a, step_sed, step_flow, assemblage):

        self.font = 10
        self.width = 1
        self.colors = terrain(np.linspace(0, 1.8, 14)) #len(reef.core.coralH)+10))
        self.colors2 = plasma(np.linspace(0, 1, 174)) #len(reef.core.layTime)+3))
        self.filename = filename
        self.input = xmlinput
        self.simtime = simtime
        self.samples = samples
        self.communities = nCommunities
        self.sedsim = sedsim
        self.sedlim = sedlim
        self.flowlim = flowlim
        self.flowsim = flowsim
        self.vis = vis   
        self.data_depths = data_depths
        self.data_vec_d = data_vec_d
        self.data_times = data_times
        self.data_vec_t = data_vec_t
        self.core_data = core_data
        self.true_pop = true_pop
        self.true_timeCarb = true_timeCarb
        self.min_m = min_m
        self.max_m = max_m
        self.min_a = min_a
        self.max_a = max_a
        self.step_m =step_m #0.002 # <1%
        self.step_a =step_a #0.002 # <1%
        self.step_sed = step_sed #0.0001 # 2%
        self.step_flow = step_flow #0.0015 # 0.05%        
        self.true_m = true_m
        self.true_ax = true_ax
        self.true_ay = true_ay
        self.true_sed = []
        self.true_flow = []
        self.assemblage = assemblage
        self.d_sedprop = float(np.count_nonzero(core_data[:,self.communities]))/core_data.shape[0]
        
    def runModel(self, reef, input_vector):
        reef.convertVector(self.communities, input_vector, self.sedsim, self.flowsim) #model.py
        self.true_sed, self.true_flow = reef.load_xml(self.input, self.sedsim, self.flowsim)
        # if self.vis[0] == True:
            # reef.core.initialSetting(size=(8,2.5), size2=(8,3.5)) # View initial parameters
        reef.run_to_time(self.simtime,showtime=100.)
        # if self.vis[1] == True:
        #     reef.plot.drawCore(lwidth = 3, colsed=self.colors, coltime = self.colors2, size=(9,8), font=8, dpi=300)
        sim_output_t, sim_timelay = reef.plot.convertTimeStructure() #modelPlot.py
        sim_output_d = reef.plot.convertDepthStructure(self.communities, self.data_depths)
        return sim_output_t, sim_output_d,sim_timelay

    def convertCoreFormat(self, core):
        vec = np.zeros(core.shape[0])
        for n in range(len(vec)):
            if not all(v == 0 for v in core[n,:]):
                idx = np.argmax(core[n,:])# get index,
                vec[n] = idx+1 # +1 so that zero is preserved as 'none'
            else:
                vec[n] = -1.
        return vec

    def diffScore(self, sim_data,synth_data,intervals):
        maxprop = np.zeros((intervals,self.communities+1))
        for n in range(intervals):
            idx_synth = np.argmax(synth_data[n,:])
            idx_sim = np.argmax(sim_data[n,:])
            if ((sim_data[n,self.communities] != 1.) and (idx_synth == idx_sim)): #where sediment !=1 and max proportions are equal:
                maxprop[n,idx_synth] = 1
        same= np.count_nonzero(maxprop)
        same = float(same)/intervals
        diff = (1-same)*100
        return diff

    def rmse(self, sim, obs):
        # where there is 1 in the sed column, count
        sed = np.count_nonzero(sim[:,self.communities])
        p_sedprop = (float(sed)/sim.shape[0])
        sedprop = np.absolute((self.d_sedprop - p_sedprop)*0.5)
        rmse =(np.sqrt(((sim - obs) ** 2).mean()))*0.5
        return rmse + sedprop

    def nonStationaryLikelihood(self,reef, synth_data, synth_depths, input_v):
        """
        1. Locate position of cut-off points and assemblage transition in the synthetic data
        1a) produce synthetic data from the p2 nd array, and make the 'data points' the location of the cut-off points and the value in that array
        2. From the bottom of sim core, find the location of the cut-off points and what is contained in each section 
        3a.  store probabilities in an empty list z, to get product of at end
        3b. In loop,
            check and see if first sim cut-point is LEQ synth cut-point
            For given sim cut-point, check that all below sim cut-point is same assemblage as synth cut point
            If it's the same, write to z the proportion of max. cut point 
            if it is different, print likelihood as 0, break loop, return zero. 
            continue to second cut-point
        """
        sim_propn, sim_depths = self.runModel(reef,input_v) # write code to get depths for the proportions 
        sim_propn = sim_propn.T
        print 'sim_propn', sim_propn
        print 'sim_depths', sim_depths
        intervals = sim_propn.shape[0]
        cutpoints_assm = []
        cutpoints_depth = []
        for i in range(1,intervals):
            a = np.argmax(sim_propn[i])
            b = np.argmax(sim_propn[i-1])
            if not (a == b):
                cutpoints_assm = np.append(cutpoints_assm, a)
                cutpoints_depth = np.append(cutpoints_depth, sim_depths[i])

        print 'list of cut-points', cutpoints_assm
        print 'depths of cut-points', cutpoints_depth
        
        z = np.zeros(len(cutpoints_assm))
        for idx, sim_cp in enumerate(cutpoints_depth):
            print 'cutpoint', sim_cp
            if sim_cp <= synth_depths[idx]:
                if np.argmax(cutpoints_assm[idx]) == np.argmax(synth_data[idx]):
                    z = np.append(z, np.argmax(cutpoints_assm[idx]))
                else:
                    likelihood=0
                    break
            else:
                likelihood=0
                break
        return likelihood

        
    # def noisyDataLikelihood(self, reef, core_data, input_v):
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

    def likelihoodWithPropn(self, reef, core_data, input_v):
        sim_t_propn, sim_d_propn, sim_timelay = self.runModel(reef, input_v)
        intervals = sim_t_propn.shape[0]
        # # Uncomment if noisy synthetic data is required.
        # self.NoiseToData(intervals,sim_t_propn)
        log_core = np.log(sim_t_propn+0.0001)
        log_core[log_core == -inf] = 0
        z = log_core * core_data
        likelihood = np.sum(z)
        diff = self.diffScore(sim_t_propn,core_data, intervals)
        # rmse = self.rmse(sim_t_propn, core_data)
        return [likelihood, sim_t_propn, sim_d_propn.T, diff]
           
    def likelihoodWithDominance(self, reef, core_data, input_v):
        sim_t_data, sim_d_data, sim_timelay = self.runModel(reef, input_v)
        # sim_t_data = sim_t_data.T
        intervals = sim_t_data.shape[0]
        z = np.zeros((intervals,self.communities+1))    
        for n in range(intervals):
            idx_data = np.argmax(core_data[n,:])
            idx_model = np.argmax(sim_t_data[n,:])
            if ((sim_t_data[n,self.communities] != 1.) and (idx_data == idx_model)): #where sediment !=1 and max proportions are equal:
                z[n,idx_data] = 1
        same = np.count_nonzero(z)
        same = float(same)/intervals
        diff = 1-same
        # rmse = self.rmse(sim_t_data, core_data)
        z = z + 0.1
        z = z/(1+(1+self.communities)*0.1)
        likelihood = np.log(z)
        # l1 = likelihood
        # likelihood = np.exp(l1)
        return [np.sum(likelihood), sim_t_data, sim_d_data.T, diff]

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

    def proposalJump(self, current, low_limit, high_limit, jump_width):
        proposal = current + np.random.normal(0, jump_width)
        if proposal >= high_limit:
            proposal = current
        elif proposal <= low_limit:
            proposal = current
        return proposal
    
    def sampler(self):
        samples = self.samples
        data_vec_t = self.data_vec_t
        data_vec_d = self.data_vec_d
        data_depths = self.data_depths
        communities = self.communities
        core_data = self.core_data

        with file(('%s/description.txt' % (self.filename)),'a') as outfile:
            outfile.write('\nstep_m: {0}'.format(self.step_m))
            outfile.write('\nstep_a: {0}'.format(self.step_a))
            outfile.write('\nstep_sed: {0}'.format(self.step_sed))
            outfile.write('\nstep_flow: {0}'.format(self.step_flow))

        # Create space to store accepted samples for posterior 
        pos_sed1 = np.zeros((samples , communities)) # sample rows, communities column
        pos_sed2 = np.zeros((samples , communities)) 
        pos_sed3 = np.zeros((samples , communities))
        pos_sed4 = np.zeros((samples , communities))
        pos_flow1 = np.zeros((samples , communities))
        pos_flow2 = np.zeros((samples , communities))
        pos_flow3 = np.zeros((samples , communities))
        pos_flow4 = np.zeros((samples , communities))
        pos_ax = np.zeros(samples)
        pos_ay = np.zeros(samples)
        pos_m = np.zeros(samples)
        # Create space to store fx of all samples
        pos_samples_d = np.zeros((samples, data_vec_d.shape[0]))
        pos_samples_t = np.zeros((samples, core_data.shape[0]))
        

        #      INITIAL PREDICTION       #
            # sed1 = np.zeros(communities)
            # sed2 = np.zeros(communities)
            # sed3 = np.zeros(communities)
            # sed4 = np.zeros(communities)
            # if self.sedsim == True:
            #     for s in range(communities):
            #         sed1[s] = pos_sed1[0,s] = np.random.uniform(self.sedlim[0],self.sedlim[1])
            #         sed2[s] = pos_sed2[0,s] = np.random.uniform(sed1[s],self.sedlim[1])
            #         sed3[s] = pos_sed3[0,s] = np.random.uniform(sed2[s],self.sedlim[1])
            #         sed4[s] = pos_sed4[0,s] = np.random.uniform(sed3[s],self.sedlim[1])

            # flow1 = np.zeros(communities)
            # flow2 = np.zeros(communities)
            # flow3 = np.zeros(communities)
            # flow4 = np.zeros(communities)
            # if self.flowsim == True:
            #     for s in range(communities):
            #         #     relaxed constraints 
            #         flow1[s] = pos_flow1[0,s] = np.random.uniform(self.flowlim[0],self.flowlim[1])
            #         flow2[s] = pos_flow2[0,s] = np.random.uniform(flow1[s],self.flowlim[1])
            #         flow3[s] = pos_flow3[0,s] = np.random.uniform(flow2[s],self.flowlim[1])
            #         flow4[s] = pos_flow4[0,s] = np.random.uniform(flow3[s],self.flowlim[1])
            
            # cm_ax = pos_ax[0] = np.random.uniform(self.min_a,self.max_a)
            # cm_ay = pos_ay[0] = np.random.uniform(self.min_a,self.max_a)
            # m = pos_m[0] = np.random.uniform(self.min_m, self.max_m)

        sed1=[0.0009, 0.0015, 0.0023]
        sed2=[0.0015, 0.0017, 0.0024]
        sed3=[0.0016, 0.0028, 0.0027]
        sed4=[0.0017, 0.0031, 0.0043]
        flow1=[0.055, 0.008 ,0.]
        flow2=[0.082, 0.051, 0.]
        flow3=[0.259, 0.172, 0.058]
        flow4=[0.288, 0.185, 0.066]   
        cm_ax = pos_ax[0] = self.true_ax #np.random.uniform(self.max_a,0.)
        cm_ay = pos_ay[0] = self.true_ay #np.random.uniform(self.max_a,0.)
        
        cm_ay = pos_ay[0] = np.random.uniform(self.min_a, self.max_a)
        m = pos_m[0] = np.random.uniform(self.min_m, self.max_m)

        if (self.sedsim == True) and (self.flowsim == False):
            v_proposal = np.concatenate((sed1,sed2,sed3,sed4))
        elif (self.flowsim == True) and (self.sedsim == False):
            v_proposal = np.concatenate((flow1,flow2,flow3,flow4))
        elif (self.sedsim == True) and (self.flowsim == True):
            v_proposal = np.concatenate((sed1,sed2,sed3,sed4,flow1,flow2,flow3,flow4))
        v_proposal = np.append(v_proposal,(cm_ax,cm_ay,m))
        pos_v = np.zeros((samples, v_proposal.size))
        print v_proposal

        # Declare pyReef-Core and initialize
        reef = Model()
        [likelihood, sim_pred_t, sim_pred_d, diff] = self.likelihoodWithPropn(reef, core_data, v_proposal)
        print '\tInitial likelihood:', likelihood
        pos_diff = np.full(samples,diff)
        pos_likl = np.full(samples, likelihood)
        core_vec = self.convertCoreFormat(sim_pred_d)
        pos_samples_t[0,:] = self.convertCoreFormat(sim_pred_t)
        pos_samples_d[0,:] = self.convertCoreFormat(sim_pred_d)
        self.saveCore(reef, 'initial')

        naccept = 0
        count_list = []
        count_list.append(0)
        # Uncomment if you want to see posterior as the simulations runs
        # saveParameters.saveParameters(self.filename, self.sedsim, self.flowsim, naccept, 
        #     pos_m[0], pos_ax[0], pos_ay[0], 
        #     pos_sed1[0,], pos_sed2[0,], pos_sed3[0,], pos_sed4[0,],
        #     pos_flow1[0,], pos_flow2[0,], pos_flow3[0,], pos_flow4[0,],
        #     pos_diff[0],pos_likl[0], pos_samples_t[0,],pos_v[0,])
        
        print 'Begin sampling using MCMC random walk'
        ##########################################
        # PLOT INITIAL PREDICTIONS AND PROPOSALS #
        ##########################################
        # Initial prediction - Depth 
        x_tick_labels = ['No growth','Shallow', 'Mod-deep', 'Deep', 'Sediment']
        x_tick_values = [-1,1,2,3,4]
        fig = plt.figure(figsize=(3,6))
        ax = fig.add_subplot(111)
        ax.set_facecolor('#f2f2f3')
        ax.plot(data_vec_d, data_depths, label='Synthetic core', color='k')
        ax.plot(core_vec, data_depths, label='Initial predicted core')
        ax.set_title("Initial Prediction", size=self.font+2)
        plt.xticks(x_tick_values, x_tick_labels,rotation=70, fontsize=self.font+1)
        ax.set_ylabel("Core depth [m]",size=self.font+1)
        ax.set_ylim([0,np.amax(data_depths)])
        ax.set_ylim(ax.get_ylim()[::-1])
        plt.legend(frameon=False, prop={'size':self.font+1},bbox_to_anchor = (1.,0.1))
        fig.savefig('%s/begin.png' % (self.filename), bbox_inches='tight',dpi=200,transparent=False)
        plt.close()
        # Accumulating figure of all proposals - Depth
        finalfig_d = plt.figure(figsize=(3,6))
        axprop_d = finalfig_d.add_subplot(111)
        # axprop_d.plot(data_vec_d, data_depths, label='Synthetic core', color='k')
        axprop_d.plot(core_vec, data_depths)
        axprop_d.set_title("Accepted Proposals", size=self.font+2)
        plt.xticks(x_tick_values, x_tick_labels,rotation=70, fontsize=self.font+1)
        axprop_d.set_ylabel("Depth [m]",size=self.font+1)
        axprop_d.set_ylim([0,np.amax(data_depths)])
        axprop_d.set_ylim(axprop_d.get_ylim()[::-1])

        # Initial prediction - Time
        reef.plot.speciesTime(colors=self.colors, size=(10,5), font=self.font, dpi=200, fname=('%s/begin.png' % (self.filename)))

        # # Accumulating figure of all proposals - time
        finalfig_t, axprop_t = plt.subplots(1, figsize=(10,5))
        axprop_t.set_facecolor('#f2f2f3')
        timecarb,pop,names = reef.plot.getTimePlotParameters()
        for x in range(len(pop)):
            axprop_t.plot(timecarb, pop[x,:],linewidth=self.width)
            axprop_t.plot(self.true_timeCarb, self.true_pop[x,:], c=self.colors[x], linewidth=self.width, color='k')
        axprop_t.grid()
        axprop_t.set_ylabel('Population',size=self.font+2)
        axprop_t.set_xlabel('Simulation time [y]',size=self.font+2)
        axprop_t.set_ylim(0., int(self.true_pop.max())+1)
        axprop_t.set_xlim(0., self.true_timeCarb.max())
        ttl = axprop_t.title
        ttl.set_position([.5, 1.05])
        axprop_t.set_title('Evolution of assemblage populations with time',size=self.font+3)



        for i in range(samples - 1):
            print '\nSample: ', i
            start = time.time()

            p_sed1=[0.0009, 0.0015, 0.0023]
            p_sed2=[0.0015, 0.0017, 0.0024]
            p_sed3=[0.0016, 0.0028, 0.0027]
            p_sed4=[0.0017, 0.0031, 0.0043]
            p_flow1=[0.055, 0.008 ,0.]
            p_flow2=[0.082, 0.051, 0.]
            p_flow3=[0.259, 0.172, 0.058]
            p_flow4=[0.288, 0.185, 0.066]
            p_ax = self.true_ax
            p_ay = self.true_ay
            p_m = self.true_m
            
            p_ay = self.proposalJump(cm_ay, self.min_a, self.max_a, self.step_a)
            p_m = self.proposalJump(m, self.min_m, self.max_m, self.step_m)
            
            # a=0
            # while a < 10:
            #     p_flow3[self.assemblage-1]= self.proposalJump(flow3[self.assemblage-1],flow2[self.assemblage-1],flow4[self.assemblage-1], self.step_flow)
            #     p_flow4[self.assemblage-1]= self.proposalJump(flow4[self.assemblage-1],flow3[self.assemblage-1],self.flowlim[1], self.step_flow)
            #     if ((p_flow1[self.assemblage-1] < p_flow2[self.assemblage-1]) and (p_flow2[self.assemblage-1] < p_flow3[self.assemblage-1])) and (p_flow3[self.assemblage-1] < p_flow4[self.assemblage-1]):
            #         a = 10
            #         break
            #     elif (((p_flow1[self.assemblage-1] > p_flow2[self.assemblage-1]) or (p_flow2[self.assemblage-1] > p_flow3[self.assemblage-1])) or (p_flow3[self.assemblage-1] > p_flow4[self.assemblage-1])) and (a==10):
            #         p_flow1[self.assemblage-1] = flow1[self.assemblage-1]
            #         p_flow2[self.assemblage-1] = flow2[self.assemblage-1]
            #         p_flow3[self.assemblage-1] = flow3[self.assemblage-1]
            #         p_flow4[self.assemblage-1] = flow4[self.assemblage-1]
            #     else:
            #         a += 1


            # p_sed1 = np.zeros(3)
            # p_sed2 = np.zeros(3)
            # p_sed3 = np.zeros(3)
            # p_sed4 = np.zeros(3)
            # for c in range(communities):
            #     a = 0
            #     while a < 10:
            #         p_sed1[c] = self.proposalJump(sed1[c], self.sedlim[0], self.sedlim[1], self.step_sed)
            #         p_sed4[c] = self.proposalJump(sed4[c], p_sed1[c], self.sedlim[1], self.step_sed)
            #         p_sed2[c] = self.proposalJump(sed2[c], p_sed1[c], p_sed4[c], self.step_sed)
            #         p_sed3[c] = self.proposalJump(sed3[c], p_sed2[c], p_sed4[c], self.step_sed)
            #         if ((p_sed1[c] < p_sed2[c]) and (p_sed2[c] < p_sed3[c])) and (p_sed3[c] < p_sed4[c]):
            #             a = 10
            #             break
            #         elif (((p_sed1[c] > p_sed2[c]) or (p_sed2[c] > p_sed3[c])) or (p_sed3[c] > p_sed4[c])) and (a==10):
            #             p_sed1[c] = sed1[c]
            #             p_sed2[c] = sed2[c]
            #             p_sed3[c] = sed3[c]
            #             p_sed4[c] = sed4[c]
            #         else:
            #             a += 1
            
            # p_flow1 = np.zeros(3)
            # p_flow2 = np.zeros(3)
            # p_flow3 = np.zeros(3)
            # p_flow4 = np.zeros(3)
            # for d in range(communities):
            #     a = 0
            #     while a < 10:
            #         p_flow1[d] = self.proposalJump(flow1[d], self.flowlim[0], self.flowlim[1], self.step_flow)
            #         p_flow4[d] = self.proposalJump(flow4[d], p_flow1[d], self.flowlim[1], self.step_flow)
            #         p_flow2[d] = self.proposalJump(flow2[d], p_flow1[d], p_flow4[d], self.step_flow)
            #         p_flow3[d] = self.proposalJump(flow3[d], p_flow2[d], p_flow4[d], self.step_flow)
            #         if ((p_flow1[d] < p_flow2[d]) and (p_flow2[d] < p_flow3[d])) and (p_flow3[d] < p_flow4[d]):
            #             a = 10
            #             break
            #         elif (((p_flow1[d] > p_flow2[d]) or (p_flow2[d] > p_flow3[d])) or (p_flow3[d] > p_flow4[d])) and (a==10):
            #             p_flow1[d] = flow1[d]
            #             p_flow2[d] = flow2[d]
            #             p_flow3[d] = flow3[d]
            #             p_flow4[d] = flow4[d]
            #         else:
            #             a += 1

            # if self.sedsim == True:
            #     tmat = np.concatenate((sed1,sed2,sed3,sed4)).reshape(4,communities)
            #     tmatrix = tmat.T
            #     t2matrix = np.zeros((tmatrix.shape[0], tmatrix.shape[1]))
            #     for x in range(communities):
            #         for s in range(tmatrix.shape[1]):
            #             t2matrix[x,s] = self.proposalJump(tmatrix[x,s], self.sedlim[0], self.sedlim[1], self.step_sed)
            #     # reorder each row , then transpose back as sed1, etc.
            #     tmp = np.zeros((communities,4))
            #     for x in range(t2matrix.shape[0]):
            #         a = np.sort(t2matrix[x,:])
            #         tmp[x,:] = a
            #     tmat = tmp.T
            #     p_sed1 = tmat[0,:]
            #     p_sed2 = tmat[1,:]
            #     p_sed3 = tmat[2,:]
            #     p_sed4 = tmat[3,:]
                
            # if self.flowsim == True:
            #     tmat = np.concatenate((flow1,flow2,flow3,flow4)).reshape(4,communities)
            #     tmatrix = tmat.T
            #     t2matrix = np.zeros((tmatrix.shape[0], tmatrix.shape[1]))
            #     for x in range(communities):#-3):
            #         for s in range(tmatrix.shape[1]):
            #             t2matrix[x,s] = self.proposalJump(tmatrix[x,s], self.flowlim[0], self.flowlim[1], self.step_flow)
            #     # reorder each row , then transpose back as flow1, etc.
            #     tmp = np.zeros((communities,4))
            #     for x in range(t2matrix.shape[0]):
            #         a = np.sort(t2matrix[x,:])
            #         tmp[x,:] = a
            #     tmat = tmp.T
            #     p_flow1 = tmat[0,:]
            #     p_flow2 = tmat[1,:]
            #     p_flow3 = tmat[2,:]
            #     p_flow4 = tmat[3,:]

            # p_ax = self.proposalJump(cm_ax, self.max_a, 0, self.step_a)
            # p_ay = self.proposalJump(cm_ay, self.max_a, 0, self.step_a)
            # p_m = self.proposalJump(m, 0, self.max_m, self.step_m)
            
            v_proposal = []
            if (self.sedsim == True) and (self.flowsim == False):
                v_proposal = np.concatenate((p_sed1,p_sed2,p_sed3,p_sed4))
            elif (self.flowsim == True) and (self.sedsim == False):
                v_proposal = np.concatenate((p_flow1,p_flow2,p_flow3,p_flow4))
            elif (self.sedsim == True) and (self.flowsim == True):
                v_proposal = np.concatenate((p_sed1,p_sed2,p_sed3,p_sed4,p_flow1,p_flow2,p_flow3,p_flow4))
            v_proposal = np.append(v_proposal,(p_ax,p_ay,p_m))

            [likelihood_proposal, sim_pred_t, sim_pred_d, diff] = self.likelihoodWithPropn(reef, core_data, v_proposal)
            diff_likelihood = likelihood_proposal - likelihood # to divide probability, must subtract
            print 'likelihood_proposal:', likelihood_proposal, 'diff_likelihood',diff_likelihood
            mh_prob = min(1, math.exp(diff_likelihood))
            u = random.uniform(0, 1)
            print 'u', u, 'and mh_probability', mh_prob
            
            if u < mh_prob: # accept
                #   Update position
                print i, ' is accepted sample'
                naccept += 1
                count_list.append(i)
                likelihood = likelihood_proposal
                m = p_m
                cm_ax = p_ax
                cm_ay = p_ay
                if self.sedsim == True:
                    sed1 = p_sed1
                    sed2 = p_sed2
                    sed3 = p_sed3
                    sed4 = p_sed4
                if self.flowsim == True:
                    flow1 = p_flow1
                    flow2 = p_flow2
                    flow3 = p_flow3
                    flow4 = p_flow4
                # self.saveCore(reef,naccept)

                print  'likelihood:',likelihood, ' and difference score:', diff, 'accepted'

                if self.sedsim == True:
                    pos_sed1[i + 1,] = sed1
                    pos_sed2[i + 1,] = sed2
                    pos_sed3[i + 1,] = sed3
                    pos_sed4[i + 1,] = sed4
                if self.flowsim == True:
                    pos_flow1[i + 1,] = flow1
                    pos_flow2[i + 1,] = flow2
                    pos_flow3[i + 1,] = flow3
                    pos_flow4[i + 1,] = flow4
                pos_ax[i + 1] = cm_ax
                pos_ay[i + 1] = cm_ay
                pos_m[i + 1] = m
                pos_v[i + 1,] = v_proposal
                pos_diff[i + 1,] = diff
                pos_likl[i + 1,] = likelihood
                pos_samples_t[i + 1,] = self.convertCoreFormat(sim_pred_t)
                core_vec = self.convertCoreFormat(sim_pred_d)
                pos_samples_d[i + 1,] = core_vec
                
                axprop_d.plot(core_vec,data_depths, label=None)  
                timecarb,pop,names = reef.plot.getTimePlotParameters()
                for x in range(len(pop)):
                    axprop_t.plot(timecarb[::-1], pop[x,:],linewidth=self.width)
        
                # Uncomment if you want to see posterior as the simulations runs
                # saveParameters.saveParameters(self.filename, self.sedsim, self.flowsim, i+1, 
                #     pos_m[i+1], pos_ax[i+1], pos_ay[i+1], 
                #     pos_sed1[i+1,], pos_sed2[i+1,], pos_sed3[i+1,], pos_sed4[i+1,],
                #     pos_flow1[i+1,], pos_flow2[i+1,], pos_flow3[i+1,], pos_flow4[i+1,],
                #     pos_diff[i+1],pos_likl[i+1], pos_samples[i+1,],pos_v[i+1,])

            else: #reject
                pos_v[i + 1,] = pos_v[i,]
                pos_samples_t[i + 1,] = pos_samples_t[i,]
                pos_samples_d[i + 1,] = pos_samples_d[i,]
                pos_diff[i + 1,] = pos_diff[i,]
                pos_likl[i + 1,] = pos_likl[i,]
                print 'REJECTED\nLikelihood:',likelihood,'and difference score:', pos_diff[i,]
                #   Copy past accepted state
                if self.sedsim == True:
                    pos_sed1[i + 1,] = pos_sed1[i,]
                    pos_sed2[i + 1,] = pos_sed2[i,]
                    pos_sed3[i + 1,] = pos_sed3[i,]
                    pos_sed4[i + 1,] = pos_sed4[i,]
                if self.flowsim == True:
                    pos_flow1[i + 1,] = pos_flow1[i,]
                    pos_flow2[i + 1,] = pos_flow2[i,]
                    pos_flow3[i + 1,] = pos_flow3[i,]
                    pos_flow4[i + 1,] = pos_flow4[i,]
                pos_ax[i+1] = pos_ax[i]
                pos_ay[i+1] = pos_ay[i]
                pos_m[i+1] = pos_m[i]
                # Uncomment if you want to see posterior as the simulations runs
                # saveParameters.saveParameters(self.filename, self.sedsim, self.flowsim, i+1, 
                #     pos_m[i+1], pos_ax[i+1], pos_ay[i+1], 
                #     pos_sed1[i+1,], pos_sed2[i+1,], pos_sed3[i+1,], pos_sed4[i+1,],
                #     pos_flow1[i+1,], pos_flow2[i+1,], pos_flow3[i+1,], pos_flow4[i+1,],
                #     pos_diff[i+1],pos_likl[i+1], pos_samples[i+1,],pos_v[i+1,])
                print i, 'rejected and retained'

            end = time.time()
            total_time = end-start
            print 'Time elapsed:', total_time

            if i==samples - 2:
                self.saveCore(reef, i+1)

        accepted_count =  len(count_list)   
        print accepted_count, ' number accepted'
        print len(count_list) / (samples * 0.01), '% was accepted'
        accept_ratio = accepted_count / (samples * 1.0) * 100

        # lgd = axprop_d.legend(frameon=False, prop={'size':self.font+1},bbox_to_anchor = (1.,0.1))
        finalfig_d.savefig('%s/proposals_d.png'% (self.filename),bbox_inches='tight',dpi=200,transparent=False)
        plt.close()

        finalfig_t.savefig('%s/proposals_t.png'% (self.filename),bbox_inches='tight',dpi=200,transparent=False)
        plt.close()

        return (pos_v, pos_diff, pos_likl, pos_samples_t, pos_samples_d, pos_sed1,pos_sed2,pos_sed3,pos_sed4,
        	pos_flow1,pos_flow2,pos_flow3,pos_flow4, pos_ax,pos_ay,pos_m, 
        	data_depths, accept_ratio, accepted_count, data_vec_t, x_tick_labels, x_tick_values)

#####################################################################

def main():
    
    #    Set all input parameters    #
    random.seed(time.time())
    samples= 10000 #input('Enter number of samples: ')
    # description = raw_input('Enter description: ')
    description = 'time-based likelihood function, self.likelihoodWithPropn, fix all, free m and a_s'
    assemblage = 2
    xmlinput = 'input_synth.xml'
    data_depths, data_vec_d = np.genfromtxt('data/synthdata_d_vec.txt', usecols=(0,1), unpack=True)
    synth_data = 'data/synthdata_t_prop_08_2.txt'
    core_data = np.loadtxt(synth_data, usecols=(1,2,3,4))  
    synth_vec = 'data/synthdata_t_vec_08_2.txt'
    data_times, data_vec_t = np.genfromtxt(synth_vec, usecols=(0, 1), unpack = True)
    true_pop = np.loadtxt('data/true_pop.txt', usecols=(0,1,2)) 
    true_pop = true_pop.T
    true_timeCarb = np.loadtxt('data/true_timecarb.txt', usecols=0)
    nCommunities = 3
    simtime = 8500
    """ Option to visualise the initial parameters [0] and visualise the cores [1] for each iteration of MCMC. 
    Mainly useful for runModel.py and slows down simulation in MCMC sampler. """
    vis = [False, False]
    sedsim, flowsim = True, True
    sedlim = [0., 0.005]
    flowlim = [0.,0.3]
    run_nb = 0

    min_a = 0
    max_a = -0.15
    min_m = 0
    max_m = 0.15
    true_m = 0.08
    true_ax = -0.01
    true_ay = -0.03

    step_sed = 0.01 * abs(sedlim[0]-sedlim[1])
    step_flow = 0.01 * abs(flowlim[0]-flowlim[1])
    step_m = 0.01 * abs(min_m-max_m)
    step_a = 0.01 * abs(min_a-max_a)

    path_name = 'results_multinom_t_freeglv'
    while os.path.exists('%s_%s' % (path_name, run_nb)):
        run_nb+=1
    if not os.path.exists('%s_%s' % (path_name, run_nb)):
        os.makedirs('%s_%s' % (path_name, run_nb))
    filename = ('%s_%s' % (path_name, run_nb))

    #    Save File of Experiment Description   #
    if not os.path.isfile(('%s/description.txt' % (filename))):
        with file(('%s/description.txt' % (filename)),'w') as outfile:
            outfile.write('Script: {0}'.format(os.path.basename(__file__)))
            outfile.write('\nDescription: {0}'.format(description))
            outfile.write('\nSimulation time: {0} yrs'.format(simtime))
            outfile.write('\nNo. samples: {0}'.format(samples))
            outfile.write('\nXML input: {0}'.format(xmlinput))
            outfile.write('\nData file: {0}'.format(synth_vec))

    mcmc = MCMC(filename, xmlinput, simtime, samples, nCommunities, sedsim, sedlim, flowsim, flowlim, vis,
        data_depths, data_vec_d, data_times, data_vec_t, core_data, true_pop, true_timeCarb,
        min_m, max_m, true_m, step_m, min_a, max_a, true_ax, true_ay ,step_a, step_sed, step_flow, assemblage)


    [pos_v, pos_diff, pos_likl, pos_samples_t, pos_samples_d, pos_sed1,pos_sed2,pos_sed3,pos_sed4,
    pos_flow1,pos_flow2,pos_flow3,pos_flow4, pos_ax,pos_ay,pos_m, data_depths,
    accept_ratio, accepted_count, data_vec_t,x_tick_labels, x_tick_values] = mcmc.sampler()
    print 'Successfully sampled'
    
    burnin = 0.1 * samples  # use post burn in samples
    pos_v = pos_v[int(burnin):, ]
    pos_sed1 = pos_sed1[int(burnin):, ]
    pos_sed2 = pos_sed2[int(burnin):, ]
    pos_sed3 = pos_sed3[int(burnin):, ]
    pos_sed4 = pos_sed4[int(burnin):, ]
    pos_flow1 = pos_flow1[int(burnin):, ]
    pos_flow2 = pos_flow2[int(burnin):, ]
    pos_flow3 = pos_flow3[int(burnin):, ]
    pos_flow4 = pos_flow4[int(burnin):, ]
    pos_ax = pos_ax[int(burnin):]
    pos_ay = pos_ay[int(burnin):]
    pos_m = pos_m[int(burnin):]
    pos_likl = pos_likl[int(burnin):]
    likl_mu = np.mean(pos_likl)
    likl_std = np.std(pos_likl)
    likl_mode, likl_count = stats.mode(pos_likl)
    pos_diff = pos_diff[int(burnin):]
    diff_mu = np.mean(pos_diff)
    diff_std = np.std(pos_diff)
    diff_mode, diff_count = stats.mode(pos_diff)
    print 'Mean diff:',diff_mu, ', standard deviation:', diff_std

    with file(('%s/out_results.txt' % (filename)),'w') as outres:
        outres.write('DIFFERENCE\n')
        outres.write('\tMean: {0}, st. dev: {1}, mode: {2}\n'.format(diff_mu, diff_std,diff_mode))
        outres.write('LIKELIHOOD\n')
        outres.write('\tMean: {0}, st. dev: {1}, mode: {2}\n'.format(likl_mu, likl_std, likl_mode))
        outres.write('Accept ratio: {0} %\nSamples accepted : {1} out of {2}'.format(accept_ratio, accepted_count, samples))

    if not os.path.isfile(('%s/pos_burnin_GLVE.csv' % (filename))):
        np.savetxt("%s/pos_burnin_GLVE.csv" % (filename), np.c_[pos_m,pos_ax,pos_ay], delimiter=',')

    if not os.path.isfile(('%s/pos_burnin_sed.csv' % (filename))):
        np.savetxt("%s/pos_burnin_sed.csv" % (filename), np.c_[pos_sed1,pos_sed2,pos_sed3, pos_sed4], delimiter=',')

    if not os.path.isfile(('%s/pos_burnin_flow.csv' % (filename))):
        np.savetxt("%s/pos_burnin_flow.csv" % (filename), np.c_[pos_flow1,pos_flow2,pos_flow3, pos_flow4], delimiter=',')

    if not os.path.isfile(('%s/pos_burnin_likl.csv' % (filename))):
        np.savetxt("%s/pos_burnin_likl.csv" % (filename), pos_likl, delimiter=',')

    if not os.path.isfile(('%s/pos_burnin_proposal.csv' % (filename))):
        np.savetxt("%s/pos_burnin_proposal.csv" % (filename), pos_v, delimiter=',')
    sample_range = np.arange(burnin+1,samples+1, 1)
    plotResults.plotPosCore(pos_samples_d, data_depths, data_vec_d, x_tick_labels,x_tick_values, mcmc.font, mcmc.width, filename)
    plotResults.boxPlots(nCommunities, pos_v, sedsim, flowsim, mcmc.font,mcmc.width,filename)    
    plotResults.plotLiklAndDiff(pos_likl, pos_diff, sample_range, mcmc.font, filename)
    plotResults.plotParameters(mcmc.filename, sample_range, mcmc.sedsim, mcmc.flowsim, mcmc.communities, 
        pos_m, pos_ax, pos_ay, mcmc.true_m, mcmc.true_ax, mcmc.true_ay, 
        pos_sed1, pos_sed2, pos_sed3, pos_sed4, mcmc.true_sed,
        pos_flow1, pos_flow2, pos_flow3, pos_flow4, mcmc.true_flow)
    print 'Finished simulations'

if __name__ == "__main__": main()
