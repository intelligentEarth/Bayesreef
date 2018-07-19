#!/usr/bin/env python
#Title           :multinom_mcmc_t.py
#Description     :This is the script for BayesReef: an MCMC random walk method applied to pyReefCore.
#Author          :Jodie Pall and Danial Azam 
#Last edit       :19/07/2018
#Version         :1.0
#Usage           :python multinom_mcmc_t.py
#Notes           :This script uses the time-structure of a core simulation (i.e. coralgal assemblage at each time interval of the simulation)
#                 from pyReef-Core as the observed data. The observed data is used by the MCMC algorithm to find the posterior probability distribution of free parameters 
#                 and model predictions. The amount of free parameters in this script is 27. 
#                 This includes 3 population dynamics parameters and 24 environmental threshold parameters.
# 
#                 Population dynamics parameters: ax, ay, m
#                 Environmental threshold parameters: flow1[,:n], flow2[,:n], flow3[,:n], flow4[,:n],
#                                                     sed1[,:n], sed2[,:n], sed3[,:n], sed4[,:n] 
#                                                     For number of assemblages (n).
# 
#References:      The MCMC method in this script is adapted from:
#                 [Chandra_ICONIP2017] R. Chandra, L. Azizi, S. Cripps, 'Bayesian neural learning via Langevin dynamicsfor chaotic time series prediction', ICONIP 2017.
#Python_version  :2.7.12
#==============================================================================

# Import the modules needed to run the script.
import os
import math
import time
import random
import csv
from numpy import inf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from pyReefCore.model import Model
from pyReefCore import (plotResults, saveParameters)
from cycler import cycler
from scipy import stats 
from matplotlib.cm import terrain, plasma, Set2

cmap=plt.cm.Set2
c = cycler('color', cmap(np.linspace(0,1,8)) )
plt.rcParams["axes.prop_cycle"] = c

class MCMC():
    def __init__(self, filename, xmlinput, simtime, samples, communities, sedsim, sedlim, flowsim, flowlim, vis,
        gt_depths, gt_vec_d, gt_timelay, gt_vec_t, gt_prop_t,
        min_m, max_m, true_m, step_m, min_a, max_a, true_ax, true_ay,step_a, step_sed, step_flow, assemblage):

        self.font = 10
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
        self.gt_vec_d = gt_vec_d
        self.gt_timelay = gt_timelay
        self.gt_vec_t = gt_vec_t
        self.gt_prop_t = gt_prop_t
        
        self.true_m = true_m
        self.true_ax = true_ax
        self.true_ay = true_ay
        self.true_sed = []
        self.true_flow = []
        self.min_m = min_m
        self.max_m = max_m
        self.min_a = min_a
        self.max_a = max_a
        
        self.step_m =step_m #0.002 # <1%
        self.step_a =step_a #0.002 # <1%
        self.step_sed = step_sed #0.0001 # 2%
        self.step_flow = step_flow #0.0015 # 0.05%        
        
        self.assemblage = assemblage
        
    def runModel(self, reef, input_vector):
        reef.convertVector(self.communities, input_vector, self.sedsim, self.flowsim) #model.py
        self.true_sed, self.true_flow = reef.load_xml(self.input, self.sedsim, self.flowsim)
        # if self.vis[0] == True:
            # reef.core.initialSetting(size=(8,2.5), size2=(8,3.5)) # View initial parameters
        reef.run_to_time(self.simtime,showtime=100.)
        # if self.vis[1] == True:
        #     reef.plot.drawCore(lwidth = 3, colsed=self.colors, coltime = self.colors2, size=(9,8), font=8, dpi=300)
        sim_output_t, sim_timelay = reef.plot.convertTimeStructure() #modelPlot.py
        sim_output_d = reef.plot.convertDepthStructure(self.communities, self.gt_depths)
        return sim_output_t, sim_output_d,sim_timelay

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
        cps = np.zeros(n)
        cps[0] = timelay[0] # (171,)
        ca_props = np.zeros((n, prop_t.shape[1]))# (171,5)
        
        for i in range(1,n):
            if vec_t[i] != vec_t[i-1]:
                cps[s] = timelay[i-1]
                ca_props[s-1] = prop_t[i-1,:]
                s += 1
            if i == n-1:
                cps[s] = timelay[i]
                ca_props[s-1] = prop_t[i,:]
        S = s
        cps = np.trim_zeros(cps, 'b') # append a zero on the end afterwards
        ca_props = ca_props[0:S,:]
        return S, cps, ca_props

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
        sim_prop_t, sim_prop_d, sim_timelay = self.runModel(reef, input_v)
        sim_vec_d = self.convertCoreFormat(sim_prop_d.T)
        sim_vec_t = self.convertCoreFormat(sim_prop_t)
        sim_prop_t5 = self.noGrowthColumn(sim_prop_t)

        # Counting segments, recording location of cutpoints and associated cagal assemblage proportions
        print 'S_star:',S_star, '\ncpts_star:',cpts_star,'\nca_props_star props:', ca_props_star
        S, cpts, ca_props = self.modelOutputParameters(sim_prop_t5,sim_vec_t,sim_timelay)
        print 's:',S, '\ncpts:',cpts,'\nca props:', ca_props
        # First reject if number of segments in sim != S_star
        if S != S_star:
            likelihood=0
            diff = 100
            return [likelihood, diff, sim_prop_t5, sim_prop_d.T, sim_vec_t, sim_vec_d]
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
        like_all_cpts_star = np.prod(likl_cpts_star)
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
        return [total_likelihood, diff, sim_prop_t, sim_prop_d.T, sim_vec_t, sim_vec_d]

    def likelihoodWithProps(self, reef, gt_prop_t, input_v):
        sim_prop_t, sim_prop_d, sim_timelay = self.runModel(reef, input_v)
        sim_prop_t5 = self.noGrowthColumn(sim_prop_t)
        intervals = sim_prop_t5.shape[0]
        # # Uncomment if noisy synthetic data is required.
        # self.NoiseToData(intervals,sim_prop_t5)
        log_core = np.log(sim_prop_t5+0.0001)
        log_core[log_core == -inf] = 0
        z = log_core * gt_prop_t
        likelihood = np.sum(z)
        diff = self.diffScore(sim_prop_t5,gt_prop_t, intervals)
        sim_vec_t = self.convertCoreFormat(sim_prop_t5)
        sim_vec_d = self.convertCoreFormat(sim_prop_d.T)
        # rmse = self.rmse(sim_prop_t5, gt_prop_t)
        return [likelihood, diff, sim_prop_t5, sim_prop_d.T, sim_vec_t, sim_vec_d]
           
    def likelihoodWithDominance(self, reef, gt_prop_t, input_v):
        sim_data_t, sim_data_d, sim_timelay = self.runModel(reef, input_v)
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
        # rmse = self.rmse(sim_data_t5, gt_prop_t)
        sim_vec_t = self.convertCoreFormat(sim_data_t5)
        sim_vec_d = self.convertCoreFormat(sim_data_d.T)
        return [likelihood, diff, sim_data_t5, sim_data_d.T, sim_vec_t, sim_vec_d]

    # def noisyDataLikelihood(self, reef, gt_prop_t, input_v):
    #         pred_core = self.run_Model(reef, input_v)
    #         pred_core = pred_core.T
    #         pred_core_w_noise = np.zeros((pred_core.shape[0], pred_core.shape[1]))
    #         intervals = pred_core.shape[0]
    #         for n in range(intervals):
    #            pred_core_w_noise[n,:] = np.random.multinomial(1,pred_core[n],size=1)
    #         pred_core_w_noise = pred_core_w_noise/1
    #         z = np.zeros((intervals,self.communities+1))  
    #         z = pred_core_w_noise * gt_prop_t
    #         loss = np.log(z)
    #         loss[loss == -inf] = 0
    #         loss = np.sum(loss)
    #         diff = self.diff_score(pred_core_w_noise,gt_prop_t, intervals)
    #         loss = np.exp(loss)
    #         return [loss, pred_core_w_noise, diff]

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
        gt_prop_t = self.gt_prop_t
        gt_vec_t = self.gt_vec_t
        gt_timelay = self.gt_timelay
        gt_vec_d = self.gt_vec_d
        gt_depths = self.gt_depths
        communities = self.communities
        

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
        pos_samples_d = np.zeros((samples, gt_vec_d.shape[0]))
        pos_samples_t = np.zeros((samples, gt_prop_t.shape[0]))

        #      INITIAL PREDICTION       #
        sed1 = np.zeros(communities)
        sed2 = np.zeros(communities)
        sed3 = np.zeros(communities)
        sed4 = np.zeros(communities)
        if self.sedsim == True:
            for s in range(communities):
                sed1[s] = pos_sed1[0,s] = np.random.uniform(self.sedlim[0],self.sedlim[1])
                sed2[s] = pos_sed2[0,s] = np.random.uniform(sed1[s],self.sedlim[1])
                sed3[s] = pos_sed3[0,s] = np.random.uniform(sed2[s],self.sedlim[1])
                sed4[s] = pos_sed4[0,s] = np.random.uniform(sed3[s],self.sedlim[1])

        flow1 = np.zeros(communities)
        flow2 = np.zeros(communities)
        flow3 = np.zeros(communities)
        flow4 = np.zeros(communities)
        if self.flowsim == True:
            for s in range(communities):
                #     relaxed constraints 
                flow1[s] = pos_flow1[0,s] = np.random.uniform(self.flowlim[0],self.flowlim[1])
                flow2[s] = pos_flow2[0,s] = np.random.uniform(flow1[s],self.flowlim[1])
                flow3[s] = pos_flow3[0,s] = np.random.uniform(flow2[s],self.flowlim[1])
                flow4[s] = pos_flow4[0,s] = np.random.uniform(flow3[s],self.flowlim[1])
        
        cm_ax = pos_ax[0] = np.random.uniform(self.min_a,self.max_a)
        cm_ay = pos_ay[0] = np.random.uniform(self.min_a,self.max_a)
        m = pos_m[0] = np.random.uniform(self.min_m, self.max_m)

        pr_flow2 = flowlim[1]-flow1[self.assemblage-1]
        pr_flow3 = flowlim[1]-flow2[self.assemblage-1]
        pr_flow4 = flowlim[1]-flow3[self.assemblage-1]
        prs_flow = np.array([pr_flow2,pr_flow3,pr_flow4])
        c_pr_flow = np.prod(prs_flow)
        pr_sed2 = sedlim[1]-sed1[self.assemblage-1]
        pr_sed3 = sedlim[1]-sed2[self.assemblage-1]
        pr_sed4 = sedlim[1]-sed3[self.assemblage-1]
        prs_sed = np.array([pr_sed2,pr_sed3,pr_sed4])
        c_pr_sed = np.prod(prs_sed)

        if (self.sedsim == True) and (self.flowsim == True):
            v_proposal = np.concatenate((sed1,sed2,sed3,sed4,flow1,flow2,flow3,flow4))
        elif (self.sedsim == True) and (self.flowsim == False):
            v_proposal = np.concatenate((sed1,sed2,sed3,sed4))
        elif (self.sedsim == False) and (self.flowsim == True):
            v_proposal = np.concatenate((flow1,flow2,flow3,flow4))
        v_proposal = np.append(v_proposal,(cm_ax,cm_ay,m))
        pos_v = np.zeros((samples, v_proposal.size))
        print v_proposal

        # Declare pyReef-Core and initialize
        reef = Model()
        S_star, cps_star, ca_props_star = self.modelOutputParameters(gt_prop_t,gt_vec_t,gt_timelay)
        [likelihood, diff, sim_pred_t, sim_pred_d, sim_vec_t, sim_vec_d] = self.likelihoodWithDependence(reef, v_proposal, S_star, cps_star, ca_props_star)
        # [likelihood, diff, sim_pred_t, sim_pred_d, sim_vec_t, sim_vec_d] = self.likelihoodWithProps(reef, self.gt_prop_t, v_proposal)

        print '\tInitial likelihood:', likelihood
        pos_diff = np.full(samples,diff)
        pos_likl = np.full(samples, likelihood)
        pos_samples_t[0,:] = sim_vec_t
        pos_samples_d[0,:] = sim_vec_d
        self.saveCore(reef, 'initial')

        naccept = 0
        count_list = []
        count_list.append(0)
        # Uncomment if you want to see posterior as the simulations runs
        saveParameters.saveParameters(self.filename, self.sedsim, self.flowsim, naccept, 
            pos_m[0], pos_ax[0], pos_ay[0], 
            pos_sed1[0,], pos_sed2[0,], pos_sed3[0,], pos_sed4[0,],
            pos_flow1[0,], pos_flow2[0,], pos_flow3[0,], pos_flow4[0,],
            pos_diff[0],pos_likl[0], pos_samples_t[0,],pos_v[0,])
        
        print 'Begin sampling using MCMC random walk'
        ##########################################
        # PLOT INITIAL PREDICTIONS AND PROPOSALS #
        ##########################################
        # Initial predictions

        x_tick_labels = ['Shallow', 'Mod-deep', 'Deep', 'Sediment','No growth']
        x_tick_values = [1,2,3,4,5]
        plotResults.plotInitialPrediction(self.filename, gt_vec_d, gt_vec_t,sim_vec_d,sim_vec_t,gt_depths,gt_timelay,
            x_tick_labels, x_tick_values)
        
        # Accumulating figure of all proposals - Depth and time

        finalfig = plt.figure(figsize=(4,4))
        suptitle = finalfig.suptitle('Accepted proposals')
        axprop_d = finalfig.add_subplot(121)
        axprop_d.set_facecolor('#f2f2f3')
        axprop_d.plot(gt_vec_d, gt_depths, label='Ground truth', color='k',linewidth=self.width-0.5)
        axprop_d.plot(sim_vec_d, gt_depths, linewidth=self.width-0.5)
        plt.xticks(x_tick_values, x_tick_labels,rotation=70)
        axprop_d.set_ylabel("Depth [m]")
        axprop_d.set_ylim([0,np.amax(gt_depths)])
        axprop_d.set_ylim(axprop_d.get_ylim()[::-1])
        
        axprop_t = finalfig.add_subplot(122)
        axprop_t.set_facecolor('#f2f2f3')
        axprop_t.plot(gt_vec_t, gt_timelay, label='Ground truth', color='k', linewidth=self.width-0.5)
        axprop_t.plot(sim_vec_t,gt_timelay, linewidth=self.width-0.5)
        plt.xticks(x_tick_values, x_tick_labels,rotation=70)
        axprop_t.set_ylabel('Simulation time [yrs]')
        axprop_t.set_ylim([0,np.amax(gt_timelay)])
        axprop_t.set_ylim(axprop_t.get_ylim())#[::-1])

        for i in range(samples - 1):
            print '\nSample: ', i
            start = time.time()

            if self.sedsim == True:
                tmat = np.concatenate((sed1,sed2,sed3,sed4)).reshape(4,communities)
                tmatrix = tmat.T
                t2matrix = np.zeros((tmatrix.shape[0], tmatrix.shape[1]))
                for x in range(communities):
                    for s in range(tmatrix.shape[1]):
                        t2matrix[x,s] = self.proposalJump(tmatrix[x,s], self.sedlim[0], self.sedlim[1], self.step_sed)
                # reorder each row , then transpose back as sed1, etc.
                tmp = np.zeros((communities,4))
                for x in range(t2matrix.shape[0]):
                    a = np.sort(t2matrix[x,:])
                    tmp[x,:] = a
                tmat = tmp.T
                p_sed1 = tmat[0,:]
                p_sed2 = tmat[1,:]
                p_sed3 = tmat[2,:]
                p_sed4 = tmat[3,:]
                
            if self.flowsim == True:
                tmat = np.concatenate((flow1,flow2,flow3,flow4)).reshape(4,communities)
                tmatrix = tmat.T
                t2matrix = np.zeros((tmatrix.shape[0], tmatrix.shape[1]))
                for x in range(communities):#-3):
                    for s in range(tmatrix.shape[1]):
                        t2matrix[x,s] = self.proposalJump(tmatrix[x,s], self.flowlim[0], self.flowlim[1], self.step_flow)
                # reorder each row , then transpose back as flow1, etc.
                tmp = np.zeros((communities,4))
                for x in range(t2matrix.shape[0]):
                    a = np.sort(t2matrix[x,:])
                    tmp[x,:] = a
                tmat = tmp.T
                p_flow1 = tmat[0,:]
                p_flow2 = tmat[1,:]
                p_flow3 = tmat[2,:]
                p_flow4 = tmat[3,:]

            p_ax = self.proposalJump(cm_ax, self.max_a, 0, self.step_a)
            p_ay = self.proposalJump(cm_ay, self.max_a, 0, self.step_a)
            p_m = self.proposalJump(m, 0, self.max_m, self.step_m)
            
            v_proposal = []
            if (self.sedsim == True) and (self.flowsim == False):
                v_proposal = np.concatenate((p_sed1,p_sed2,p_sed3,p_sed4))
            elif (self.flowsim == True) and (self.sedsim == False):
                v_proposal = np.concatenate((p_flow1,p_flow2,p_flow3,p_flow4))
            elif (self.sedsim == True) and (self.flowsim == True):
                v_proposal = np.concatenate((p_sed1,p_sed2,p_sed3,p_sed4,p_flow1,p_flow2,p_flow3,p_flow4))
            v_proposal = np.append(v_proposal,(p_ax,p_ay,p_m))
            [likelihood_proposal, diff, sim_pred_t, sim_pred_d, sim_vec_t, sim_vec_d] = self.likelihoodWithDependence(reef, v_proposal, S_star, cps_star, ca_props_star)
            # [likelihood_proposal, diff, sim_pred_t, sim_pred_d, sim_vec_t, sim_vec_d] = self.likelihoodWithProps(reef, self.gt_prop_t, v_proposal)

            diff_likelihood = likelihood_proposal - likelihood # to divide probability, must subtract
            print 'likelihood_proposal:', likelihood_proposal, 'diff_likelihood',diff_likelihood
            
            p_pr_flow2 = flowlim[1] - p_flow1[self.assemblage-1]
            p_pr_flow3 = flowlim[1] - p_flow2[self.assemblage-1]
            p_pr_flow4 = flowlim[1] - p_flow3[self.assemblage-1]
            all_flow_pr = np.array([p_pr_flow2,p_pr_flow3,p_pr_flow4])
            p_pr_flow = np.prod(all_flow_pr)

            p_pr_sed2 = sedlim[1] - p_sed1[self.assemblage-1]
            p_pr_sed3 = sedlim[1] - p_sed2[self.assemblage-1]
            p_pr_sed4 = sedlim[1] - p_sed3[self.assemblage-1]
            all_sed_pr = np.array([p_pr_sed2,p_pr_sed3,p_pr_sed4])
            p_pr_sed = np.prod(all_sed_pr)

            log_pr_flow_p = np.log(p_pr_flow)
            log_pr_sed_p = np.log(p_pr_sed)
            log_pr_flow_c = np.log(c_pr_flow)
            log_pr_sed_c = np.log(c_pr_sed)
            log_pr_p = log_pr_flow_p+log_pr_sed_p
            log_pr_c = log_pr_flow_c+log_pr_sed_c
            log_pr_diff = log_pr_p - log_pr_c

            mh_prob = min(1, math.exp(diff_likelihood+log_pr_diff))
            u = random.uniform(0, 1)
            print 'u', u, 'and mh_probability', mh_prob
            
            if u < mh_prob: # accept
                #   Update position
                print i, ' is accepted sample'
                naccept += 1
                count_list.append(i)
                likelihood = likelihood_proposal
                c_pr_flow = p_pr_flow
                c_pr_sed = p_pr_sed
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
                pos_samples_t[i + 1,] = sim_vec_t
                pos_samples_d[i + 1,] = sim_vec_d
                
                axprop_d.plot(sim_vec_d,gt_depths, linewidth=self.width-0.5)
                axprop_t.plot(sim_vec_t,gt_timelay, linewidth=self.width-0.5)  
        
                # Uncomment if you want to see posterior as the simulations runs
                saveParameters.saveParameters(self.filename, self.sedsim, self.flowsim, i+1, 
                    pos_m[i+1], pos_ax[i+1], pos_ay[i+1], 
                    pos_sed1[i+1,], pos_sed2[i+1,], pos_sed3[i+1,], pos_sed4[i+1,],
                    pos_flow1[i+1,], pos_flow2[i+1,], pos_flow3[i+1,], pos_flow4[i+1,],
                    pos_diff[i+1],pos_likl[i+1], pos_samples_t[i+1,],pos_v[i+1,])

            else: #reject
                pos_v[i + 1,] = pos_v[i,]
                pos_samples_t[i + 1,] = pos_samples_t[i,]
                pos_samples_d[i + 1,] = pos_samples_d[i,]
                pos_diff[i + 1,] = pos_diff[i,]
                pos_likl[i + 1,] = pos_likl[i,]
                print 'REJECTED.\nLikelihood:',likelihood,'and difference score:', pos_diff[i,]
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
                saveParameters.saveParameters(self.filename, self.sedsim, self.flowsim, i+1, 
                    pos_m[i+1], pos_ax[i+1], pos_ay[i+1], 
                    pos_sed1[i+1,], pos_sed2[i+1,], pos_sed3[i+1,], pos_sed4[i+1,],
                    pos_flow1[i+1,], pos_flow2[i+1,], pos_flow3[i+1,], pos_flow4[i+1,],
                    pos_diff[i+1],pos_likl[i+1], pos_samples_t[i+1,],pos_v[i+1,])
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
        finalfig.tight_layout(pad=2.0)
        finalfig.savefig('%s/proposals.png'% (self.filename),bbox_inches='tight',dpi=200,transparent=False)
        plt.close()

        return (pos_v, pos_diff, pos_likl, pos_samples_t, pos_samples_d, pos_sed1,pos_sed2,pos_sed3,pos_sed4,
        	pos_flow1,pos_flow2,pos_flow3,pos_flow4, pos_ax,pos_ay,pos_m, 
        	gt_depths, accept_ratio, accepted_count, gt_vec_t, x_tick_labels, x_tick_values)

#####################################################################

def main():
    #    Set all input parameters    #
    random.seed(time.time())
    samples= 5 
    description = 'Time-based likelihood. self.likelihoodWithDependence'
    assemblage = 2
    xmlinput = 'input_synth.xml'
    gt_depths, gt_vec_d = np.genfromtxt('data/synthdata_d_vec_08.txt', usecols=(0,1), unpack=True)
    synth_data = 'data/synthdata_t_prop_08_1.txt'
    gt_prop_t = np.loadtxt(synth_data, usecols=(1,2,3,4,5)) 
    synth_vec = 'data/synthdata_t_vec_08_1.txt'
    gt_timelay, gt_vec_t = np.genfromtxt(synth_vec, usecols=(0, 1), unpack = True)
    gt_timelay = gt_timelay[::-1]

    nCommunities = 3
    simtime = 8500
    """ Option to visualise the initial parameters [0] and visualise the cores [1] for each iteration of MCMC. 
    Mainly useful for runModel.py and slows down simulation in MCMC sampler. """
    vis = [False, False]
    sedsim, flowsim = True, True
    sedlim = [0., 0.005]
    flowlim = [0.,0.3]
    run_nb = 0

    min_a = -0.15
    max_a = 0.
    min_m = 0.
    max_m = 0.15
    true_m = 0.08
    true_ax = -0.01
    true_ay = -0.03

    step_pcent = 0.01
    step_sed = step_pcent * abs(sedlim[0]-sedlim[1])
    step_flow = step_pcent * abs(flowlim[0]-flowlim[1])
    step_m = step_pcent * abs(min_m-max_m)
    step_a = step_pcent * abs(min_a-max_a)

    path_name = 'results-multinomial-t'
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
            outfile.write('\nData files: {0}, {1}'.format(synth_vec, synth_data))
            outfile.write('\nStepsize as percent of prior range: {0} %'.format(step_pcent*100))


    mcmc = MCMC(filename, xmlinput, simtime, samples, nCommunities, sedsim, sedlim, flowsim, flowlim, vis,
        gt_depths, gt_vec_d, gt_timelay, gt_vec_t, gt_prop_t,
        min_m, max_m, true_m, step_m, min_a, max_a, true_ax, true_ay ,step_a, step_sed, step_flow, assemblage)


    [pos_v, pos_diff, pos_likl, pos_samples_t, pos_samples_d, pos_sed1,pos_sed2,pos_sed3,pos_sed4,
    pos_flow1,pos_flow2,pos_flow3,pos_flow4, pos_ax,pos_ay,pos_m, gt_depths,
    accept_ratio, accepted_count, gt_vec_t,x_tick_labels, x_tick_values] = mcmc.sampler()
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
    plotResults.plotPosCore(filename,pos_samples_d,pos_samples_t,gt_vec_d,gt_vec_t,gt_depths,gt_timelay,x_tick_labels,x_tick_values, mcmc.font)
    plotResults.boxPlots(nCommunities, pos_v, sedsim, flowsim, mcmc.font,mcmc.width,filename)    
    plotResults.plotLiklAndDiff(pos_likl, pos_diff, sample_range, mcmc.font, filename)
    plotResults.plotParameters(mcmc.filename, assemblage, sample_range, mcmc.sedsim, mcmc.flowsim, mcmc.communities, 
        pos_m, pos_ax, pos_ay, mcmc.true_m, mcmc.true_ax, mcmc.true_ay, 
        pos_sed1, pos_sed2, pos_sed3, pos_sed4, mcmc.true_sed,
        pos_flow1, pos_flow2, pos_flow3, pos_flow4, mcmc.true_flow, mcmc.font, mcmc.width)
    print 'Finished simulations'

if __name__ == "__main__": main()
