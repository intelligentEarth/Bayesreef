#!/usr/bin/env python
#Title           :multinom_mcmc_d_constrained.py
#Description     :This is the script for BayesReef: an MCMC random walk method applied to pyReefCore.
#Author          :Jodie Pall and Danial Azam 
#Last edit       :19/07/2018
#Version         :1.0
#Usage           :python multinom_mcmc_d_constrained.py
#Notes           :Compared to mcmc_multinomial_d.py, the amount of free parameters can be limited from 1-11 in this script.
# 
#                 This script uses the depth-structure of a core simulation (i.e. coralgal assemblage at each 0.2 cm depth interval of a simulated core)
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
from scipy.stats import norm
from matplotlib.cm import terrain, plasma, Set2
import argparse


#Initialise and parse inputs
parser=argparse.ArgumentParser(description='Bayesreef modelling')

parser.add_argument('-s','--samples', help='Number of samples', default=10000, dest="samples",type=int)
parser.add_argument('-cs','--cov_start', help='Covariance calculation start', default=50, dest="cov_start",type=int)
parser.add_argument('-ci','--cov_interval', help='Covariance Interval', default=10, dest="cov_interval",type=int)
parser.add_argument('-uc','--use_cov', help='Flag for adaptive', default=1, dest="use_cov",type=int)
parser.add_argument('-f','--frozen', help='0 for 3 and 1 for 11 params', default=0, dest="frozen",type=int)


args = parser.parse_args()

if args.use_cov == 1:
    args.use_cov = True
else:
    args.use_cov = False

samples = args.samples #10000  # total number of samples by all the chains (replicas) in parallel tempering
cov_start = args.cov_start
cov_interval = args.cov_interval
use_cov = args.use_cov
frozen_assemparams = args.frozen

print '\n use_cov', use_cov , '\n'

cmap=plt.cm.Set2
c = cycler('color', cmap(np.linspace(0,1,8)) )
plt.rcParams["axes.prop_cycle"] = c

class MCMC():
    def __init__(self, filename, xmlinput, simtime, samples, communities, sedsim, sedlim, flowsim, flowlim, vis,
        gt_depths, gt_vec_d, gt_prop_d,gt_timelay, gt_vec_t,
        min_m, max_m, true_m, step_m, min_a, max_a, true_ax, true_ay,step_a, step_sed, step_flow, assemblage, use_cov):
        self.font = 10
        self.width = 1
        self.colors = terrain(np.linspace(0, 1.8, 14)) #len(reef.core.coralH)+10))
        self.colors2 = plasma(np.linspace(0, 1, 174)) #len(reef.core.layTime)+3))
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
        self.gt_vec_d = gt_vec_d
        self.gt_prop_d = gt_prop_d
        self.gt_timelay = gt_timelay
        self.gt_vec_t = gt_vec_t
        
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

        self.counter = 0
        self.cholesky = [] 
        self.cov_init = False
        self.use_cov = use_cov # decide if rw or adaptive mcmc proposals. False indicates   RW proposals 
        
    def runModel(self, reef, input_vector):
        reef.convertVector(self.communities, input_vector, self.sedsim, self.flowsim) #model.py
        self.true_sed, self.true_flow = reef.load_xml(self.input, self.sedsim, self.flowsim)
        # if self.vis[0] == True:
            # reef.core.initialSetting(size=(8,2.5), size2=(8,3.5)) # View initial parameters
        reef.run_to_time(self.simtime,showtime=100.)
        # if self.vis[1] == True:
        #     reef.plot.drawCore(lwidth = 3, colsed=self.colors, coltime = self.colors2, size=(9,8), font=8, dpi=300)
        sim_output_t, sim_timelay = reef.plot.convertTimeStructure() #modelPlot.py
        sim_output_d = reef.plot.convertDepthStructure(self.communities, self.gt_depths) #modelPlot.py
        # predicted_core = reef.convert_core(self.communities, output_core, self.core_depths) #model.py
        # return predicted_core
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

    def noGrowthColumn(self, sim_prop):
        # Creates additional binary column that takes a value of 1 where there is no growth, otherwise 0.
        v_nogrowth = np.zeros((sim_prop.shape[0],1))
        for a in range(sim_prop.shape[0]):
            if np.amax(sim_prop[a,:]) == 0.:
                v_nogrowth[a,:] = 1.
        sim_prop = np.append(sim_prop,v_nogrowth,axis=1)
        return sim_prop

    def likelihoodWithProps(self, reef, gt_prop_d, input_v):
        sim_prop_t, sim_prop_d, sim_timelay = self.runModel(reef, input_v)
        sim_prop_d = sim_prop_d.T
        # sim_prop_d5 = self.noGrowthColumn(sim_prop_d)
        # sim_prop_d5 = sim_prop_d5.T
        intervals = sim_prop_d.shape[0]

        log_core = np.log(sim_prop_d+0.0001)
        log_core[log_core == -inf] = 0
        z = log_core * gt_prop_d
        likelihood = np.sum(z)
        diff = self.diffScore(sim_prop_d,gt_prop_d, intervals)
        sim_vec_d = self.convertCoreFormat(sim_prop_d)
        sim_vec_t = self.convertCoreFormat(sim_prop_t)
        # rmse = self.rmse(sim_prop_t5, gt_prop_d)
        return [likelihood, diff, sim_prop_d, sim_vec_d, sim_vec_t]

    def likelihoodWithDominance(self, reef, synth_data, input_v):
        sim_data = self.runModel(reef, input_v)
        sim_data = sim_data.T
        intervals = sim_data.shape[0]
        z = np.zeros((intervals,self.communities+1))    
        for n in range(intervals):
            idx_data = np.argmax(synth_data[n,:])
            idx_model = np.argmax(sim_data[n,:])
            if ((sim_data[n,self.communities] != 1.) and (idx_data == idx_model)): #where sediment !=1 and max proportions are equal:
                z[n,idx_data] = 1
        same = np.count_nonzero(z)
        same = float(same)/intervals
        diff = 1-same
        # rmse = self.rmse(sim_data, synth_data)
        z = z + 0.1
        z = z/(1+(1+self.communities)*0.1)
        likelihood = np.log(z)   
        return [np.sum(likelihood), sim_data, diff]

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

    def proposalJump(self, current ,low_limit, high_limit, proposal):
        # proposal = current + np.random.normal(0, jump_width)
        if proposal >= high_limit:
            proposal = current
        elif proposal <= low_limit:
            proposal = current
        return proposal
    
    def computeCovariance(self, i, pos_v):
        cov_mat = np.cov(pos_v[:i,].T)



        # np.savetxt('cov_mat_%s.txt' %(), cov_mat )
        cov_noise = (self.prop_step * self.prop_step)*np.identity(cov_mat.shape[0], dtype = float)
        covariance = np.add(cov_mat, cov_noise)        
        L = np.linalg.cholesky(covariance)
        self.cholesky = L

        self.cov_init = True

        return cov_mat, cov_noise

    def sampler(self):
        samples = self.samples
        gt_vec_t = self.gt_vec_t
        gt_timelay = self.gt_timelay
        gt_vec_d = self.gt_vec_d
        gt_depths = self.gt_depths
        communities = self.communities
        flowlim = self.flowlim
        sedlim = self.sedlim

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

        num_accepted = np.zeros(samples)

        diff_proposals = np.zeros((samples , 3))

        pos_samples_d = np.zeros((samples, gt_vec_d.shape[0]))
        pos_samples_t = np.zeros((samples, gt_vec_t.shape[0]))
        pr_flow = np.zeros((samples , communities))
        pr_sed = np.zeros((samples , communities))

        sed1 = [0.0001, 0.0015, 0.0023]   # manually set inital values nearby true ones
        sed2 = [0.0012, 0.0017, 0.0024]
        sed3 = [0.0018, 0.0028, 0.0027]
        sed4 = [0.0020, 0.0031, 0.0043]
        flow1=[0.050, 0.008 ,0.]
        flow2=[0.078, 0.051, 0.]
        flow3=[0.250, 0.172, 0.058]
        flow4=[0.270, 0.185, 0.066]


        '''p_sed1_true=[0.0009, 0.0015, 0.0023]
        p_sed2_true=[0.0015, 0.0017, 0.0024]
        p_sed3_true=[0.0016, 0.0028, 0.0027]
        p_sed4_true=[0.0017, 0.0031, 0.0043]
        p_flow1_true=[0.055, 0.008 ,0.]
        p_flow2_true=[0.082, 0.051, 0.]
        p_flow3_true=[0.259, 0.172, 0.058]
        p_flow4_true=[0.288, 0.185, 0.066]'''

        '''cm_ax = -0.05   # for 11 param experiments, we assign cm matrix to true initial value, to make search less difficult. cm are free params for 11 param exp
        pos_ax[0] = cm_ax
        cm_ay = -0.06
        pos_ay[0] = cm_ay
        m = 0.11
        pos_m[0] = m'''


        #true_m = 0.08
        #true_ax = -0.01
        #true_ay = -0.03


        cm_ax = np.random.uniform(self.min_a,self.max_a)
        pos_ax[0] = cm_ax
        cm_ay = np.random.uniform(self.min_a,self.max_a)
        pos_ay[0] = cm_ay
        m = np.random.uniform(self.min_m, self.max_m)
        pos_m[0] = m

        pr_flow2 = flowlim[1]-flow1[0]
        pr_flow3 = flowlim[1]-flow2[0]
        pr_flow4 = flowlim[1]-flow3[0]
        prs_flow = np.array([pr_flow2,pr_flow3,pr_flow4])
        c_pr_flow = np.prod(prs_flow)
        pr_sed2 = sedlim[1]-sed1[0]
        pr_sed3 = sedlim[1]-sed2[0]
        pr_sed4 = sedlim[1]-sed3[0]
        prs_sed = np.array([pr_sed2,pr_sed3,pr_sed4])
        c_pr_sed = np.prod(prs_sed)
        c_cs_prod = c_pr_flow*c_pr_sed

        if (self.sedsim == True) and (self.flowsim == True):
            v_proposal = np.concatenate((sed1,sed2,sed3,sed4,flow1,flow2,flow3,flow4))
            v_proposal = np.append(v_proposal,(cm_ax,cm_ay,m))
            self.prop_step = np.repeat(np.array([self.step_sed]), 4*self.communities, axis = 0)
            self.prop_step = np.append(self.prop_step,(np.repeat(np.array([self.step_flow]), 4*self.communities, axis = 0)))
            self.prop_step = np.append(self.prop_step,([self.step_a, self.step_a, self.step_m]))
        else:
            v_proposal = np.array([cm_ax,cm_ay,m])
            self.prop_step = np.array([self.step_a, self.step_a, self.step_m])
        
        pos_v = np.zeros((samples, v_proposal.size))
        print 'v_proposal', v_proposal

        # Declare pyReef-Core and initialize
        reef = Model()

        [likelihood, diff, sim_pred_d,sim_vec_d, sim_vec_t] = self.likelihoodWithProps(reef, self.gt_prop_d, v_proposal)
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
            pos_m[0], pos_ax[0], pos_ay[0], pos_sed1[0,], pos_sed2[0,], pos_sed3[0,], pos_sed4[0,],
            pos_flow1[0,], pos_flow2[0,], pos_flow3[0,], pos_flow4[0,],
            pos_diff[0],pos_likl[0], pos_samples_d[0,],pos_v[0,])

        # print 'Begin sampling using MCMC random walk'
        x_tick_labels = ['Shallow', 'Mod-deep', 'Deep', 'Sediment','No growth']
        x_tick_values = [1,2,3,4,5]
        

        plotResults.plotInitialPrediction(self.filename, gt_vec_d, gt_vec_t,sim_vec_d,sim_vec_t,gt_depths,
            gt_timelay, x_tick_labels, x_tick_values)
                
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


        p_sed1 = sed1 # initial starting position
        p_sed2 = sed2
        p_sed3 = sed3 # initial starting position
        p_sed4 = sed4
        p_flow1 = flow1
        p_flow2 = flow2
        p_flow3 = flow3
        p_flow4 = flow4
        # p_sed1=[0.0009, 0.0015, 0.0023] # true
        # p_sed2=[0.0015, 0.0017, 0.0024]
        # p_sed3=[0.0016, 0.0028, 0.0027]
        # p_sed4=[0.0017, 0.0031, 0.0043]
        # p_flow1=[0.055, 0.008 ,0.]
        # p_flow2=[0.082, 0.051, 0.]
        # p_flow3=[0.259, 0.172, 0.058]
        # p_flow4=[0.288, 0.185, 0.066]
        
        p_sed1_true=[0.0009, 0.0015, 0.0023]
        p_sed2_true=[0.0015, 0.0017, 0.0024]
        p_sed3_true=[0.0016, 0.0028, 0.0027]
        p_sed4_true=[0.0017, 0.0031, 0.0043]
        p_flow1_true=[0.055, 0.008 ,0.]
        p_flow2_true=[0.082, 0.051, 0.]
        p_flow3_true=[0.259, 0.172, 0.058]
        p_flow4_true=[0.288, 0.185, 0.066]

        for i in range(samples - 1):
            print '\nSample: ', i
            start = time.time()


            if self.cov_init and self.use_cov : #Adaptive MCMC proposals
                # print 'self. cholesky', self.cholesky
                # print ' v _ prop', v_proposal   
                v_p = np.random.normal(size = v_proposal.shape)
                x = np.dot(self.cholesky,v_p)
                v_proposal_cov = v_proposal + x
                p_sed1[0:self.communities] = v_proposal_cov[0:3]
                p_sed2[0:self.communities] = v_proposal_cov[3:6]
                p_sed3[0:self.communities] = v_proposal_cov[6:9]
                p_sed4[0:self.communities] = v_proposal_cov[9:12]
                p_flow1[0:self.communities] = v_proposal_cov[12:15]
                p_flow2[0:self.communities] = v_proposal_cov[15:18]
                p_flow3[0:self.communities] = v_proposal_cov[18:21]
                p_flow4[0:self.communities] = v_proposal_cov[21:24]
                p_ax  = v_proposal_cov[24]
                p_ay = v_proposal_cov[25]
                p_m = v_proposal_cov[26]

                diff_proposals[i,:] = x[24:]

                print '\n using cov \n'
            else:  # classic random walk
                ax_ = np.random.normal(0, self.step_a)
                p_ax = cm_ax + ax_
                ay_ = np.random.normal(0, self.step_a)
                p_ay = cm_ay + ay_ 
                m_  = np.random.normal(0, self.step_m)
                p_m = m + m_

                for comm in range(self.communities):
                    p_flow1[comm] = flow1[comm] + np.random.normal(0, self.step_flow)
                    p_flow2[comm] = flow2[comm] + np.random.normal(0, self.step_flow)
                    p_flow3[comm] = flow3[comm] + np.random.normal(0, self.step_flow)
                    p_flow4[comm] = flow4[comm] + np.random.normal(0, self.step_flow)                
                    p_sed1[comm] = sed1[comm] + np.random.normal(0, self.step_sed)
                    p_sed2[comm] = sed2[comm] + np.random.normal(0, self.step_sed)
                    p_sed3[comm] = sed3[comm] + np.random.normal(0, self.step_sed)
                    p_sed4[comm] = sed4[comm] + np.random.normal(0, self.step_sed) 

                diff_proposals[i,:] =  np.array([ax_, ay_, m_])

                print '\n using rw \n '
            # frozen_assemparams # 0 means a,a,m and 1 means that assemble 2 and 3 are frozen to true (makes 11 params) , 2 would mean that only assemble 3 is frozen (makes 19 free params) and 3 would mean that all are free   (makes 27 free params)            


            p_sed1[frozen_assemparams:] = p_sed1_true[frozen_assemparams:] # for experimental purpsoe, if you wish to free only 1 assemblege and keep rest frozen to true
            p_sed2[frozen_assemparams:] = p_sed2_true[frozen_assemparams:]
            p_sed3[frozen_assemparams:] = p_sed3_true[frozen_assemparams:]
            p_sed4[frozen_assemparams:] = p_sed4_true[frozen_assemparams:]
            
            p_flow1[frozen_assemparams:] = p_flow1_true[frozen_assemparams:]
            p_flow2[frozen_assemparams:] = p_flow2_true[frozen_assemparams:]
            p_flow3[frozen_assemparams:] = p_flow3_true[frozen_assemparams:]
            p_flow4[frozen_assemparams:] = p_flow4_true[frozen_assemparams:]

            
            p_ax = self.proposalJump(cm_ax, self.min_a, self.max_a, p_ax)
            p_ay = self.proposalJump(cm_ay, self.min_a, self.max_a, p_ay)
            p_m = self.proposalJump(m, self.min_m, self.max_m, p_m)

            for comm in range(self.communities):
                p_flow1[comm] = self.proposalJump(flow1[comm],flowlim[0],flowlim[1], p_flow1[comm]) 
                p_flow2[comm] = self.proposalJump(flow2[comm],flow1[comm], flowlim[1], p_flow2[comm])
                p_flow3[comm] = self.proposalJump(flow3[comm],flow2[comm], flowlim[1], p_flow3[comm]) 
                p_flow4[comm] = self.proposalJump(flow4[comm],flow3[comm], flowlim[1], p_flow4[comm])
                p_sed1[comm] = self.proposalJump(sed1[comm],sedlim[0],sedlim[1], p_sed1[comm]) 
                p_sed2[comm] = self.proposalJump(sed2[comm],sed1[comm], sedlim[1], p_sed2[comm])
                p_sed3[comm] = self.proposalJump(sed3[comm],sed2[comm], sedlim[1], p_sed3[comm]) 
                p_sed4[comm] = self.proposalJump(sed4[comm],sed3[comm], sedlim[1], p_sed4[comm])
            
            
            tmat_s = np.concatenate((p_sed1,p_sed2,p_sed3,p_sed4)).reshape(4,communities)
            tmatrix = tmat_s.T
            tmp = np.zeros((communities,4))
            for x in range(tmatrix.shape[0]):
                a = np.sort(tmatrix[x,:])
                tmp[x,:] = a
            tmat = tmp.T
            p_sed1 = tmat[0,:]
            p_sed2 = tmat[1,:]
            p_sed3 = tmat[2,:]
            p_sed4 = tmat[3,:]

            tmat_f = np.concatenate((p_flow1,p_flow2,p_flow3,p_flow4)).reshape(4,communities)
            tmatrix = tmat_f.T
            tmp = np.zeros((communities,4))
            for x in range(tmatrix.shape[0]):
                a = np.sort(tmatrix[x,:])
                tmp[x,:] = a
            tmat = tmp.T
            p_flow1 = tmat[0,:]
            p_flow2 = tmat[1,:]
            p_flow3 = tmat[2,:]
            p_flow4 = tmat[3,:]

            v_proposal = np.concatenate((p_sed1,p_sed2,p_sed3,p_sed4,p_flow1,p_flow2,p_flow3,p_flow4))
            v_proposal = np.append(v_proposal,(p_ax,p_ay,p_m))

            print 'V_proposalllll ,', v_proposal

            [likelihood_proposal, diff, sim_pred_d,sim_vec_d, sim_vec_t] = self.likelihoodWithProps(reef, self.gt_prop_d, v_proposal)
            diff_likelihood = likelihood_proposal - likelihood # to divide log probability, must subtract
            print 'likelihood_proposal:', likelihood_proposal, 'diff_likelihood',diff_likelihood

            p_cs_flow1 = norm.cdf(flowlim[1],p_flow1, self.step_flow) - norm.cdf(flowlim[0],p_flow1, self.step_flow) 
            p_cs_flow2 = norm.cdf(flowlim[1],p_flow2, self.step_flow) - norm.cdf(flowlim[0],p_flow2, self.step_flow)
            p_cs_flow3 = norm.cdf(flowlim[1],p_flow3, self.step_flow) - norm.cdf(flowlim[0],p_flow3, self.step_flow)
            p_cs_flow4 = norm.cdf(flowlim[1],p_flow4, self.step_flow) - norm.cdf(flowlim[0],p_flow4, self.step_flow)
            p_cs_m = norm.cdf(self.max_m,p_m, self.step_m) - norm.cdf(self.min_a,p_m, self.step_m)
            p_cs_ax = norm.cdf(self.max_m,p_ax, self.step_a) - norm.cdf(self.min_a,p_ax, self.step_a)
            p_cs_ay = norm.cdf(self.max_m,p_ay, self.step_a) - norm.cdf(self.min_a,p_ay, self.step_a)
            p_cs_flow1_prod = np.prod(p_cs_flow1)
            p_cs_flow2_prod = np.prod(p_cs_flow2)
            p_cs_flow3_prod = np.prod(p_cs_flow3)
            p_cs_flow4_prod = np.prod(p_cs_flow4)
            p_cs_all = np.array([p_cs_m,p_cs_ax,p_cs_ay])
            p_cs_all = np.array([p_cs_flow1_prod, p_cs_flow2_prod, p_cs_flow3_prod, p_cs_flow4_prod, p_cs_m, p_cs_ax, p_cs_ay])
            p_cs_prod = np.prod(p_cs_all)
            q_ratio = c_cs_prod/p_cs_prod
            print 'q_ratio', q_ratio

            if not os.path.isfile(('%s/pos_priors.csv' % (self.filename))):
                with file(('%s/pos_priors.csv' % (self.filename)),'wb') as outfile:
                    writer = csv.writer(outfile, delimiter=',')
                    titlelist = ["p_cs_prod","c_cs_prod","q_ratio","likelihood_proposal","likelihood","diff_likelihood"]
                    writer.writerow(titlelist)
                    data = [p_cs_prod,c_cs_prod,q_ratio,likelihood_proposal,likelihood,diff_likelihood]
                    writer.writerow(data)
            else:
                with file(('%s/pos_priors.csv' % (self.filename)),'ab') as outfile:
                    writer = csv.writer(outfile, delimiter=',')
                    data = [p_cs_prod,c_cs_prod,q_ratio,likelihood_proposal,likelihood,diff_likelihood]
                    writer.writerow(data)


            mh_prob = min(1, math.exp(diff_likelihood)*q_ratio)
            # mh_prob = min(1, math.exp(diff_likelihood))
            u = random.uniform(0, 1)
            print 'u', u, 'and mh_probability', mh_prob

            num_accepted[i] = naccept
            
            if u < mh_prob: # accept
                #   Update position
                print i, ' is accepted sample','   - number of accepted : ', naccept
                naccept += 1

                
                count_list.append(i)
                likelihood = likelihood_proposal
                c_cs_prod = p_cs_prod
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
                    pos_diff[i+1],pos_likl[i+1], pos_samples_d[i+1,],pos_v[i+1,])

            else: #reject
                pos_v[i + 1,] = pos_v[i,]
                pos_samples_t[i + 1,] = pos_samples_t[i,]
                pos_samples_d[i + 1,] = pos_samples_d[i,]
                pos_diff[i + 1,] = pos_diff[i,]
                pos_likl[i + 1,] = pos_likl[i,]
                print 'REJECTED\nLikelihood:',likelihood,'and difference score:', pos_diff[i,]
                print ' - number of accepted solutions  : ', naccept
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
                    pos_diff[i+1],pos_likl[i+1], pos_samples_d[i+1,],pos_v[i+1,])
                print i, 'rejected and retained'

            end = time.time()
            total_time = end-start
            print 'Time elapsed:', total_time

            if i==samples - 2:
                self.saveCore(reef, i+1)
            if i in range(cov_start, samples, cov_interval) :
                print 'cov computed = i ',i, '\n'
                cov_mat, cov_noise = self.computeCovariance(i,pos_v)

                print(cov_mat[24:27,24:27], ' cov_mat') 

                print(cov_noise[24:27,24:27], ' cov_noise')

                print(self.cholesky[24:27,24:27], ' cholesky')



        accepted_count =  len(count_list)   
        print accepted_count, ' number accepted'
        print len(count_list) / (samples * 0.01), '% was accepted'
        accept_ratio = accepted_count / (samples * 1.0) * 100

        finalfig.tight_layout(pad=2.0)
        finalfig.savefig('%s/proposals_d.png'% (self.filename),bbox_inches='tight',dpi=200,transparent=False)
        plt.close()

        return (pos_v, pos_diff, pos_likl, pos_samples_t, pos_samples_d, pos_sed1,pos_sed2,pos_sed3,pos_sed4,
            pos_flow1,pos_flow2,pos_flow3,pos_flow4, pos_ax,pos_ay,pos_m, 
            accept_ratio, accepted_count,x_tick_labels, x_tick_values, diff_proposals, num_accepted)

#####################################################################

def main():
    
    #    Set all input parameters    #
    random.seed(time.time())
    # samples= 100
    # description = raw_input('Enter description: ')
    description = 'depth-based likelihood, self.likelihoodWithProps'
    assemblage = 3
    xmlinput = 'input_synth.xml'
    synth_prop = 'data/synth_core_prop_d_08.txt'
    synth_vec = 'data/synth_core_vec_d_08.txt'
    gt_depths, gt_vec_d = np.genfromtxt(synth_vec, usecols=(0,1), unpack=True)
    gt_timelay, gt_vec_t = np.genfromtxt('data/synthdata_t_vec_08_1.txt', usecols=(0, 1), unpack = True)
    gt_prop_d = np.loadtxt(synth_prop, usecols=(1,2,3,4))
    gt_timelay = gt_timelay[::-1]




    print(gt_depths, ' gt depths')
    print(gt_vec_d, ' gt_vec_d') 

    print(gt_timelay, ' gt_timelay') 
    print(gt_vec_t, ' gt_vec_t')

    print(gt_prop_d, ' gt_prop_d')

    





    # use_cov = False
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

    step_stdev = 0.01
    step_sed = step_stdev * abs(sedlim[0]-sedlim[1])
    step_flow = step_stdev * abs(flowlim[0]-flowlim[1])
    step_m = step_stdev * abs(min_m-max_m)
    step_a = step_stdev * abs(min_a-max_a)

    path_name = 'results-multinomial-d'
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
            outfile.write('\nData files: {0}, {1}'.format(synth_vec, synth_prop))
            outfile.write('\nStepsize as percentage of prior range: {0} %'.format(step_stdev*100))

    mcmc = MCMC(filename, xmlinput, simtime, samples, nCommunities, sedsim, sedlim, flowsim, flowlim, vis,
        gt_depths, gt_vec_d, gt_prop_d, gt_timelay, gt_vec_t,
        min_m, max_m, true_m, step_m, min_a, max_a, true_ax, true_ay ,step_a, step_sed, step_flow, assemblage, use_cov)

    [pos_v, pos_diff, pos_likl, pos_samples_t, pos_samples_d, pos_sed1,pos_sed2,pos_sed3,pos_sed4,
    pos_flow1,pos_flow2,pos_flow3,pos_flow4, pos_ax,pos_ay,pos_m,
    accept_ratio, accepted_count,x_tick_labels, x_tick_values, diff_proposals, num_accepted] = mcmc.sampler()
    print 'Successfully sampled'
    
    
    burnin = 0.15 * samples  # use post burn in samples
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

    '''if not os.path.isfile(('%s/pos_burnin_sed.csv' % (filename))):
        np.savetxt("%s/pos_burnin_sed.csv" % (filename), np.c_[pos_sed1,pos_sed2,pos_sed3, pos_sed4], delimiter=',')

    if not os.path.isfile(('%s/pos_burnin_flow.csv' % (filename))):
        np.savetxt("%s/pos_burnin_flow.csv" % (filename), np.c_[pos_flow1,pos_flow2,pos_flow3, pos_flow4], delimiter=',')

    if not os.path.isfile(('%s/pos_burnin_likl.csv' % (filename))):
        np.savetxt("%s/pos_burnin_likl.csv" % (filename), pos_likl, delimiter=',')

    if not os.path.isfile(('%s/pos_burnin_proposal.csv' % (filename))):
        np.savetxt("%s/pos_burnin_proposal.csv" % (filename), pos_v, delimiter=',')'''

    if not os.path.isfile(('%s/diff_pro.csv' % (filename))):
        np.savetxt("%s/diff_pro.csv" % (filename), diff_proposals, delimiter=',')


    if not os.path.isfile(('%s/num_accepted.csv' % (filename))):
        np.savetxt("%s/num_accepted.csv" % (filename), num_accepted, delimiter=',')


    sample_range = np.arange(burnin+1,samples+1, 1)
    
    # JODIE: REDEFINE NEW PLOTTING FOR DEPTH ONLY
    plotResults.plotPosCore(filename,pos_samples_d,pos_samples_t,gt_vec_d,gt_vec_t,gt_depths,gt_timelay,x_tick_labels,x_tick_values, mcmc.font)
    plotResults.plotPosCoreDepth(filename, pos_samples_d, gt_vec_d, gt_depths, x_tick_labels, x_tick_values, mcmc.font)
    plotResults.boxPlots(nCommunities, pos_v, sedsim, flowsim, mcmc.font,mcmc.width,filename)    
    plotResults.plotLiklAndDiff(pos_likl, pos_diff, sample_range, mcmc.font, filename)
    plotResults.plotParameters(mcmc.filename, assemblage, sample_range, mcmc.sedsim, mcmc.flowsim, mcmc.communities, 
        pos_m, pos_ax, pos_ay, mcmc.true_m, mcmc.true_ax, mcmc.true_ay, 
        pos_sed1, pos_sed2, pos_sed3, pos_sed4, mcmc.true_sed,
        pos_flow1, pos_flow2, pos_flow3, pos_flow4, mcmc.true_flow,mcmc.font, mcmc.width)
    print 'Finished simulations'

if __name__ == "__main__": main()