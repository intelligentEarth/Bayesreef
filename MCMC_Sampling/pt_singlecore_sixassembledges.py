# !/usr/bin/python
# BayesReef: a MCMC random walk method applied to pyReef-Core
# Authors: Jodie Pall and Danial Azam (2017)
# Adapted from: [Chandra_ICONIP2017] R. Chandra, L. Azizi, S. Cripps, 'Bayesian neural learning via Langevin dynamicsfor chaotic time series prediction', ICONIP 2017.
# (to be addeded on https://www.researchgate.net/profile/Rohitash_Chandra)

import os
import math
import time
import random
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import ticker
import matplotlib.mlab as mlab
from pyReefCore.model import Model
import fnmatch
import matplotlib as mpl
from cycler import cycler
from scipy import stats 

cmap=plt.cm.Set2
c = cycler('color', cmap(np.linspace(0,1,8)) )
plt.rcParams["axes.prop_cycle"] = c

config = 0 # for parameter limits config

class MCMC():
    def __init__(self, simtime, samples, communities, core_data, core_depths,timestep,filename, xmlinput,   vis, true_vec_parameters, problem, num_replica, max_temp, burn_in, pt_stage):
        self.filename = filename
        self.input = xmlinput
        self.communities = communities
        self.samples = samples       
        self.core_data = core_data
        self.core_depths = core_depths
        self.timestep = timestep
        self.vis = vis
        self.sedsim = True
        self.flowsim = True
        
        self.sedlimits = []
        self.flowlimits = []

        self.simtime = simtime
        self.font = 4
        self.width = 1
        self.d_sedprop = float(np.count_nonzero(core_data[:,self.communities]))/core_data.shape[0]
        self.initial_sed = []
        self.initial_flow = []
   

        self.true_values = true_vec_parameters
        self.problem = problem

        self.num_chains = num_replica
        self.maxtemp = max_temp

        self.adapttemp = 1
        self.temperature = 1 

        self.burn_in = burn_in
        self.pt_stage = pt_stage

        if config ==1:
            self.step_m = 0.1 
            self.step_a = 0.02   

            self.step_sed = 0.005 
            self.step_flow = 0.1 

            self.max_a = -0.15 
            self.max_m = 0.25

            self.sedlim = [0., 0.01]
            self.flowlim = [0.,0.5]
        elif config == 2: 
            self.step_m = 0.2
            self.step_a = 0.2  

            self.step_sed = 0.01 
            self.step_flow = 0.2 

            self.max_a = -0.5 
            self.max_m = 0.5

            self.sedlim = [0., 0.05]
            self.flowlim = [0.,1] 
        else: 
            self.step_m = 0.02 
            self.step_a = 0.02 

            self.max_a = -0.15
            self.max_m = 0.15

            self.sedlim = [0., 0.005]
            self.flowlim = [0.,0.3]

            self.step_sed = 0.001 
            self.step_flow = 0.05




    def run_Model(self, reef, input_vector):
        reef.convert_vector(self.communities, input_vector, self.sedsim, self.flowsim) #model.py
        self.initial_sed, self.initial_flow = reef.load_xml(self.input, self.sedsim, self.flowsim)


        print(self.initial_sed, self.initial_flow , '   * initial sed and initial flow')
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
        #predicted_core = reef.convert_core(self.communities, output_core, self.core_depths) #model.py
        #return predicted_core 
        return output_core

    def pos_sedflow(self, pos):


        pos_sed1 = pos[0:3,:]   
        pos_sed2 = pos[3:6,:]
        pos_sed3 = pos[6:9,:]
        pos_sed4 = pos[9:12,:]
        pos_flow1 = pos[12:15,:]
        pos_flow2 = pos[15:19,:]
        pos_flow3 = pos[19:22,:] 
        pos_flow4 = pos[22:25,:] 


        nb_bins=30
        slen = np.arange(0,pos_sed1.shape[0],1)
 
 


        # PLOT SEDIMENT AND FLOW RESPONSE THRESHOLDS #

        if self.communities == 3:
        	a_labels = ['Shallow windward', 'Moderate-deep windward', 'Deep windward']#, 'Shallow leeward', 'Moderate-deep leeward', 'Deep leeward']
    	else:
    		a_labels = ['Windward Shallow', 'Windward Mod-deep', 'Windward Deep', 'Sediment','Leeward Shallow', 'Leeward Mod-deep', 'Leeward Deep']
    	 
        
        sed1_mu, sed1_ub, sed1_lb, sed2_mu, sed2_ub, sed2_lb, sed3_mu, sed3_ub, sed3_lb, sed4_mu, sed4_ub, sed4_lb = (np.zeros(self.communities) for i in range(12))
        if ((self.sedsim != False)):
            for a in range(self.communities):
                sed1_mu[a] = np.mean(pos_sed1[:,a])
                sed1_ub[a] = np.percentile(pos_sed1[:,a], 95, axis=0)
                sed1_lb[a] = np.percentile(pos_sed1[:,a], 5, axis=0)
                
                sed2_mu[a] = np.mean(pos_sed2[:,a])
                sed2_ub[a] = np.percentile(pos_sed2[:,a], 95, axis=0)
                sed2_lb[a] = np.percentile(pos_sed2[:,a], 5, axis=0)
                
                sed3_mu[a] = np.mean(pos_sed3[:,a])
                sed3_ub[a] = np.percentile(pos_sed3[:,a], 95, axis=0)
                sed3_lb[a] = np.percentile(pos_sed3[:,a], 5, axis=0)
                
                sed4_mu[a] = np.mean(pos_sed4[:,a])
                sed4_ub[a] = np.percentile(pos_sed4[:,a], 95, axis=0)
                sed4_lb[a] = np.percentile(pos_sed4[:,a], 5, axis=0)

                sed1_mu_=sed1_mu[a]
                sed2_mu_=sed2_mu[a]
                sed3_mu_=sed3_mu[a]
                sed4_mu_=sed4_mu[a]
                sed1_min=sed1_lb[a]
                sed2_min=sed2_lb[a]
                sed3_min=sed3_lb[a]
                sed4_min=sed4_lb[a]
                sed1_max=sed1_ub[a]
                sed2_max=sed2_ub[a]
                sed3_max=sed3_ub[a]
                sed4_max=sed4_ub[a]
                sed1_med=np.median(pos_sed1[:,a])
                sed2_med=np.median(pos_sed2[:,a])
                sed3_med=np.median(pos_sed3[:,a])
                sed4_med=np.median(pos_sed4[:,a])
                sed1_mode, count=stats.mode(pos_sed1[:,a])
                sed2_mode, count=stats.mode(pos_sed2[:,a])
                sed3_mode, count=stats.mode(pos_sed3[:,a])
                sed4_mode, count=stats.mode(pos_sed4[:,a])


                with file(('%s/summ_stats.txt' % (self.filename)),'a') as outfile:
                    #outfile.write('\n# Sediment threshold: {0}\n'.format(a_labels[a]))
                    outfile.write('5TH %ILE, 95TH %ILE, MEAN, MEDIAN\n')
                    outfile.write('Sed1\n{0}, {1}, {2}, {3}\n'.format(sed1_min,sed1_max,sed1_mu_,sed1_med))
                    outfile.write('Sed2\n{0}, {1}, {2}, {3}\n'.format(sed2_min,sed2_max,sed2_mu_,sed2_med))
                    outfile.write('Sed3\n{0}, {1}, {2}, {3}\n'.format(sed3_min,sed3_max,sed3_mu_,sed3_med))
                    outfile.write('Sed4\n{0}, {1}, {2}, {3}\n'.format(sed4_min,sed4_max,sed4_mu_,sed4_med))
                    outfile.write('Modes\n\tSed1:\t{0}\n\tSed2:\t{1}\n\tSed3:\t{2}\n\tSed4:\t{3}'.format(sed1_mode,sed2_mode,sed3_mode,sed4_mode))

                cy = [0,100,100,0]
                cmu = [sed1_mu[a], sed2_mu[a], sed3_mu[a], sed4_mu[a]]
                c_lb = [sed1_mu[a]-sed1_lb[a], sed2_mu[a]-sed2_lb[a], sed3_mu[a]-sed3_lb[a], sed4_mu[a]-sed4_lb[a]]
                c_ub = [sed1_ub[a]-sed1_mu[a], sed2_ub[a]-sed2_mu[a], sed3_ub[a]-sed3_mu[a], sed4_ub[a]-sed4_mu[a]]
                
                fig = plt.figure(figsize=(6,4))
                ax = fig.add_subplot(111)
                ax.set_facecolor('#f2f2f3')
                if self.problem ==1:
                    ax.plot(self.initial_sed[a,:], cy, linestyle='--', linewidth=self.width, marker='.',color='k', label='True')
                ax.plot(cmu, cy, linestyle='-', linewidth=self.width,marker='.', color='sandybrown', label='Estimate')
                ax.errorbar(cmu[0:2],cy[0:2],xerr=[c_lb[0:2],c_ub[0:2]],capsize=5,elinewidth=1, color='darksalmon',mfc='darksalmon',fmt='.',label=None)
                ax.errorbar(cmu[2:4],cy[2:4],xerr=[c_lb[2:4],c_ub[2:4]],capsize=5,elinewidth=1, color='sienna',mfc='sienna',fmt='.',label=None)

                ax.tick_params(axis='both', which='major', labelsize=10)
                ax.tick_params(axis='both', which='minor', labelsize=10)

                ax.set_xlabel("Sediment input (m/year)", fontsize=11)
                ax.set_ylabel("Max. growth rate", fontsize=11) 
                


                plt.title('Sediment exposure threshold function\n(%s assemblage)' % (a_labels[a]), size=11, y=1.06)
                #plt.ylabel('Proportion of maximum growth rate [%]',size=self.font+1)
                #plt.xlabel('Sediment input [m/year]',size=self.font+1)
                plt.ylim(-2.,110)
                lgd = plt.legend(frameon=False, prop={'size':10}, bbox_to_anchor = (1.,0.2))
                plt.savefig('%s/sediment_response_%s.pdf' % (self.filename, a+1), bbox_extra_artists=(lgd,),bbox_inches='tight',dpi=300,transparent=False)
                plt.clf()

        flow1_mu, flow1_ub,flow1_lb, flow2_mu, flow2_ub,flow2_lb, flow3_mu, flow3_ub,flow3_lb, flow4_mu, flow4_ub,flow4_lb = (np.zeros(self.communities) for i in range(12))
        if (self.flowsim != False):
            for a in range(self.communities):
                flow1_mu[a] = np.mean(pos_flow1[:,a])
                flow1_ub[a] = np.percentile(pos_flow1[:,a], 95, axis=0)
                flow1_lb[a] = np.percentile(pos_flow1[:,a], 5, axis=0)
                
                flow2_mu[a] = np.mean(pos_flow2[:,a])
                flow2_ub[a] = np.percentile(pos_flow2[:,a], 95, axis=0)
                flow2_lb[a] = np.percentile(pos_flow2[:,a], 5, axis=0)
                
                flow3_mu[a] = np.mean(pos_flow3[:,a])
                flow3_ub[a] = np.percentile(pos_flow3[:,a], 95, axis=0)
                flow3_lb[a] = np.percentile(pos_flow3[:,a], 5, axis=0)
                
                flow4_mu[a] = np.mean(pos_flow4[:,a])
                flow4_ub[a] = np.percentile(pos_flow4[:,a], 95, axis=0)
                flow4_lb[a] = np.percentile(pos_flow4[:,a], 5, axis=0)

                flow1_mu_ = flow1_mu[a]
                flow2_mu_ = flow2_mu[a]
                flow3_mu_ = flow3_mu[a]
                flow4_mu_ = flow4_mu[a]
                flow1_min= flow1_lb[a]
                flow1_max=flow1_ub[a]
                flow1_med=np.median(pos_flow1[:,a])
                flow2_min=flow2_lb[a]
                flow2_max=flow2_ub[a]
                flow2_med=np.median(pos_flow2[:,a])
                flow3_min=flow3_lb[a]
                flow3_max=flow3_ub[a]
                flow3_med=np.median(pos_flow3[:,a])
                flow4_min=flow4_lb[a]
                flow4_max=flow4_ub[a]
                flow4_med=np.median(pos_flow4[:,a])
                flow1_mode, count= stats.mode(pos_flow1[:,a])
                flow2_mode, count= stats.mode(pos_flow2[:,a])
                flow3_mode, count= stats.mode(pos_flow3[:,a])
                flow4_mode, count= stats.mode(pos_flow4[:,a])

                with file(('%s/summ_stats.txt' % (self.filename)),'a') as outfile:
                    #outfile.write('\n# Water flow threshold: {0}\n'.format(a_labels[a]))
                    outfile.write('#5TH %ILE, 95TH %ILE, MEAN, MEDIAN\n')
                    outfile.write('# flow1\n{0}, {1}, {2}, {3}\n'.format(flow1_min,flow1_max,flow1_mu_,flow1_med))
                    outfile.write('# flow2\n{0}, {1}, {2}, {3}\n'.format(flow2_min,flow2_max,flow2_mu_,flow2_med))
                    outfile.write('# flow3\n{0}, {1}, {2}, {3}\n'.format(flow3_min,flow3_max,flow3_mu_,flow3_med))
                    outfile.write('# flow4\n{0}, {1}, {2}, {3}\n'.format(flow4_min,flow4_max,flow4_mu_,flow4_med))
                    outfile.write('Modes\n\tFlow1:\t{0}\n\tFlow2:\t{1}\n\tFlow3:\t{2}\n\tFlow4:\t{3}'.format(flow1_mode,flow2_mode,flow3_mode,flow4_mode))

                cy = [0,100,100,0]
                cmu = [flow1_mu[a], flow2_mu[a], flow3_mu[a], flow4_mu[a]]
                c_lb = [flow1_mu[a]-flow1_lb[a], flow2_mu[a]-flow2_lb[a], flow3_mu[a]-flow3_lb[a], flow4_mu[a]-flow4_lb[a]]
                c_ub = [flow1_ub[a]-flow1_mu[a], flow2_ub[a]-flow2_mu[a], flow3_ub[a]-flow3_mu[a], flow4_ub[a]-flow4_mu[a]]

                
                fig = plt.figure(figsize=(6,4))

                params = {'legend.fontsize': 15, 'legend.handlelength': 2}
                plt.rcParams.update(params)
                ax = fig.add_subplot(111)
                ax.set_facecolor('#f2f2f3')
                if self.problem ==1:
                    ax.plot(self.initial_flow[a,:], cy, linestyle='--', linewidth=self.width, marker='.', color='k',label='True')
                ax.plot(cmu, cy, linestyle='-', linewidth=self.width, marker='.', color='steelblue', label='Estimate')
                ax.errorbar(cmu[0:2],cy[0:2],xerr=[c_lb[0:2],c_ub[0:2]],capsize=5,elinewidth=1,color='lightsteelblue',mfc='lightsteelblue',fmt='.',label=None)
                ax.errorbar(cmu[2:4],cy[2:4],xerr=[c_lb[2:4],c_ub[2:4]],capsize=5,elinewidth=1,color='lightslategrey',mfc='lightslategrey',fmt='.',label=None)

                ax.tick_params(axis='both', which='major', labelsize=10)
                ax.tick_params(axis='both', which='minor', labelsize=10)

                ax.set_xlabel("Fluid flow (m/sec)", fontsize=11)
                ax.set_ylabel("Max. growth rate", fontsize=11) 

                plt.title('Hydrodynamic energy exposure threshold function\n(%s assemblage)' % (a_labels[a]), size=11, y=1.06)
                #plt.ylabel('Proportion of maximum growth rate [%]', size=self.font+1)
                #plt.xlabel('Fluid flow [m/sec]', size=self.font+1)
                plt.ylim(-2.,110.)
                lgd = plt.legend(frameon=False, prop={'size':10}, bbox_to_anchor = (1.,0.2))
                plt.savefig('%s/flow_response_%s.pdf' % (self.filename, a+1),  bbox_extra_artists=(lgd,), bbox_inches='tight',dpi=300,transparent=False)
                plt.clf()

 
 
        

    def convert_core_format(self, core, communities):
        vec = np.zeros(core.shape[0])
        for n in range(len(vec)):
            idx = np.argmax(core[n,:])# get index,
            vec[n] = idx+1 # +1 so that zero is preserved as 'none'
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

    def diff_score(self, z,intervals):
        same= np.count_nonzero(z)
        same = float(same)/intervals
        diff = 1-same
        print 'diff:', diff
        return diff*100

    '''def diff_score_new(self, sim_data,synth_data,intervals):
        maxprop = np.zeros((intervals,sim_data.shape[1]))
        for n in range(intervals):
            idx_synth = np.argmax(synth_data[n,:])
            idx_sim = np.argmax(sim_data[n,:])
            if ((sim_data[n,self.communities] != 1.) and (idx_synth == idx_sim)): #where sediment !=1 and max proportions are equal:
                maxprop[n,idx_synth] = 1
        diff = (1- float(np.count_nonzero(maxprop))/intervals)*100
        return diff'''

    def give_weight(self, arr):   
        index_array = np.zeros(arr.shape[0]) 
        for i in range(0, arr.shape[0]): 
            if (arr[i] == 0): 
                index_array[i] = 1
            else:  
                index_array[i] = 0 

        return index_array


    def convertmat_assemindex(self, arr):   
        index_array = np.zeros(arr.shape[0]) 
        for i in range(0, arr.shape[0]): 
            for j in range(0, arr.shape[1]):  
                if (arr[i][j] == 1): 
                    index_array[i] = j 
        return index_array

    def score_updated(self, predictions, targets):
        # where there is 1 in the sed column, count

        predictions = np.where(predictions > 0.5, 1, 0) 

        p = self.convertmat_assemindex(predictions) #predictions.dot(1 << np.arange(predictions.shape[-1])) 

        a =  self.convertmat_assemindex(self.core_data)  
 
        '''sed = np.count_nonzero(predictions[:,self.communities])
        p_sedprop = (float(sed)/predictions.shape[0])
        sedprop = np.absolute((self.d_sedprop - p_sedprop)*0.5)'''

        diff = np.absolute( p-a)
        print diff, ' diff abs'

        weight_array = self.give_weight(diff)

        score = np.sum(weight_array)/weight_array.shape[0]
 
        #rmse =(np.sqrt(((p - a) ** 2).mean()))*0.5


        
        return (1- score) * 100  #+ sedprop 


    def likelihood_func(self, reef, core_data, input_v):
        pred_core = self.run_Model(reef, input_v)
        pred_core = pred_core.T
        intervals = pred_core.shape[0]
        z = np.zeros((intervals,self.communities+1))   

        #print(z, intervals, ' is z int') 
        for n in range(intervals):
            idx_data = np.argmax(core_data[n,:])
            idx_model = np.argmax(pred_core[n,:])
            if ((pred_core[n,self.communities] != 1.) and (idx_data == idx_model)): #where sediment !=1 and max proportions are equal:
                z[n,idx_data] = 1
        diff = self.diff_score(z,intervals)

        #diff = self.diffScore(sim_prop_d,gt_prop_d, intervals)
        diff_ = self.score_updated(pred_core, core_data)

        print(diff, diff_, ' diff and new diff....')
        
        z = z + 0.1
        z = z/(1+(1+self.communities)*0.1)
        loss = np.log(z)
        # print 'sum of loss:', np.sum(loss)        
        return [np.sum(loss) *(1.0/self.adapttemp), pred_core, diff_]

    def save_core(self,reef,naccept):
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
        
    
    '''def save_core(self,reef,naccept):
        path = '%s/%s' % (self.filename, naccept)
        if not os.path.exists(path):
            os.makedirs(path)
        
        #     Initial settings     #
        reef.core.initialSetting(size=(8,2.5), size2=(8,4.5), dpi=300, fname='%s/a_thres_%s_' % (path, naccept))
        from matplotlib.cm import terrain, plasma
        nbcolors = len(reef.core.coralH)+10
        colors = terrain(np.linspace(0, 1.8, nbcolors))
        nbcolors = len(reef.core.layTime)+3
        colors2 = plasma(np.linspace(0, 1, nbcolors))
        
        #      Community population evolution    #
        reef.plot.speciesDepth(colors=colors, size=(8,4), font=8, dpi=300, fname =('%s/b_popd_%s.png' % (path,naccept)))
        reef.plot.speciesTime(colors=colors, size=(8,4), font=8, dpi=300,fname=('%s/c_popt_%s.png' % (path,naccept)))
        reef.plot.accomodationTime(size=(8,4), font=8, dpi=300, fname =('%s/d_acct_%s.pdf' % (path,naccept)))
        
        #      Draw core      #
        reef.plot.drawCore(lwidth = 3, colsed=colors, coltime = colors2, size=(9,8), font=8, dpi=300, 
                           figname=('%s/e_core_%s' % (path, naccept)), filename=('%s/core_%s.csv' % (path, naccept)), sep='\t')
        
        # pdflist = [f for f in os.listdir(os.curdir) if fnmatch.fnmatch(f, ('*_%s*.pdf' % (naccept)))]
        # print pdflist
        # merger = PdfFileMerger()
        # for pdf in pdflist:
        #     merger.append(PdfFileReader(file(pdf, 'rb')))
        # merger.write('output_%s.pdf' % (naccept))       
        
        return'''

    def initial_replicaproposal(self): 

        """"Windward sedlim = 0.005, flowlim = 0.3
            sedlim_1 = [[0., 0.0035]]
            sedlim_2 = [[0.001,0.0035]]
            sedlim_3 = [[0.001,0.005]]
            sedlim_4 = sedlim_5 = sedlim_6 = [[0.,0.]] 
            # sedlim_4 = [[0.001,0.0035]]
            # sedlim_5 = [[0.002,0.004]]
            # sedlim_6 = [[0.002,0.005]]
            flowlim_1 = [[0.02,0.3]]
            flowlim_2 = [[0.005.,0.2]]
            flowlim_3 = [[0.,0.15]]
            flowlim_4 = [[0.005,0.2]]
            flowlim_5 = [[0.002,0.1]]
            flowlim_6 = [[0.,0.1]]
            Leeward sedlim = 0.005, flowlim = 0.2
            # sedlim_1 = [[0.0005,0.0035]]
            # sedlim_2 = [[0,1e-3]]
            # sedlim_3 = [[0,2e-4]]
            sedlim_1 = sedlim_2 = sedlim_3 = [[0.,0.]] 
            sedlim_4 = [[0.0005,0.0035]]
            sedlim_5 = [[0.0005, 0.003]]
            sedlim_6 = [[0. 0.005]]
            flowlim_1 = [[0.05,0.3]]
            flowlim_2 = [[0.05,0.3]]
            flowlim_3 = [[0,0.2]]
            flowlim_4 = [[0.01,0.3]]
            flowlim_5 = [[0,0.2]]
            flowlim_6 = [[0,0.1]]
        """

        sed1 = np.zeros(self.communities)
        sed2 = np.zeros(self.communities)
        sed3 = np.zeros(self.communities)
        sed4 = np.zeros(self.communities)

        flow1 = np.zeros(self.communities)
        flow2 = np.zeros(self.communities)
        flow3 = np.zeros(self.communities)
        flow4 = np.zeros(self.communities)

  
        for s in range(self.communities):
            '''sed1[s]  = np.random.uniform(0.,0.)
            sed2[s]  = np.random.uniform(0.,0.)
            sed3[s]  = np.random.uniform(0.005,0.005)
            sed4[s]  = np.random.uniform(0.005,0.005)'''


            sed1[s] = np.random.uniform(self.sedlim[0],self.sedlim[1])
            sed2[s] = np.random.uniform(sed1[s],self.sedlim[1])
            sed3[s] = np.random.uniform(sed2[s],self.sedlim[1])
            sed4[s] = np.random.uniform(sed3[s],self.sedlim[1])

 
        for x in range(self.communities):
            #     relaxed constraints 
            '''flow1[s]   = np.random.uniform(0.,0.)
            flow2[s]   = np.random.uniform(0.,0.)
            flow3[s]   = np.random.uniform(0.3,0.3)
            flow4[s]   = np.random.uniform(0.3,0.3)'''

            flow1[x] = np.random.uniform(self.flowlim[0], self.flowlim[1])
            flow2[x] = np.random.uniform(flow1[x], self.flowlim[1])
            flow3[x] = np.random.uniform(flow2[x], self.flowlim[1])
            flow4[x] = np.random.uniform(flow3[x], self.flowlim[1])


       
        cm_ax   = np.random.uniform(self.max_a,0.)
        cm_ay  = np.random.uniform(self.max_a,0.)
        m   = np.random.uniform(0.,self.max_m)

        sedlim_1 = [[0., 0.0035]]
        sedlim_2 = [[0.001,0.0035]]
        sedlim_3 = [[0.001,0.005]]

        flowlim_1 = [[0.01,0.3]]
        flowlim_2 = [[0.,0.2]]
        flowlim_3 = [[0.,0.1]]


 
        self.sedlimits = np.concatenate((sedlim_1,sedlim_2,sedlim_3))#sedlim_4,sedlim_5,sedlim_6)) 
        self.flowlimits = np.concatenate((flowlim_1,flowlim_2,flowlim_3))#flowlim_4,flowlim_5,flowlim_6))'''


        glv_pro = np.array([cm_ax,cm_ay,m])
 
        v_proposal = np.concatenate((sed1,sed2,sed3,sed4,flow1,flow2,flow3,flow4))
        #v_proposal = np.append(v_proposal,(cm_ax,cm_ay,m))

        return np.hstack((v_proposal,glv_pro)) #np.ravel(v_proposal) #, m, cm_ax, cm_ay


    def proposal_vec(self, v_current):


        size_sed = 4 * self.communities
        size_flow = 4 * self.communities

        max_a = self.max_a
        max_m = self.max_m

 

        #if self.sedsim == True:
        tmat = v_current[0:size_sed]#np.concatenate((sed1,sed2,sed3,sed4)).reshape(4,self.communities)
        tmat = tmat.reshape(4,self.communities)

         
        tmatrix = tmat.T
        t2matrix = np.zeros((tmatrix.shape[0], tmatrix.shape[1]))
        for x in range(self.communities):#-3):
            for s in range(tmatrix.shape[1]):
                t2matrix[x,s] = tmatrix[x,s] + np.random.normal(0,self.step_sed)
                if t2matrix[x,s] >= self.sedlim[1]:
                    t2matrix[x,s] = tmatrix[x,s]
                elif t2matrix[x,s] <= 0:
                    t2matrix[x,s] = tmatrix[x,s]
            # reorder each row , then transpose back as sed1, etc.
        tmp = np.zeros((self.communities,4))
        for x in range(t2matrix.shape[0]):
            a = np.sort(t2matrix[x,:])
            tmp[x,:] = a
        tmat = tmp.T
        p_sed1 = tmat[0,:]
        p_sed2 = tmat[1,:]
        p_sed3 = tmat[2,:]
        p_sed4 = tmat[3,:]
            
        #if self.flowsim == True:
        tmat = v_current[size_sed:size_sed+size_flow] #np.concatenate((flow1,flow2,flow3,flow4)).reshape(4,self.communities)
        tmat = tmat.reshape(4,self.communities)

        tmatrix = tmat.T
        t2matrix = np.zeros((tmatrix.shape[0], tmatrix.shape[1]))
        for x in range(self.communities):#-3):
            for s in range(tmatrix.shape[1]):
                t2matrix[x,s] = tmatrix[x,s] + np.random.normal(0,self.step_flow)
                if t2matrix[x,s] >= self.flowlim[1]:
                    t2matrix[x,s] = tmatrix[x,s]
                elif t2matrix[x,s] <= 0:
                    t2matrix[x,s] = tmatrix[x,s]
            # reorder each row , then transpose back as flow1, etc.
        tmp = np.zeros((self.communities,4))
        for x in range(t2matrix.shape[0]):
            a = np.sort(t2matrix[x,:])
            tmp[x,:] = a
        tmat = tmp.T
        p_flow1 = tmat[0,:]
        p_flow2 = tmat[1,:]
        p_flow3 = tmat[2,:]
        p_flow4 = tmat[3,:]

  

        cm_ax = v_current[size_sed+size_flow] 
        cm_ay = v_current[size_sed+size_flow+1] 
        m = v_current[size_sed+size_flow+2] 



        p_ax = cm_ax + np.random.normal(0,self.step_a,1)
        if p_ax > 0:
            p_ax = cm_ax
        elif p_ax < max_a:
            p_ax = cm_ax
        p_ay = cm_ay + np.random.normal(0,self.step_a,1)
        if p_ay > 0:
            p_ay = cm_ay
        elif p_ay < max_a:
            p_ay = cm_ay   
        p_m = m + np.random.normal(0,self.step_m,1)
        if p_m < 0:
            p_m = m
        elif p_m > max_m:
            p_m = m  

        glv_pro = np.array([p_ax,p_ay,p_m])
 

        v_proposal = np.concatenate((p_sed1,p_sed2,p_sed3,p_sed4,p_flow1,p_flow2,p_flow3,p_flow4))
        

        for a in glv_pro:
            v_proposal = np.append(v_proposal, a)
 
        return v_proposal  #np.ravel(v_proposal)

    def assign_temperature(self):  

        temp_ladder = []
 

        tmpr_rate = float(self.maxtemp /self.num_chains)
        temp = 1
        for i in range(0, self.num_chains):
            temp_ladder.append(temp)
            temp += tmpr_rate

        temp_ladder = [1, 1.05, 1.1, 1.15 , 1.2, 1.3, 1.4, 1.6, 1.9, 2.5]

        #temp_ladder = [1, 1 , 1 , 1  , 1 , 1 , 1 , 1 , 1 , 1 ]

        return   temp_ladder



    def sampler(self):

        start = time.time()


        data_size = self.core_data.shape[0]
        total_samples = self.samples
        x_data = self.core_depths
        y_data = self.core_data
        nreplicas = self.num_chains
        samples = total_samples/nreplicas

        data_vec = self.convert_core_format(self.core_data, self.communities)

        temp_ladder = self.assign_temperature()

        print temp_ladder, ' is temp ladder ------------------------------------'



        burnin = int(self.burn_in * samples)
        #pt_stage = int(0.99 * samples) # paralel tempering is used only for exploration, it does not form the posterior, later mcmc in parallel is used with swaps 
        pt_stage = int(self.pt_stage * samples) # paralel tempering is used only for exploration, it does not form the posterior, later mcmc in parallel is used with swaps 
      
        swap_interval = 1 # when to check to swap 

        with file(('%s/description.txt' % (self.filename)),'a') as outfile:
            outfile.write('\n\tstep_m: {0}'.format(self.step_m))
            outfile.write('\n\tstep_a: {0}'.format(self.step_a))
            outfile.write('\n\tstep_sed: {0}'.format(self.step_sed))
            outfile.write('\n\tstep_flow: {0}'.format(self.step_flow))

        
 
        rep_likelihood = np.zeros(nreplicas)
        rep_likelihood_pro = np.zeros(nreplicas)
        rep_predcore = np.zeros((nreplicas, samples ))
        rep_diffscore = np.zeros((nreplicas, samples ))

        rep_likelihoodlist = np.zeros((nreplicas, samples ))

        rep_acceptlist = np.zeros((nreplicas, samples ))




        # Create space to store fx of all samples
        list_predcore = np.zeros((nreplicas, samples, self.core_data.shape[0]))

        num_param = 3 + (self.communities * 8 )  # 3  for the mal, cim_ax, cim_ay 

        replica_pro = np.zeros((nreplicas, num_param)) # proposal for each replica 
        replicapos_v = np.zeros((nreplicas, samples , num_param)) # pos for each replica 
        pos_v = np.zeros((total_samples, num_param)) # pos for all replicas 
         

        reef = Model() # initiate the pyReef-Core module 

    

        m = 0
        cm_ax = 0
        cm_ay = 0


        #self.save_core(reef, 'initial')



        for r in range(nreplicas):
            replica_pro[r,:] = self.initial_replicaproposal() 
            print(replica_pro[r,:], replica_pro[r,:].shape,  ' proposal INIT ')
  
            likelihood, rep_predcore_, rep_diffscore[r,0] = self.likelihood_func(reef, self.core_data, replica_pro[r,:]) 

            rep_likelihood[r] = likelihood *(1.0/temp_ladder[r])


            replicapos_v[r,0,:] = replica_pro[r,:]
       
            list_predcore[r,0,:] = self.convert_core_format(rep_predcore_, self.communities)
       
            print (r,'\tinitial likelihood:', rep_likelihood[r], 'and difference score:', rep_diffscore[r,0])



            
            #count_list = []
            #count_list.append(0)
 

        print(' begin sampling ....')

        total_accept = 0
        #naccept = 0
        naccept = np.zeros(nreplicas)

        init_count = 0
 
        for i in range(samples - 1 ): 
        

            v_current = replica_pro[r,:]  

            for r in range(nreplicas):  
                #for s in range(swap_interval):  
                
                print '\nSample - Replica  ', i, r

                v_proposal = self.proposal_vec(v_current) 

                self.adapttemp =  temp_ladder[r] 

                if i < pt_stage: 
                    self.adapttemp =  temp_ladder[r]
                    #likelihood_proposal = rep_likelihood_pro[r] *(1.0/temp_ladder[r])
                    #rep_likelihood_pro[r] = likelihood_proposal
                else:
                    self.adapttemp = 1

        
 
                #print(v_proposal, ' proposal ')
                if i == pt_stage and init_count ==0: 
                    print ' moving to mcmc sampling ------------------  **** ------'
                    likelihood, rep_predcore_, diffscore  = self.likelihood_func(reef, self.core_data, v_proposal)
                    rep_likelihood[r] = likelihood
                    init_count = 1

                likelihood_proposal, rep_predcore_, diffscore  = self.likelihood_func(reef, self.core_data, v_proposal)

                rep_likelihood_pro[r] = likelihood_proposal

 
                diff_likelihood = rep_likelihood_pro[r] - rep_likelihood[r] # to divide probability, must subtract
                print 'likelihood_proposal:',  rep_likelihood_pro[r] , 'diff_likelihood',diff_likelihood, ' diff_score', diffscore


                mh_prob = min(1, math.exp(diff_likelihood))
                u = random.uniform(0, 1)
                print 'u', u, 'and mh_probability', mh_prob

                rep_diffscore[r,i +1] = diffscore
 
                rep_acceptlist[r,i +1] = naccept[r] 

                rep_likelihoodlist[r,i +1] = likelihood_proposal



                
                if u < mh_prob: # accept 

                    naccept[r] += 1 


                    
                    rep_likelihood[r] = likelihood_proposal.copy()
                    replica_pro[r,:]  = v_proposal 

                    replicapos_v[r,i+1,:] = v_proposal

                    list_predcore[r,i+1,:] = self.convert_core_format(rep_predcore_, self.communities)
                    
                    print  i, 'likelihood:', likelihood_proposal, ' and difference score:',  diffscore,   naccept[r] , 'accepted'

                    #print v_proposal, ' accepted proposal *** '

 
               
                else: #reject
 
                    replicapos_v[r,i+1,:] = replicapos_v[r,i,:] 

                    list_predcore[r,i+1,:] = list_predcore[r,i,:]
                    #rep_diffscore[r,i +1] = rep_diffscore[r,i]
 
                    print i,  naccept[r] , 'rejected and retained'

            print ' time to check swap --------------------------------------- *'

            for s in range(1, nreplicas): 

                lhood1 = rep_likelihood[s-1]
                lhood2 = rep_likelihood[s] 

                swap_proposal = min(1, math.exp(lhood2 - lhood1))
                '''try:
                    swap_proposal =  min(1,0.5*np.exp(min(709, lhood2 - lhood1)))
                except OverflowError:
                    swap_proposal = 1'''

                u = np.random.uniform(0,1) 

                print s, lhood1, lhood2, swap_proposal, u , lhood2 - lhood1, '   s, lhood1, lhood2, swap_proposal, u, lhood2 - lhood1 '
                if u < swap_proposal:  
                    temp =  replica_pro[s-1,:]   
                    replica_pro[s-1,:] = replica_pro[s,:].copy()
                    replica_pro[s,:] = temp.copy()

                    print s, ' swapped * ++ '
                else:
                    print s, ' no swap * ' 

  
             
        end = time.time()

        total_time = end-start
        print 'Time elapsed:', total_time

        print naccept, ' list accepted '

 
        accept_ratio = np.sum(naccept)/ (self.samples * 1.0) * 100

        print  accept_ratio, '% was accepted'

        burn_pos = replicapos_v[:,burnin:,:] 
        burn_listpred = list_predcore[:,burnin:,:]
 


        posterior = burn_pos.transpose(2,0,1).reshape(num_param,-1) 

        predcore_list = burn_listpred.transpose(2,0,1).reshape(self.core_data.shape[0],-1) 

        #print(posterior, ' posterior after burn')  
        #print(predcore_list, ' predcore_list')  
        #print(rep_diffscore, ' rep_diffscore  ...')  

        #print self.true_values

        diffscore = rep_diffscore[:,burnin:]


        for s in range( 0, num_param):  
            print self.true_values[s]  
            
            self.plot_figure(posterior[s,:], 'pos_distri_'+str(s),  self.true_values[s] , nreplicas ) 

        self.pos_sedflow(posterior) 

 

        return (rep_diffscore, accept_ratio, posterior, predcore_list,  x_data, y_data, data_vec, rep_acceptlist, rep_likelihoodlist, diffscore, total_time/3600)


    def plot_figure(self, list, title, real_value, nreplicas  ): 

        list_points =  list
        fname = self.filename
         


        size = 14

        #fig, ax = plt.subplots()


        '''plt.tick_params(labelsize=size)
        params = {'legend.fontsize': size, 'legend.handlelength': 2}
        plt.rcParams.update(params)
        plt.grid(alpha=0.75)

        plt.hist(list_points,  bins = 20, color='#0504aa',
                            alpha=0.7)   

        plt.title("Posterior distribution ", fontsize = size)
        plt.xlabel(' Parameter value  ', fontsize = size)
        plt.ylabel(' Frequency ', fontsize = size)
        if self.problem == 1:
            plt.axvline(x=real_value, linewidth=2, color='r')
            print real_value, ' is real'
        #plt.tight_layout()  
        #ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.autoscale()'''


        fig, ax = plt.subplots()
   


        ax.hist(list_points,    bins=20, rwidth=0.9,   color='#607c8e')
 
        ax.tick_params(axis="x", labelsize=14)
        ax.tick_params(axis="y", labelsize=14)
        #ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.3f}')) 


        fmtr = ticker.StrMethodFormatter(('{x:,g}'))
        ax.yaxis.set_major_formatter(fmtr)
 
        ax.yaxis.set_minor_formatter(ticker.StrMethodFormatter('{x:.3f}'))

        ax.set_xlabel("Parameter", fontsize=15)
        ax.set_ylabel("Frequency", fontsize=15) 

        if self.problem == 1:
            plt.axvline(x=real_value, linewidth=2, color='b')
 
        
        ax.grid(linestyle='-', linewidth='0.2', color='grey')
        plt.tight_layout()  
        plt.savefig(fname   +'/posterior/'+ title  + '_posterior.pdf')
        plt.clf()


        '''plt.tick_params(labelsize=size)
        params = {'legend.fontsize': size, 'legend.handlelength': 2}
        plt.rcParams.update(params)
        plt.grid(alpha=0.75)

        listx = np.asarray(np.split(list_points,  nreplicas ))
        plt.plot(listx.T)   

        plt.title("Parameter trace plot", fontsize = size)
        plt.xlabel(' Number of Samples  ', fontsize = size)
        plt.ylabel(' Parameter value ', fontsize = size)
        #plt.tight_layout()  
        plt.autoscale()'''


        #p = np.linspace(1000, 500, 100)
        #T = np.linspace(300, 200, p.size)


        fig, ax = plt.subplots()

        listx = np.asarray(np.split(list_points,  nreplicas ))
        ax.plot(listx.T)    

        ax.tick_params(axis="x", labelsize=14)
        ax.tick_params(axis="y", labelsize=14)
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.3f}')) 
 
        ax.yaxis.set_minor_formatter( ticker.StrMethodFormatter(('{x:,g}')) )
  
        ax.set_xlabel("Iterations", fontsize=15)
        ax.set_ylabel("Parameter", fontsize=15) 
        
        ax.grid(linestyle='-', linewidth='0.2', color='grey')
        plt.tight_layout()  

        plt.savefig(fname  +'/posterior/'+ title  + '_trace.pdf')
        plt.clf()

def make_directory (directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def core_convertbinary(core_data):




    core_binary = np.zeros((core_data.shape[0], 7))


    for i in range(core_data.shape[0]):  
    	assem_num = int(round(core_data[i] * 7) -1)
    	core_binary[i,assem_num] = 1
    	#print(assem_num, ' assem_num')

    #print(core_binary)

    return core_binary


#####################################################################
#####################################################################
#####################################################################
#####################################################################
#####################################################################

def main():
    
    #    Set all input parameters    #
    random.seed(time.time())


    samples=20000  

    num_replica = 10
    max_temp = 2.5

    burn_in = 0.5
    pt_stage = 0.95

    problem = 1 # 1. is synthetic core (3 communities/assembledges), 2. is Henon island real core (3 communities/assembledges) 3.  (see xml file )

 

    if problem ==1:
    	simtime = 8500
    	timestep = np.arange(0,simtime+1,50)
    	xmlinput = 'input_synth_.xml'
    	datafile = 'data/synth_core.txt'
    	core_depths = np.genfromtxt(datafile, usecols=(0), unpack = True) 
    	core_data = np.loadtxt('data/synth_core_bi.txt')

        true_vec_parameters = np.loadtxt('data/true_values.txt')

        print true_vec_parameters, ' true values'


    	nCommunities = 3


    elif problem ==2:
    	simtime = 8500
    	timestep = np.arange(0,simtime+1,50)
    	xmlinput = 'input_hi3_threeasembleges.xml'
    	datafile = 'data/hi3.txt'
    	core_depths = np.genfromtxt(datafile, usecols=(0), unpack = True) 
    	core_data = np.loadtxt('data/hi3_binary.txt') 
    	nCommunities = 3


    elif problem ==3:
    	simtime = 8500
    	timestep = np.arange(0,simtime+1,50)
    	xmlinput = 'input_synth_sixassem.xml'
    	datafile = 'data/synth_core.txt'
    	core_depths = np.genfromtxt(datafile, usecols=(0), unpack = True) 
    	core_data =   core_convertbinary(np.genfromtxt(datafile, usecols=(1), unpack = True) )  
        true_vec_parameters = np.zeros(51)#np.loadtxt('data/true_values_six.txt')

        print true_vec_parameters, ' true values' 

    	nCommunities = 6 # no assem



    elif problem ==4:

    	simtime = 8500
    	timestep = np.arange(0,simtime+1,50)
    	xmlinput = 'input_hi3.xml'
    	datafile = 'data/hi3.txt'
    	core_depths = np.genfromtxt(datafile, usecols=(0), unpack = True) 
    	core_data =   core_convertbinary(np.genfromtxt(datafile, usecols=(1), unpack = True) )  
        true_vec_parameters = np.zeros(51)#np.loadtxt('data/true_values_six.txt') 
 
    	nCommunities = 6

    elif problem ==5:
    	simtime = 8500
    	timestep = np.arange(0,simtime+1,50)
    	xmlinput = 'input_oti5.xml'
    	datafile = 'data/oti5.txt'
    	core_depths = np.genfromtxt(datafile, usecols=(0), unpack = True) 
    	core_data =   core_convertbinary(np.genfromtxt(datafile, usecols=(1), unpack = True) )  
        true_vec_parameters = np.zeros(51)#np.loadtxt('data/true_values_six.txt') 

    	nCommunities = 6
    elif problem ==6:
        simtime = 8500
        timestep = np.arange(0,simtime+1,50)
        xmlinput = 'input_oti2.xml'
        datafile = 'data/oti2.txt'
        core_depths = np.genfromtxt(datafile, usecols=(0), unpack = True) 
        core_data =   core_convertbinary(np.genfromtxt(datafile, usecols=(1), unpack = True) )  
        true_vec_parameters = np.zeros(51)#np.loadtxt('data/true_values_six.txt') 

        nCommunities = 6


    description = ''



    vis = [True, True] # first for initialisation, second for cores
    #sedsim, flowsim = True, True
    run_nb = 0
    while os.path.exists('results_syn%s' % (run_nb)):
        run_nb+=1
    if not os.path.exists('results_syn%s' % (run_nb)):
        os.makedirs('results_syn%s' % (run_nb))
    filename = ('results_syn%s' % (run_nb))

    
    make_directory(filename+'/posterior')
    make_directory(filename+'/results')

    #    Save File of Run Description   #
    if not os.path.isfile(('%s/description.txt' % (filename))):
        with file(('%s/description.txt' % (filename)),'w') as outfile:
            outfile.write('Test Description\n')
            outfile.write(description)
            outfile.write('\nSpecifications')
            outfile.write('\n\tmcmc.py')
            outfile.write('\n\tSimulation time: {0} yrs'.format(simtime)) 
            outfile.write('\n\tNo. samples: {0}'.format(samples))

            outfile.write('\n\tNo. chains: {0}'.format(num_replica))
            outfile.write('\n\tXML input: {0}'.format(xmlinput))
            outfile.write('\n\tData file: {0}'.format(datafile))
    
  

    mcmc = MCMC(simtime, samples, nCommunities, core_data, core_depths, timestep,  filename, xmlinput, 
                vis, true_vec_parameters, problem, num_replica, max_temp, burn_in, pt_stage)


    rep_diffscore, accept_ratio, pos_v, predcore_list, x_data, y_data, data_vec, rep_acceptlist, rep_likelihoodlist, diffscore, time_taken  = mcmc.sampler()

    print 'successfully sampled'

    score = diffscore.flatten()

    mean_score = np.mean(score)
    std_score = np.std(score)

    print  mean_score, std_score, accept_ratio, time_taken, '  mean score, std score, accept_ratio, time'

    np.savetxt(filename+'/score_summary.txt', [mean_score, std_score, accept_ratio, time_taken], fmt='%1.2f') 




 




 


    plt.plot( rep_diffscore.flatten())
    plt.title('Difference Score Evolution')
    plt.xlabel('Samples')
    plt.ylabel('Score') 
    plt.savefig( filename+'/rep_diffscore.png')
    plt.clf()

    #print(rep_diffscore, ' rep_diffscore')


    np.savetxt(filename+'/rep_diffscore.txt', diffscore, fmt='%1.2f')  
    np.savetxt(filename+'/predcore_list.txt', predcore_list, fmt='%1.2f')  
    np.savetxt(filename+'/posterior.txt', pos_v, fmt='%1.4e')

    sed_pos = pos_v[0:12,:]
    #print sed_pos, sed_pos.shape
    flow_pos = pos_v[12:24,:]
    glv_pos =   pos_v[24:,]
 


    mpl_fig = plt.figure()
    ax = mpl_fig.add_subplot(111) 
    size = 12 
    ax.tick_params(labelsize=size) 
    plt.legend(loc='upper right')  
    ax.boxplot(sed_pos.T) 
    ax.set_xlabel('Parameter ID', fontsize=size)
    ax.set_ylabel('Sediment Posterior', fontsize=size) 
    plt.title("Boxplot of Sediment Posterior", fontsize=size) 
    plt.savefig(filename+'/sed_pos.pdf')
    plt.clf()


    mpl_fig = plt.figure()
    ax = mpl_fig.add_subplot(111) 
    size = 12
    ax.tick_params(labelsize=size)
    ax.boxplot(flow_pos.T) 
    ax.set_xlabel('Parameter ID', fontsize=size)
    ax.set_ylabel('Flow Posterior', fontsize=size) 
    plt.title("Boxplot of Flow Posterior", fontsize=size) 
    plt.savefig(filename+'/flow_pos.pdf')
    plt.clf()


    mpl_fig = plt.figure()
    ax = mpl_fig.add_subplot(111) 
    size = 12 
    ax.tick_params(labelsize=size)
    ax.boxplot(glv_pos.T) 
    ax.set_xlabel('Parameter ID', fontsize=size)
    ax.set_ylabel('GLV Posterior', fontsize=size) 
    plt.title("Boxplot of GLV Posterior", fontsize=size) 
    plt.savefig(filename+'/glv_pos.pdf')
    plt.clf()



    plt.plot( rep_acceptlist.flatten()) 
    plt.xlabel('Samples')
    plt.ylabel('accepted') 
    plt.savefig( filename+'/rep_acceptlist.png')
    plt.clf()



    plt.plot( rep_likelihoodlist.flatten()) 
    plt.xlabel('Samples')
    plt.ylabel('likelihood') 
    plt.savefig( filename+'/rep_likelihoodlist.png')


     

    fx_mu = predcore_list.mean(axis=1)
    fx_high = np.percentile(predcore_list, 95, axis=1)
    fx_low = np.percentile(predcore_list, 5, axis=1)

    print data_vec.shape, '   data_vec'
    print x_data.shape, '   x_data'
    print fx_mu.shape, '  mean pred'
    print fx_high.shape, '  high'

    '''fig = plt.figure(figsize=(3,6))
    plt.plot(data_vec, x_data,label='Reef-core ground-truth', color='k')
    plt.plot(fx_mu,x_data, label='Model Pred. (mean)',linewidth=1,linestyle='--')
    plt.plot(fx_low, x_data, label='Model  Pred. (5th percentile)',linewidth=1,linestyle='--')
    plt.plot(fx_high,x_data, label='Model  Pred. (95th percentile)',linewidth=1,linestyle='--')
    plt.fill_betweenx(x_data, fx_low, fx_high, facecolor='mediumaquamarine', alpha=0.4, label=None)
    
    #plt.title("Reef-Core  vs  Prediction", size=mcmc.font+2)
    plt.ylim([0.,np.amax(core_depths)])
    plt.ylim(plt.ylim()[::-1])
    plt.ylabel('Depth [m]', size=mcmc.font+1)
    x_tick_labels = ['No growth','Shallow', 'Mod-deep', 'Deep', 'Sediment']
    x_tick_values = [0,1,2,3,4]
    plt.xticks(x_tick_values, x_tick_labels,rotation=70, fontsize=mcmc.font+1)
    plt.legend(frameon=False, prop={'size':mcmc.font+1}, bbox_to_anchor = (1.,0.2))
    plt.savefig('%s/predictions.png' % (filename), bbox_inches='tight', dpi=300,transparent=False)
    plt.clf()'''

    font = 8

    if nCommunities == 3:  
    	x_labels = ['Shallow', 'Mod-deep', 'Deep', 'Sediment', 'No growth', ]
    	x_values = [1,2,3,4,5]
    else: 
    	x_labels = [ 'W shallow', 'W Mod-deep', 'W Deep', 'Sediment','L Shallow', 'L Mod-deep', 'L Deep', 'No growth']
    	x_values = [1,2,3,4,5,6,7, 8 ]


    fig = plt.figure(figsize=(4,4))
    suptitle = fig.suptitle('')
    params = {'legend.fontsize': size, 'legend.handlelength': 2}
    plt.rcParams.update(params)
    ax1 = fig.add_subplot(121)
    ax1.set_facecolor('#f2f2f3')
    ax1.plot(data_vec, x_data, label='Ground truth', color='k',linewidth=0.7)
    ax1.plot(fx_mu, x_data, label='Pred. (mean)',linestyle='--', linewidth=0.7)
    ax1.plot(fx_high, x_data, label='Pred. (5th percentile)',linestyle='--',linewidth=0.7)
    ax1.plot(fx_low, x_data, label='Pred. (95th percentile)',linestyle='--',linewidth=0.7)
    ax1.fill_betweenx(x_data, fx_low, fx_high, facecolor='mediumaquamarine', alpha=0.4)
    ax1.set_ylabel('Depth (meters)')
    ax1.set_ylim([0,np.amax(core_depths)])
    ax1.set_ylim(ax1.get_ylim()[::-1])
    ax1.set_xticks(x_values)
    ax1.set_xticklabels(x_labels, rotation=70)
    ax1.tick_params(axis='both', which='major', labelsize=8)
    ax1.tick_params(axis='both', which='minor', labelsize=8)
 

    lgd = fig.legend(frameon=False,bbox_to_anchor = (0.45,0.19), borderpad=2., prop={'size':font-1})
    plt.tight_layout(pad=2.5)
    fig.savefig('%s/core_prediction.pdf' % (filename), bbox_extra_artists=(lgd,suptitle), bbox_inches='tight',dpi=200,transparent=False)
    plt.close('all')





    with file(('%s/out_results.txt' % (filename)),'w') as outres:
        #outres.write('Mean diff: {0}\nStandard deviation: {1}\nMode: {2}\n'.format(diff_mu, diff_std,diff_mode))
        outres.write('Accept ratio: {0} %\nSamples accepted : {1}'.format(accept_ratio,  samples))
  


    print 'Finished simulations'
if __name__ == "__main__": main()
