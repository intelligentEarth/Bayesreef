##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
##                                                                                   ##
##  This file forms part of the BayesReef modelling software                         ##
##                                                                                   ##
##  For full license and copyright information, please refer to the LICENSE.md file  ##
##  located at the project root, or contact the authors.                             ##
##                                                                                   ##
##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
"""
This script is intended to plot histograms and boxplots for the posterior
distribution of parameters used in BayesReef.
"""

import os
import csv
import numpy as np
import scipy as sp
from scipy import stats 
import matplotlib.pyplot as plt

def plotPosCore(filedir, pos_samples_1, pos_samples_2, gt_1, gt_2, y_1, y_2, x_labels, x_values, font):

    mu_1 = pos_samples_1.mean(axis=0)
    high_1 = np.percentile(pos_samples_1, 95, axis=0)
    low_1 = np.percentile(pos_samples_1, 5, axis=0)
    mu_2 = pos_samples_2.mean(axis=0)
    high_2 = np.percentile(pos_samples_2, 95, axis=0)
    low_2 = np.percentile(pos_samples_2, 5, axis=0)

    fig = plt.figure(figsize=(4,4))
    suptitle = fig.suptitle('Mean predictions and uncertainty')
    ax1 = fig.add_subplot(121)
    ax1.set_facecolor('#f2f2f3')
    ax1.plot(gt_1, y_1, label='Ground truth', color='k',linewidth=0.7)
    ax1.plot(mu_1, y_1, label='Pred. (mean)',linestyle='--', linewidth=0.7)
    ax1.plot(high_1, y_1, label='Pred. (5th %ile)',linestyle='--',linewidth=0.7)
    ax1.plot(low_1, y_1, label='Pred. (95th %ile)',linestyle='--',linewidth=0.7)
    ax1.fill_betweenx(y_1, low_1, high_1, facecolor='mediumaquamarine', alpha=0.4)
    ax1.set_ylabel('Depth [m]')
    ax1.set_ylim([0,np.amax(y_1)])
    ax1.set_ylim(ax1.get_ylim()[::-1])
    ax1.set_xticks(x_values)
    ax1.set_xticklabels(x_labels, rotation=70)
    
    ax2 = fig.add_subplot(122)
    ax2.set_facecolor('#f2f2f3')
    ax2.plot(gt_2, y_2, color='k',linewidth=0.7)
    ax2.plot(mu_2, y_2,linestyle='--',linewidth=0.7)
    ax2.plot(high_2, y_2,linestyle='--',linewidth=0.7)
    ax2.plot(low_2, y_2,linestyle='--',linewidth=0.7)
    ax2.fill_betweenx(y_2, low_2, high_2, facecolor='mediumaquamarine', alpha=0.4)
    ax2.set_ylabel('Simulation time [yrs]')
    ax2.set_ylim([0,np.amax(y_2)])
    ax2.set_ylim(ax2.get_ylim()[::-1])
    ax2.set_xticks(x_values)
    ax2.set_xticklabels(x_labels, rotation=70)
    lgd = fig.legend(frameon=False,bbox_to_anchor = (0.95,0.19), borderpad=2., prop={'size':font-3})
    plt.tight_layout(pad=2.5)
    fig.savefig('%s/mcmcres.png' % (filedir), bbox_extra_artists=(lgd,suptitle), bbox_inches='tight',dpi=200,transparent=False)
    plt.close('all')

def plotInitialPrediction(filedir, gt_1, gt_2, init_1, init_2, y_1, y_2,x_labels, x_values):
    
    fig = plt.figure(figsize=(4,4))
    suptitle = fig.suptitle('Initial Prediction')
    ax1 = fig.add_subplot(121)
    ax1.set_facecolor('#f2f2f3')
    ax1.plot(gt_1,y_1, label='Ground truth', color='k',linewidth=0.7)
    ax1.plot(init_1, y_1, label='Initial prediction',linewidth=0.7)
    ax1.set_ylabel('Depth [m]')
    ax1.set_ylim([0,np.amax(y_1)])
    ax1.set_ylim(ax1.get_ylim()[::-1])
    ax1.set_xticks(x_values)
    ax1.set_xticklabels(x_labels, rotation=70)
    
    ax2 = fig.add_subplot(122)
    ax2.set_facecolor('#f2f2f3')
    ax2.plot(gt_2,y_2, color='k',linewidth=0.7)
    ax2.plot(init_2, y_2,linewidth=0.7)
    ax2.set_ylabel('Simulation time [yrs]')
    ax2.set_ylim([0,np.amax(y_2)])
    ax2.set_ylim(ax2.get_ylim()[::-1])
    ax2.set_xticks(x_values)
    ax2.set_xticklabels(x_labels, rotation=70)
    lgd = fig.legend(frameon=False,bbox_to_anchor = (0.95,0.093)) #, prop={'size':font+1},bbox_to_anchor = (1.,1.))
    plt.tight_layout(pad=2.0)
    fig.savefig('%s/initpred.png' % (filedir), bbox_extra_artists=(lgd,suptitle), bbox_inches='tight',dpi=200,transparent=False)
    plt.close('all')

# def meanConfidenceInterval(data,confidence=95):
#     a = 1.0*np.array(data)
#     print 'array of data', a, 'data', data
#     n = len(a)
#     m, se = np.mean(a), sp.stats.sem(a)
#     h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
#     return m, m-h, m+h

def plotLiklAndDiff(pos_likl,pos_diff, s_range, font, filename):
    fig = plt.figure(figsize=(6,4))
    ax= fig.add_subplot(111)
    ax.set_facecolor('#f2f2f3')
    plt.plot(s_range,pos_diff,'-')
    plt.title("Difference score evolution", size=font+2)
    plt.ylabel("Difference", size=font+1)
    plt.xlabel("Number of samples", size=font+1)
    plt.xlim(np.amin(s_range),np.amax(s_range))
    plt.savefig('%s/evol_diff.png' % (filename), bbox_inches='tight',dpi=300,transparent=False)
    plt.close()

    fig = plt.figure(figsize=(6,4))
    ax= fig.add_subplot(111)
    ax.set_facecolor('#f2f2f3')
    plt.plot(s_range,pos_likl,'-')
    plt.title("Likelihood evolution", size=font+2)
    plt.ylabel("Log likelihood", size=font+1)
    plt.xlabel("Number of samples", size=font+1)
    # plt.xlim(0,len(pos_likl)-1)
    plt.xlim(np.amin(s_range),np.amax(s_range))
    plt.savefig('%s/evol_likl.png' % (filename), bbox_inches='tight',dpi=300,transparent=False)
    plt.close()

def plotParameters(fname, s_range, sedsim, flowsim,communities, 
    pos_m, pos_ax, pos_ay, true_m, true_ax, true_ay,
    pos_sed1, pos_sed2, pos_sed3, pos_sed4, true_sed,
    pos_flow1, pos_flow2, pos_flow3, pos_flow4,true_flow):
    nb_bins=30
    # s_range = np.arange(0,pos_m.shape[0],1)
    font = 10
    width = 1

    #########################
    #   MALTHUS PARAMETER   #
    #########################
    mmin, mmax = min(pos_m), max(pos_m)
    mspace = np.linspace(mmin,mmax,len(pos_m))
    # mm,ms = stats.norm.fit(pos_m)
    # pdf_m = stats.norm.pdf(mspace,mm,ms)
    mmean=np.mean(pos_m)
    mmedian=np.median(pos_m)
    mmode,count=stats.mode(pos_m)

    fig = plt.figure(figsize=(6,8))
    ax = fig.add_subplot(111)
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    ax.set_title(' Malthusian Parameter', fontsize= font+2)#, y=1.02)
    
    ax1 = fig.add_subplot(211)
    ax1.set_facecolor('#f2f2f3')
    ax1.hist(pos_m, bins=25, alpha=0.5, facecolor='sandybrown', normed=True)
    ax1.axvline(true_m, linestyle='-', color='black', linewidth=1,label='True value')
    ax1.grid(True)
    ax1.set_ylabel('Frequency',size=font+1)
    ax1.set_xlabel(r'$\varepsilon$', size=font+1)
    
    ax2 = fig.add_subplot(212)
    ax2.set_facecolor('#f2f2f3')
    ax2.plot(s_range,pos_m,linestyle='-', linewidth=width, color='k', label=None)
    ax2.set_title(r'Trace of $\varepsilon$',size=font+2)
    ax2.set_xlabel('Samples',size=font+1)
    ax2.set_ylabel(r'$\varepsilon$', size=font+1)
    ax2.set_xlim([np.amin(s_range),np.amax(s_range)])
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    plt.savefig('%s/malthus.png'% (fname), bbox_inches='tight', dpi=300, transparent=False)
    plt.close()
    
    #########################
    #    COMMUNITY MATRIX   #
    #########################
    a1min, a1max = min(pos_ax), max(pos_ax)
    a1space = np.linspace(a1min,a1max,len(pos_ax))
    # a1m,a1s = stats.norm.fit(pos_ax)
    # pdf_a1 = stats.norm.pdf(a1space,a1m,a1s)
    a2min, a2max = min(pos_ay), max(pos_ay)
    a2space = np.linspace(a2min,a2max,len(pos_ay))
    # a2m,a2s = stats.norm.fit(pos_ay)
    # pdf_a2 = stats.norm.pdf(a2space,a2m,a2s)
    a1min=a1min
    a1max=a1max
    a1mean=np.mean(pos_ax)
    a1median=np.median(pos_ax)
    a1mode,count=stats.mode(pos_ax)
    a2min=a2min
    a2max=a2max
    a2mean=np.mean(pos_ay)
    a2median=np.median(pos_ay)
    a2mode,count=stats.mode(pos_ay)

    ###################
    #   MAIN DIAGONAL #
    ###################
    fig = plt.figure(figsize=(6,8))
    ax = fig.add_subplot(111)
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    ax.set_title('Community Interaction Matrix Parameters', fontsize= font+2, y=1.03)
    ax1 = fig.add_subplot(211)
    ax1.set_facecolor('#f2f2f3')
    ax1.hist(pos_ax, bins=25, alpha=0.5, facecolor='mediumaquamarine', normed=True)
    ax1.axvline(true_ax, linestyle='-', color='black', linewidth=1,label='True value')
    ax1.grid(True)
    ax1.set_ylabel('Frequency',size=font+1)
    ax1.set_title(r'Main diagonal value ($\alpha_{m}$)',size=font+2)
    ax1.set_xlabel(r'$\alpha_{m}$', size=font+1)
    ax2 = fig.add_subplot(212)
    ax2.set_facecolor('#f2f2f3')
    ax2.plot(s_range,pos_ax,linestyle='-', linewidth=width, color='k', label=None)
    ax2.set_xlabel('Samples',size=font+1)
    ax2.set_ylabel(r'$\alpha_{m}$', size=font+1)
    ax2.set_title(r'Trace of $\alpha_{m}$',size=font+2)
    ax2.set_xlim([np.amin(s_range),np.amax(s_range)])
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    plt.savefig('%s/comm_ax.png'% (fname),bbox_inches='tight', dpi=300,transparent=False)
    plt.close()

    #############################
    #   SUPER-/SUB-DIAGONALS    #
    ############################# 
    fig = plt.figure(figsize=(6,8))
    ax = fig.add_subplot(111)
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    ax.set_title('Community Interaction Matrix Parameters', fontsize= font+2, y=1.03)
    ax1 = fig.add_subplot(211)
    ax1.set_facecolor('#f2f2f3')
    ax1.hist(pos_ay, bins=25, alpha=0.5, facecolor='mediumaquamarine', normed=True)
    ax1.axvline(true_ay, linestyle='-', color='black', linewidth=1,label='True value')
    ax1.grid(True)
    ax1.set_title(r'Super- and sub-diagonal values ($\alpha_{s}$)',size=font+2)
    ax1.set_xlabel(r'$\alpha_{s}$', size=font+1)
    ax1.set_ylabel('Frequency',size=font+1)
    ax2 = fig.add_subplot(212)
    ax2.set_facecolor('#f2f2f3')
    ax2.plot(s_range,pos_ay,linestyle='-', linewidth=width, color='k', label=None)
    ax2.set_title(r'Trace of $\alpha_{s}$',size=font+2)
    ax2.set_xlabel('Samples',size=font+1)
    ax2.set_ylabel(r'$\alpha_{s}$', size=font+1)
    ax2.set_xlim([np.amin(s_range),np.amax(s_range)])
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    plt.savefig('%s/comm_ay.png' % (fname), dpi=300, bbox_inches='tight',transparent=False)
    plt.close()

    ############################
    # WRITE SUMMARY STATISTICS #
    ############################
    if not os.path.isfile(('%s/summ_stats.csv' % (fname))):
        with file(('%s/summ_stats.csv' % (fname)),'wb') as outfile:
            writer = csv.writer(outfile, delimiter=',')
            writer.writerow(["Param","Min", "Max","Mean", "Median", "Mode"])
            writer.writerow(["Malth", mmin,mmax,mmean,mmedian, float(mmode)])
            writer.writerow(["A main", a1min,a1max,a1mean,a1median,float(a1mode)])
            writer.writerow(["A super/sub", a2min,a2max,a2mean,a2median,float(a2mode)])
    

    #############################################
    #   SEDIMENT AND FLOW RESPONSE THRESHOLDS   #
    #############################################

    a_labels = ['Shallow', 'Moderate-deep', 'Deep']
    
    sed1_mu, sed1_ub, sed1_lb, sed2_mu, sed2_ub, sed2_lb, sed3_mu, sed3_ub, sed3_lb, sed4_mu, sed4_ub, sed4_lb = (np.zeros(communities) for i in range(12))
    if ((sedsim != False)):
        for a in range(communities):
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
            sed1_mode,count=stats.mode(pos_sed1[:,a])
            sed2_mode,count=stats.mode(pos_sed2[:,a])
            sed3_mode,count=stats.mode(pos_sed3[:,a])
            sed4_mode,count=stats.mode(pos_sed4[:,a])
            with file(('%s/summ_stats.csv' % (fname)),'ab') as outfile:
                writer = csv.writer(outfile, delimiter=',')
                writer.writerow([a_labels[a]])
                writer.writerow(["Sed 1", sed1_min,sed1_max,sed1_mu_,sed1_med, float(sed1_mode)])
                writer.writerow(["Sed 1", sed2_min,sed2_max,sed2_mu_,sed2_med, float(sed2_mode)])
                writer.writerow(["Sed 3", sed3_min,sed3_max,sed3_mu_,sed3_med, float(sed3_mode)])
                writer.writerow(["Sed 4", sed4_min,sed4_max,sed4_mu_,sed4_med, float(sed4_mode)])

            cy = [0,100,100,0]
            cmu = [sed1_mu[a], sed2_mu[a], sed3_mu[a], sed4_mu[a]]
            c_lb = [sed1_mu[a]-sed1_lb[a], sed2_mu[a]-sed2_lb[a], sed3_mu[a]-sed3_lb[a], sed4_mu[a]-sed4_lb[a]]
            c_ub = [sed1_ub[a]-sed1_mu[a], sed2_ub[a]-sed2_mu[a], sed3_ub[a]-sed3_mu[a], sed4_ub[a]-sed4_mu[a]]
            
            fig = plt.figure(figsize=(6,4))
            ax = fig.add_subplot(111)
            ax.set_facecolor('#f2f2f3')
            ax.plot(true_sed[a,:], cy, linestyle='--', linewidth=width, marker='.',color='k', label='Synthetic data')
            ax.plot(cmu, cy, linestyle='-', linewidth=width,marker='.', color='sandybrown', label='Mean')
            ax.errorbar(cmu[0:2],cy[0:2],xerr=[c_lb[0:2],c_ub[0:2]],capsize=5,elinewidth=1, color='darksalmon',mfc='darksalmon',fmt='.',label=None)
            ax.errorbar(cmu[2:4],cy[2:4],xerr=[c_lb[2:4],c_ub[2:4]],capsize=5,elinewidth=1, color='sienna',mfc='sienna',fmt='.',label=None)
            plt.title('Sediment exposure threshold function\n(%s assemblage)' % (a_labels[a]), size=font+2, y=1.06)
            plt.ylabel('Proportion of maximum growth rate [%]',size=font+1)
            plt.xlabel('Sediment input [m/year]',size=font+1)
            plt.ylim(-2.,110)
            lgd = plt.legend(frameon=False, prop={'size':font+1}, bbox_to_anchor = (1.,0.2))
            plt.savefig('%s/sediment_response_%s.png' % (fname, a+1), bbox_extra_artists=(lgd,),bbox_inches='tight',dpi=300,transparent=False)
            plt.close()

            ####################################################
            #   TRACE PLOTS OF SEDIMENT EXPOSURE THRESHOLDS    #
            ####################################################

            fig = plt.figure(figsize=(12,8))
            ax = fig.add_subplot(111)
            ax.spines['top'].set_color('none')
            ax.spines['bottom'].set_color('none')
            ax.spines['left'].set_color('none')
            ax.spines['right'].set_color('none')
            ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
            ax.set_title('Trace plots of sediment input thresholds: %s ' % a_labels[a], fontsize= font+2, y=1.03)

            ax1 = fig.add_subplot(221)
            ax1.set_facecolor('#f2f2f3')
            ax1.plot(s_range,pos_sed1[:,a], linestyle='-', color='k', linewidth=width)
            ax1.set_xlabel('Samples', size=font+1)
            ax1.set_ylabel(r'$f^1_{sed}$', size=font+1)
            ax1.set_xlim([np.amin(s_range),np.amax(s_range)])

            ax2 = fig.add_subplot(222)
            ax2.set_facecolor('#f2f2f3')
            ax2.plot(s_range,pos_sed2[:,a], linestyle='-', color='k', linewidth=width)
            ax2.set_xlabel('Samples', size=font+1)
            ax2.set_ylabel(r'$f^2_{sed}$', size=font+1)
            ax2.set_xlim([np.amin(s_range),np.amax(s_range)])

            ax3 = fig.add_subplot(223)
            ax3.set_facecolor('#f2f2f3')
            ax3.plot(s_range,pos_sed3[:,a], linestyle='-', color='k', linewidth=width)
            ax3.set_xlabel('Samples', size=font+1)
            ax3.set_ylabel(r'$\it{f}^3_{sed}$', size=font+1)
            ax3.set_xlim([np.amin(s_range),np.amax(s_range)])

            ax4 = fig.add_subplot(224)
            ax4.set_facecolor('#f2f2f3')
            ax4.plot(s_range,pos_sed4[:,a], linestyle='-', color='k', linewidth=width)
            ax4.set_xlabel('Samples', size=font+1)
            ax4.set_ylabel(r'$\it{f}^4_{sed}$', size=font+1)
            ax4.set_xlim([np.amin(s_range),np.amax(s_range)])                
            
            fig.tight_layout()
            plt.savefig('%s/sed_threshold_trace_%s.png'% (fname, a),bbox_inches='tight', dpi=300,transparent=False)
            plt.close()


    flow1_mu, flow1_ub,flow1_lb, flow2_mu, flow2_ub,flow2_lb, flow3_mu, flow3_ub,flow3_lb, flow4_mu, flow4_ub,flow4_lb = (np.zeros(communities) for i in range(12))
    if (flowsim != False):
        for a in range(communities):
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
            flow1_mode,count= stats.mode(pos_flow1[:,a])
            flow2_mode,count= stats.mode(pos_flow2[:,a])
            flow3_mode,count= stats.mode(pos_flow3[:,a])
            flow4_mode,count= stats.mode(pos_flow4[:,a])
            with file(('%s/summ_stats.csv' % (fname)),'ab') as outfile:
                writer = csv.writer(outfile, delimiter=',')
                writer.writerow([a_labels[a]])
                writer.writerow(["Flow 1", flow1_min,flow1_max,flow1_mu_,flow1_med, float(flow1_mode)])
                writer.writerow(["Flow 1", flow2_min,flow2_max,flow2_mu_,flow2_med, float(flow2_mode)])
                writer.writerow(["Flow 3", flow3_min,flow3_max,flow3_mu_,flow3_med, float(flow3_mode)])
                writer.writerow(["Flow 4", flow4_min,flow4_max,flow4_mu_,flow4_med, float(flow4_mode)])


            cy = [0,100,100,0]
            cmu = [flow1_mu[a], flow2_mu[a], flow3_mu[a], flow4_mu[a]]
            c_lb = [flow1_mu[a]-flow1_lb[a], flow2_mu[a]-flow2_lb[a], flow3_mu[a]-flow3_lb[a], flow4_mu[a]-flow4_lb[a]]
            c_ub = [flow1_ub[a]-flow1_mu[a], flow2_ub[a]-flow2_mu[a], flow3_ub[a]-flow3_mu[a], flow4_ub[a]-flow4_mu[a]]

            
            fig = plt.figure(figsize=(6,4))
            ax = fig.add_subplot(111)
            ax.set_facecolor('#f2f2f3')
            ax.plot(true_flow[a,:], cy, linestyle='--', linewidth=width, marker='.', color='k',label='Synthetic data')
            ax.plot(cmu, cy, linestyle='-', linewidth=width, marker='.', color='steelblue', label='Mean')
            ax.errorbar(cmu[0:2],cy[0:2],xerr=[c_lb[0:2],c_ub[0:2]],capsize=5,elinewidth=1,color='lightsteelblue',mfc='lightsteelblue',fmt='.',label=None)
            ax.errorbar(cmu[2:4],cy[2:4],xerr=[c_lb[2:4],c_ub[2:4]],capsize=5,elinewidth=1,color='lightslategrey',mfc='lightslategrey',fmt='.',label=None)
            plt.title('Hydrodynamic energy exposure threshold function\n(%s assemblage)' % (a_labels[a]), size=font+2, y=1.06)
            plt.ylabel('Proportion of maximum growth rate [%]', size=font+1)
            plt.xlabel('Fluid flow [m/sec]', size=font+1)
            plt.ylim(-2.,110.)
            lgd = plt.legend(frameon=False, prop={'size':font+1}, bbox_to_anchor = (1.,0.2))
            plt.savefig('%s/flow_response_%s.png' % (fname, a+1),  bbox_extra_artists=(lgd,), bbox_inches='tight',dpi=300,transparent=False)
            plt.close()

            #########################################################
            #   TRACE PLOTS OF FLOW VELOCITY EXPOSURE THRESHOLDS    #
            #########################################################
            fig = plt.figure(figsize=(12,8))
            ax = fig.add_subplot(111)
            ax.spines['top'].set_color('none')
            ax.spines['bottom'].set_color('none')
            ax.spines['left'].set_color('none')
            ax.spines['right'].set_color('none')
            ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
            ax.set_title('Trace plots of flow velocity Thresholds: %s ' % a_labels[a], fontsize= font+2, y=1.03)

            ax1 = fig.add_subplot(221)
            ax1.set_facecolor('#f2f2f3')
            ax1.plot(s_range,pos_flow1[:,a], linestyle='-', color='k', linewidth=width)
            ax1.set_xlabel('Samples', size=font+1)
            ax1.set_ylabel(r'$f^1_{flow}$', size=font+1)
            ax1.set_xlim([np.amin(s_range),np.amax(s_range)])

            ax2 = fig.add_subplot(222)
            ax2.set_facecolor('#f2f2f3')
            ax2.plot(s_range,pos_flow2[:,a], linestyle='-', color='k', linewidth=width)
            ax2.set_xlabel('Samples', size=font+1)
            ax2.set_ylabel(r'$f^2_{flow}$', size=font+1)
            ax2.set_xlim([np.amin(s_range),np.amax(s_range)])

            ax3 = fig.add_subplot(223)
            ax3.set_facecolor('#f2f2f3')
            ax3.plot(s_range,pos_flow3[:,a], linestyle='-', color='k', linewidth=width)
            ax3.set_xlabel('Samples', size=font+1)
            ax3.set_ylabel(r'$\it{f}^3_{flow}$', size=font+1)
            ax3.set_xlim([np.amin(s_range),np.amax(s_range)])

            ax4 = fig.add_subplot(224)
            ax4.set_facecolor('#f2f2f3')
            ax4.plot(s_range,pos_flow4[:,a], linestyle='-', color='k', linewidth=width)
            ax4.set_xlabel('Samples', size=font+1)
            ax4.set_ylabel(r'$\it{f}^4_{flow}$', size=font+1)
            ax4.set_xlim([np.amin(s_range),np.amax(s_range)])                
            
            fig.tight_layout()
            plt.savefig('%s/flow_threshold_trace_%s.png'% (fname, a),bbox_inches='tight', dpi=300,transparent=False)
            plt.close()


# Script to make box plots 
def boxPlots(communities, pos_v, sedsim, flowsim, font, width, filename):
    if communities == 3:
        if ((sedsim == True) and (flowsim == False)) or ((sedsim == False) and (flowsim == True)):
            v_glve = np.zeros((pos_v.shape[0],3))
            v_glve[:,0:3] = pos_v[:,12:15]

            com_1=[0,3,6,9]
            com_2=[1,4,7,10]
            com_3=[2,5,8,11]
            new_v = np.zeros((pos_v.shape[0],12))
            
            for i in range(4):
                new_v[:,i] = pos_v[:,com_1[i]]
            for i in range(4,8):
                new_v[:,i] = pos_v[:,com_2[i-4]]
            for i in range(8,12):
                new_v[:,i] = pos_v[:,com_3[i-8]]

            mpl_fig = plt.figure(figsize=(8,4))
            ax = mpl_fig.add_subplot(111)
            ax.spines['top'].set_color('none')
            ax.spines['bottom'].set_color('none')
            ax.spines['left'].set_color('none')
            ax.spines['right'].set_color('none')
            ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
            ax.set_title('Posterior values', fontsize= font+2, y=1.02)
            ax1 = mpl_fig.add_subplot(121)
            ax1.boxplot(v_glve)
            ax1.set_xlabel('GLVE parameters', size=font+2)
            ax2 = mpl_fig.add_subplot(122)
            ax2.boxplot(new_v)
            ax2.set_xlabel('Assemblage exposure thresholds', size=font+2)
            plt.savefig('%s/v_pos_boxplot.png'% (filename), dpi=300,transparent=False)
            plt.close()
        elif ((sedsim == True) and (flowsim == True)):
            v_glve = np.zeros((pos_v.shape[0],3))
            v_glve[:,0:3] = pos_v[:,24:27]
            com_1=[0,3,6,9,12,15,18,21]
            com_2=[1,4,7,10,13,16,19,22]
            com_3=[2,5,8,11,14,17,20,23]

            v_sed = np.zeros((pos_v.shape[0],12))
            v_flow = np.zeros((pos_v.shape[0],12))
            for i in range(4):
                v_sed[:,i] = pos_v[:,com_1[i]]
            for i in range(4,8):
                v_sed[:,i] = pos_v[:,com_2[i-4]]
            for i in range(8,12):
                v_sed[:,i] = pos_v[:,com_3[i-8]]

            for i in range(4):
                v_flow[:,i] = pos_v[:,com_1[i+4]]
            for i in range(4,8):
                v_flow[:,i] = pos_v[:,com_2[i]]
            for i in range(8,12):
                v_flow[:,i] = pos_v[:,com_3[i-4]]


            mpl_fig = plt.figure(figsize=(14,4))
            ax = mpl_fig.add_subplot(111)
            ax.spines['top'].set_color('none')
            ax.spines['bottom'].set_color('none')
            ax.spines['left'].set_color('none')
            ax.spines['right'].set_color('none')
            ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
            ax.set_title('Posterior values', fontsize= font+2, y=1.02)
            ax1 = mpl_fig.add_subplot(131)
            ax1.boxplot(v_glve)
            ax1.set_xlabel('GLVE parameters', size=font+2)
            ax2 = mpl_fig.add_subplot(132)
            ax2.boxplot(v_sed)
            ax2.set_xlabel('Assemblage sediment exposure thresholds', size=font+2)
            ax3 = mpl_fig.add_subplot(133)
            ax3.boxplot(v_flow)
            ax3.set_xlabel('Assemblage flow exposure thresholds', size=font+2)
            plt.savefig('%s/v_pos_boxplot.png'% (filename), dpi=300,transparent=False)
            # mpl_fig = plt.figure(figsize=(10,4))
            # ax = mpl_fig.add_subplot(111)
            # ax.boxplot(new_v)
            # ax.set_ylabel('Posterior values')
            # ax.set_xlabel('Input vector')
            # plt.title("Boxplot of posterior distribution \nfor GLVE and threshold parameters", size=font+2)
            # plt.savefig('%s/v_pos_boxplot.pdf'% (filename), dpi=300)
            # # plt.savefig('%s/v_pos_boxplot.svg'% (filename), format='svg', dpi=300)
            plt.close()
    elif communities == 6:
        if ((sedsim == True) and (flowsim == False)) or ((sedsim == False) and (flowsim == True)):
            v_glve = np.zeros((pos_v.shape[0],3))
            v_glve[:,0:3] = pos_v[:,24:27]

            com_1=[0,6,12,18]
            com_2=[1,7,13,19]
            com_3=[2,8,14,20]
            com_4=[3,9,15,21]
            com_5=[4,10,16,22]
            com_6=[5,11,17,23]
            new_v = np.zeros((pos_v.shape[0],24))
            for i in range(4):
                new_v[:,i] = pos_v[:,com_1[i]]
            for i in range(4,8):
                new_v[:,i] = pos_v[:,com_2[i-4]]
            for i in range(8,12):
                new_v[:,i] = pos_v[:,com_3[i-8]]
            for i in range(12,16):
                new_v[:,i] = pos_v[:,com_4[i-12]]
            for i in range(16,20):
                new_v[:,i] = pos_v[:,com_5  [i-16]]
            for i in range(20,24):
                new_v[:,i] = pos_v[:,com_6[i-20]]

            mpl_fig = plt.figure(figsize=(8,4))
            ax = mpl_fig.add_subplot(111)
            ax.spines['top'].set_color('none')
            ax.spines['bottom'].set_color('none')
            ax.spines['left'].set_color('none')
            ax.spines['right'].set_color('none')
            ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
            ax.set_title('Posterior values', fontsize= font+2, y=1.02)
            ax1 = mpl_fig.add_subplot(121)
            ax1.boxplot(v_glve)
            ax1.set_xlabel('GLVE parameters', size=font+2)
            ax2 = mpl_fig.add_subplot(122)
            ax2.boxplot(new_v)
            ax2.set_xlabel('Assemblage exposure thresholds', size=font+2)
            plt.savefig('%s/v_pos_boxplot.png'% (filename), dpi=300,transparent=False)
            # for i in range(3,7):
            #     new_v[:,i] = pos_v[:,com_1[i-3]]
            # for i in range(7,11):
            #     new_v[:,i] = pos_v[:,com_2[i-7]]
            # for i in range(11,15):
            #     new_v[:,i] = pos_v[:,com_3[i-11]]

            # mpl_fig = plt.figure(figsize=(6,4))
            # ax = mpl_fig.add_subplot(111)
            # print 'pos_v.size',pos_v.size, 'pos_v.shape',pos_v.shape
            # ax.boxplot(new_v)
            # ax.set_ylabel('Posterior values')
            # ax.set_xlabel('Input vector')
            # plt.title("Boxplot of posterior distribution \nfor GLVE and threshold parameters", size=font+2)
            # plt.savefig('%s/v_pos_boxplot.pdf'% (filename), dpi=300)
            # plt.savefig('%s/v_pos_boxplot.svg'% (filename), format='svg', dpi=300)
            plt.close()
        elif ((sedsim == True) and (flowsim == True)):
            v_glve = np.zeros((pos_v.shape[0],3))
            v_glve[:,0:3] = pos_v[:,48:51]

            com_1=[0,6,12,18,24,30,36,42]
            com_2=[1,7,13,19,25,31,37,43]
            com_3=[2,8,14,20,26,32,38,44]
            com_4=[3,9,15,21,27,33,39,45]
            com_5=[4,10,16,22,28,34,40,46]
            com_6=[5,11,17,23,29,35,41,47]
            v_sed = np.zeros((pos_v.shape[0],24))
            v_flow = np.zeros((pos_v.shape[0],24))

            for i in range(4):
                v_sed[:,i] = pos_v[:,com_1[i]]
            for i in range(4,8):
                v_sed[:,i] = pos_v[:,com_2[i-4]]
            for i in range(8,12):
                v_sed[:,i] = pos_v[:,com_3[i-8]]
            for i in range(12,16):
                v_sed[:,i] = pos_v[:,com_4[i-12]]
            for i in range(16,20):
                v_sed[:,i] = pos_v[:,com_5[i-16]]
            for i in range(20,24):
                v_sed[:,i] = pos_v[:,com_6[i-20]]

            for i in range(4):
                v_flow[:,i] = pos_v[:,com_1[i+4]]
            for i in range(4,8):
                v_flow[:,i] = pos_v[:,com_2[i]]
            for i in range(8,12):
                v_flow[:,i] = pos_v[:,com_3[i-4]]
            for i in range(12,16):
                v_flow[:,i] = pos_v[:,com_4[i-8]]
            for i in range(16,20):
                v_flow[:,i] = pos_v[:,com_5[i-12]]
            for i in range(20,24):
                v_flow[:,i] = pos_v[:,com_6[i-16]]


            mpl_fig = plt.figure(figsize=(14,4))
            ax = mpl_fig.add_subplot(111)
            ax.spines['top'].set_color('none')
            ax.spines['bottom'].set_color('none')
            ax.spines['left'].set_color('none')
            ax.spines['right'].set_color('none')
            ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
            ax.set_title('Posterior values', fontsize= font+2, y=1.02)
            ax1 = mpl_fig.add_subplot(131)
            ax1.boxplot(v_glve)
            ax1.set_xlabel('GLVE parameters', size=font+2)
            ax2 = mpl_fig.add_subplot(132)
            ax2.boxplot(v_sed)
            ax2.set_xlabel('Assemblage sediment exposure thresholds', size=font+2)
            ax3 = mpl_fig.add_subplot(133)
            ax3.boxplot(v_flow)
            ax3.set_xlabel('Assemblage flow exposure thresholds', size=font+2)
            plt.savefig('%s/v_pos_boxplot.png'% (filename), dpi=300,transparent=False)
            # mpl_fig = plt.figure(figsize=(10,4))
            # ax = mpl_fig.add_subplot(111)
            # ax.boxplot(new_v)
            # ax.set_ylabel('Posterior values')
            # ax.set_xlabel('Input vector')
            # plt.title("Boxplot of posterior distribution \nfor GLVE and threshold parameters", size=font+2)
            # plt.savefig('%s/v_pos_boxplot.pdf'% (filename), dpi=300)
            # # plt.savefig('%s/v_pos_boxplot.svg'% (filename), format='svg', dpi=300)
            plt.close()