##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
##                                                                                   ##
##  This file forms part of the pyReefCore synthetic coral reef core model app.      ##
##                                                                                   ##
##  For full license and copyright information, please refer to the LICENSE.md file  ##
##  located at the project root, or contact the authors.                             ##
##                                                                                   ##
##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
"""
Here we set plotting functions used to visualise pyReef dataset.
"""

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import gridspec
import matplotlib.pyplot as plt

from scipy.ndimage.filters import gaussian_filter

import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

class modelPlot():
    """
    Class for plotting outputs from pyReef model.
    """

    def __init__(self, input=None):
        """
        Constructor.
        """
        self.names = np.empty(input.speciesNb+1, dtype="S14")
        self.names[:input.speciesNb] = input.speciesName
        self.names[-1] = 'Sediment'
        self.step = int(input.laytime/input.tCarb)
        self.pop = None
        self.timeCarb = None
        self.depth = None
        self.accspace = None
        self.timeLay = None
        self.surf = None
        self.sedH = None
        self.sealevel = None
        self.sedinput = None
        self.waterflow = None

        return

    def accomodationTime(self, colors=None, size=(10,5), font=9, dpi=80, fname=None):
        """
        This function estimates the accomodation space through time.

        Parameters
        ----------

        variable : colors
            Matplotlib color map to use

        variable : size
            Figure size

        variable : font
            Figure font size

        variable : dpi
            Figure resolution

        variable : fname
            Save PNG filename.
        """

        matplotlib.rcParams.update({'font.size': font})

        # Define figure size
        fig, ax = plt.subplots(1,figsize=size, dpi=dpi)
        ax.set_facecolor('#f2f2f3')

        # Plotting curves
        ax.plot(self.timeCarb[:-2], self.accspace[:-2], linewidth=3,c=colors)

        # Legend, title and labels
        plt.grid()
        plt.xlabel('Time [y]',size=font+2)
        plt.ylabel('accomodation space [m]',size=font+2)
        plt.xlim(0., self.timeCarb.max())


        ttl = ax.title
        ttl.set_position([.5, 1.05])
        plt.title('Accomodation space evolution through time',size=font+3)
        # plt.show()
        plt.close()

        if fname is not None:
            fig.savefig(fname, bbox_inches='tight')

        return

    def speciesTime(self, colors=None, size=(10,5), font=9, dpi=80, fname=None):
        """
        This function estimates the coral growth based on newly computed population.

        Parameters
        ----------

        variable : colors
            Matplotlib color map to use

        variable : size
            Figure size

        variable : font
            Figure font size

        variable : dpi
            Figure resolution

        variable : fname
            Save PNG filename.
        """

        matplotlib.rcParams.update({'font.size': font})

        # Define figure size
        fig, ax = plt.subplots(1,figsize=size, dpi=dpi)
        ax.set_facecolor('#f2f2f3')

        # Plotting curves
        for s in range(len(self.pop)):
            ax.plot(self.timeCarb, self.pop[s,:], label=self.names[s],linewidth=3,c=colors[s])

        # Legend, title and labels
        plt.grid()
        lgd = plt.legend(frameon=False,loc=4,prop={'size':font+1}, bbox_to_anchor=(1.2,-0.02))
        plt.xlabel('Simulation time [y]',size=font+2)
        plt.ylabel('Population',size=font+2)
        plt.ylim(0., int(self.pop.max())+1)
        plt.xlim(0., self.timeCarb.max())


        ttl = ax.title
        ttl.set_position([.5, 1.05])
        plt.title('Evolution of species populations with time',size=font+3)
        # plt.show()

        if fname is not None:
            fig.savefig(fname, bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.close()
        return

    def speciesDepth(self, colors=None, size=(10,5), font=9, dpi=80, fname=None):
        """
        Variation of coral growth with depth

        Parameters
        ----------

        variable : colors
            Matplotlib color map to use

        variable : size
            Figure size

        variable : font
            Figure font size

        variable : dpi
            Figure resolution

        variable : fname
            Save PNG filename.
        """

        matplotlib.rcParams.update({'font.size': font})

        # Define figure size
        fig, ax = plt.subplots(1,figsize=size, dpi=dpi)
        ax.set_facecolor('#f2f2f3')

        # Plotting curves
        bottom = self.surf + self.depth.sum()
        d = bottom - np.cumsum(self.depth)
        for s in range(len(self.pop)):
            ax.plot(d, self.pop[s,::self.step], label=self.names[s],linewidth=3,c=colors[s])

        # Legend, title and labels
        plt.grid()
        lgd = plt.legend(frameon=False,loc=4,prop={'size':font+1}, bbox_to_anchor=(1.2,-0.02))
        plt.xlabel('Depth [m]',size=font+2)
        plt.ylabel('Population',size=font+2)
        plt.ylim(0., int(self.pop.max())+1)
        plt.xlim(d.max(), d.min())

        ttl = ax.title
        ttl.set_position([.5, 1.05])
        plt.title('Evolution of species populations with depth',size=font+3)
        # plt.show()

        if fname is not None:
            fig.savefig(fname, bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.close()
        return
    
    def writeCore(self, communities):
        # write to a text file a column of depth intervals and the assemblage at each depth interval 
        p2 = np.zeros((self.sedH.shape))
        ids = np.where(self.depth[:-1]>0)[0]
        p2[:,ids] = self.sedH[:,ids]/self.depth[ids]
        bottom = self.surf + self.depth[:-1].sum()
        d = bottom - np.cumsum(self.depth[:-1])
        # print 'depth', d, 'd.size', d.size
        increment = 0.2
        core_depths = np.arange(0,bottom, increment)
        depth_incrementor = core_depths[-1] #13.5
        # print 'core_depths', core_depths
        counter = 0
        output_core = np.zeros((communities+1, core_depths.size))
        id_prev = 999
        idx = 0
        for i in range (0,core_depths.size):
            # print 'i', i
            if not ((np.sum(p2[:,i]) == 0) and (depth_incrementor == -0.1)): 
                # as long as there is growth and the core is not filled
                # if not ((np.sum(p2[0:communities,i]) == 0) and (p2[communities,i] == 1)):
                idx = (np.abs(d-depth_incrementor)).argmin()
                if not ((idx == d.size-1) and (idx == id_prev)):
                    # as long as idx has a corresponding value that is not the end of the core
                    # print 'current depth_incrementor', depth_incrementor, 'm'
                    output_core[:,counter] = p2[:,idx]
                    # print 'p2[:,idx]    output_core[:,counter]\n', output_core[:,counter]
                    counter += 1
                    # print 'counter: ', counter
                    depth_incrementor -= increment
                    # print 'next depth_incrementor', depth_incrementor, 'm \n\n'
                    id_prev=idx
        return p2, output_core, core_depths

    def convertDepthStructure(self, communities, core_depths):
        ids = np.where(self.depth[:-1]>0)[0]
        p2 = np.zeros((self.sedH.shape))
        p2[:,ids] = self.sedH[:,ids]/self.depth[ids]

        bottom = self.surf + self.depth[:-1].sum()
        d = bottom - np.cumsum(self.depth[:-1])

        counter = 0
        max_depth=np.amax(core_depths)
        depth_incrementor = max_depth
        output_core = np.zeros((communities+1, core_depths.shape[0]))
        depth_increment = 0.2 
        id_prev = 999
        idx = 0

        for i in range (0,len(core_depths)):
            if not ((np.sum(p2[:,i]) == 0) and (depth_incrementor == -0.1)): 
            # as long as there is growth and the core is not filled
                idx = (np.abs(d-depth_incrementor)).argmin()
                if not ((idx == d.size-1) and (idx == id_prev)):
                    # as long as idx has a corresponding value that is not the end of the core
                    output_core[:,counter] = p2[:,idx]
                    counter += 1
                    depth_incrementor -= depth_increment
                    id_prev=idx
        return output_core

    def convertTimeStructure(self):
        ids = np.where(self.depth[:-1]>0)[0] #ids are all depth intervals where growth is continuous, before it stops
        propn_asmb_time = np.zeros((self.sedH.shape)) #proportion of each coral assemblage
        propn_asmb_time[:,ids] = self.sedH[:,ids]/self.depth[ids]
        # print 'propn_asmb_time', propn_asmb_time.T[:40,:]
        # print 'timelay shape', self.timeLay.shape
        # print 'self.pop', self.pop.shape, self.pop[1:40,:]
        # print 'self.depth', self.depth.size, self.depth #171
        # print 'accspace', self.accspace.shape, self.accspace #341
        # print 'timlay', self.timeLay.shape, self.timeLay #171
        # print 'surf', self.surf.shape, self.surf
        # # print 'sedH', self.sedH.shape, self.sedH #4*171
        return propn_asmb_time.T, self.timeLay

    def getTimePlotParameters(self, colors=None):
        return self.timeCarb, self.pop, self.names
    
    def drawCore(self, depthext = None, thext = None, propext = [0.,1.], lwidth = 3,
                 colsed=None, coltime=None, size=(8,10), font=8, dpi=80, figname=None,
                 filename = None, sep = '\t'):
        """Plot core evolution

        Parameters
        ----------

        variable : depthext
            Core depth extension to plot [m]

        variable : thext
            Core thickness range to plot [m]

        variable : propext
            Core ranging proportion to plot between [0,1.]

        variable : lwidth
            Figure lines width

        variable : colsed
            Matplotlib color map to use for production plots

        variable : coltime
            Matplotlib color map to use for time layer plots

        variable : size
            Figure size

        variable : font
            Figure font size

        variable : dpi
            Figure resolution

        variable : figname
            Save figure (the type of file needs to be provided e.g. .png or .pdf).

        variable : filename
            Save model output to a CSV file.

        variable : sep
            Separator used in the CSV file.
        """
        p1 = self.sedH[:,:-1] #thicknesses of each coral assemblage
        ids = np.where(self.depth[:-1]>0)[0] #ids are all depth intervals where growth is continuous, before it stops
        p2 = np.zeros((self.sedH.shape)) #proportion of each coral assemblage
        p3 = np.zeros((self.sedH.shape))

        p2[:,ids] = self.sedH[:,ids]/self.depth[ids]
        p3[:,ids] = np.cumsum(self.sedH[:,ids]/self.depth[ids],axis=0) #
        bottom = self.surf + self.depth[:-1].sum()
        d = bottom - np.cumsum(self.depth[:-1])
        facies = np.argmax(p1, axis=0)
        if thext == None:
            thext = [0.,p1.max()]

        if depthext == None:
            depthext = [self.surf,bottom]


        colsed[len(self.sedH)-1]=np.array([244./256.,164/256.,96/256.,1.])

        # Define figure size
        fig = plt.figure(figsize=size, dpi=dpi, constrained_layout=True)
        gs = gridspec.GridSpec(1,11)
        ax1 = fig.add_subplot(gs[:3])
        ax2 = fig.add_subplot(gs[3:6], sharey=ax1)
        ax3 = fig.add_subplot(gs[6:9], sharey=ax1)
        ax4 = fig.add_subplot(gs[9], sharey=ax1)
        ax5 = fig.add_subplot(gs[10], sharey=ax1)
        ax3.set_facecolor('#f2f2f3')
        ax4.set_facecolor('#f2f2f3')
        ax5.set_facecolor('#f2f2f3')
        x = np.zeros(2)
        y = np.zeros(2)
        x[0] = 0.
        x[1] = 1.
        old = np.zeros(2)
        old[0] = bottom
        old[1] = bottom

        # Plotting curves
        for s in range(len(self.sedH)):
            ax1.plot(p1[s,:], d, label=self.names[s], linewidth=lwidth, c=colsed[s])
            ax2.plot(p2[s,:-1], d, label=self.names[s], linewidth=lwidth, c=colsed[s])
            if s == 0:
                ax3.fill_betweenx(d, 0, p3[s,:-1], color=colsed[s])
            else:
                ax3.fill_betweenx(d, p3[s-1,:-1], p3[s,:-1], color=colsed[s])
            ax3.plot(p3[s,:-1], d, 'k--', label=self.names[s], linewidth=lwidth-1)

        for s in range(len(d)):
            y[0] = d[s]
            y[1] = d[s]
            ax4.fill_between(x, old, y, color=coltime[s])
            ax5.fill_between(x, old, y, color=colsed[facies[s]])
            old[0] = y[0]
            old[1] = y[1]
            ax4.plot(x,y,'w')

        # Legend, title and labels
        ax1.grid()
        ax2.grid()
        ax3.grid()
        ax4.get_xaxis().set_visible(False)
        ax4.get_yaxis().set_visible(False)
        ax5.get_xaxis().set_visible(False)
        ax5.get_yaxis().set_visible(False)
        lgd = ax1.legend(frameon=True, loc='lower left', prop={'size':font+1})
        ax1.locator_params(axis='x', nbins=5)
        ax2.locator_params(axis='x', nbins=5)
        ax3.locator_params(axis='x', nbins=5)
        ax1.locator_params(axis='y', nbins=10)

        # Axis
        ax1.set_ylabel('Depth below ocean surface [m]', size=font+4)
        ax1.set_ylim(depthext[1], depthext[0])
        ax1.set_xlim(thext[0], thext[1])
        ax2.set_ylim(depthext[1], depthext[0])
        ax2.set_xlim(propext[0], propext[1])
        ax3.set_ylim(depthext[1], depthext[0])
        ax4.set_ylim(depthext[1], depthext[0])
        ax5.set_ylim(depthext[1], depthext[0])
        ax3.set_xlim(0., 1.)
        ax1.xaxis.tick_top()
        ax2.xaxis.tick_top()
        ax3.xaxis.tick_top()
        ax1.tick_params(axis='y', pad=5)
        ax1.tick_params(axis='x', pad=5)
        ax2.tick_params(axis='x', pad=5)
        ax3.tick_params(axis='x', pad=5)

        # Title
        tt1 = ax1.set_title('Thickness [m]', size=font+3)
        tt2 = ax2.set_title('Proportion [%]', size=font+3)
        tt3 = ax3.set_title('Accumulated [%]', size=font+3)
        tt4 = ax4.set_title('Time \n layers', size=font+3)
        tt5 = ax5.set_title('Bio \n Facies', size=font+3)
        tt1.set_position([.5, 1.04])
        tt2.set_position([.5, 1.04])
        tt3.set_position([.5, 1.04])
        tt4.set_position([.5, 1.025])
        tt5.set_position([.5, 1.025])
        fig.tight_layout()
        plt.tight_layout()
        #labels = [item.get_text() for item in ax2.get_yticklabels()]
        #for l in range(len(labels)):
        #    labels[l] = ' '
        #ax2.set_yticklabels(labels)
        #ax3.set_yticklabels(labels)
        # plt.show()
        if figname is not None:
            fig.savefig(figname, bbox_extra_artists=(lgd,), bbox_inches='tight')
            print 'Figure has been saved in',figname
        plt.close()



        # Define figure size
        fig = plt.figure(figsize=size, dpi=dpi)
        gs = gridspec.GridSpec(1,11)
        ax1 = fig.add_subplot(gs[:3])
        ax2 = fig.add_subplot(gs[3:6], sharey=ax1)
        ax3 = fig.add_subplot(gs[6:9], sharey=ax1)

        ax1.plot(self.sealevel, self.timeLay, linewidth=lwidth, c='slateblue')
        ax2.plot(self.waterflow, self.timeLay, linewidth=lwidth, c='darkcyan')
        ax3.plot(self.sedinput, self.timeLay, linewidth=lwidth, c='sandybrown')

        ax1.set_ylabel('Simulation time [a]', size=font+4)
        ax1.set_ylim(self.timeLay.min(), self.timeLay.max())
        ax2.set_ylim(self.timeLay.min(), self.timeLay.max())
        ax3.set_ylim(self.timeLay.min(), self.timeLay.max())
        ax1.set_facecolor('#f2f2f3')
        ax2.set_facecolor('#f2f2f3')
        ax3.set_facecolor('#f2f2f3')
        ax1.locator_params(axis='x', nbins=5)
        ax2.locator_params(axis='x', nbins=5)
        ax3.locator_params(axis='x', nbins=5)

        # Title
        tt1 = ax1.set_title('Sea-level [m]', size=font+3)
        tt2 = ax2.set_title('Water flow [m/second]', size=font+3)
        tt3 = ax3.set_title('Sediment input [m/year]', size=font+3)
        tt1.set_position([.5, 1.04])
        tt2.set_position([.5, 1.04])
        tt3.set_position([.5, 1.04])
        fig.tight_layout()
        plt.tight_layout()
        # plt.show()
        if figname is not None:
            fig.savefig(figname+'_envi.pdf',bbox_inches='tight')
            # fig.savefig(figname+'_envi.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')
            # print 'Figure has been saved in',figname+'_envi.pdf'
        plt.close()
        print ''

        if filename is not None:
            tmp = np.column_stack((d.T,p1.T))
            tmp1 = np.column_stack((tmp,p2[:,:-1].T))
            tmp2 = np.column_stack((tmp1,p3[:,:-1].T))
            tmp3 = np.column_stack((tmp2,self.sealevel[:-1].T))
            tmp4 = np.column_stack((tmp3,self.waterflow[:-1].T))
            tmp5 = np.column_stack((tmp4,self.sedinput[:-1].T))

            cols = []
            cols.append('depth')
            for s in range(len(self.names)):
                cols.append('th_'+self.names[s])
            for s in range(len(self.names)):
                cols.append('prop_'+self.names[s])
            for s in range(len(self.names)):
                cols.append('acc_'+self.names[s])
            cols.append('sealevel')
            cols.append('waterflow')
            cols.append('sedinput')

            df = pd.DataFrame(tmp5)
            df.columns = cols
            df.to_csv(filename, sep=sep, encoding='utf-8', index=False)
            print 'Model results have been saved in',filename

        return
