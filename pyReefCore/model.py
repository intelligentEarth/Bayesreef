##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
##                                                                                   ##
##  This file forms part of the pyReefCore synthetic coral reef core model app.      ##
##                                                                                   ##
##  For full license and copyright information, please refer to the LICENSE.md file  ##
##  located at the project root, or contact the authors.                             ##
##                                                                                   ##
##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
"""
   pyReefCore Model main entry file.
"""
import time
import numpy as np
import mpi4py.MPI as mpi
from scipy import linalg, mat, dot
import operator

from pyReefCore import (preProc, xmlParser, enviForce, coralGLV, coreData, modelPlot)

# profiling support
import cProfile
import os
import pstats
import StringIO

class Model(object):
    """State object for the pyReef model."""

    def __init__(self):
        """
        Constructor.
        """
        # Simulation state
        self.dt = 0.
        self.tNow = 0.
        self.tDisp = 0.
        self.waveID = 0
        self.outputStep = 0
        self.applyDisp = False
        self.simStarted = False

        self.dispRate = None

        self._rank = mpi.COMM_WORLD.rank
        self._size = mpi.COMM_WORLD.size
        self._comm = mpi.COMM_WORLD

        # Initialise pre-processing functions
        self.enviforcing = preProc.preProc()
        # Optimising algorithm score
        self.score =0.
        self.run_count=0
        # Optimised parameters
        self.opt_Sed = []
        self.opt_Flow = []
        self.opt_malthusParam = []
        self.opt_cMatrix = []
        self.initial_sed = []
        self.initial_flow = []

    def load_xml(self, filename, sedsim, flowsim, verbose=False):
        """
        Load an XML configuration file.
        """

        # Only the first node should create a unique output dir
        self.input = xmlParser.xmlParser(filename, makeUniqueOutputDir=(self._rank == 0))
        self.tNow = self.input.tStart
        self.tCoral = self.tNow
        self.tLayer = self.tNow + self.input.laytime

        ################### Reassign input parameters with input vector values ###################
        self.initial_sed = self.input.__dict__["enviSed"]
        self.initial_flow = self.input.__dict__["enviFlow"]

        if (sedsim == True) or (flowsim == True):
            if (sedsim == True) and (flowsim == False):
                self.input.__dict__["enviSed"] = self.opt_Sed
                # envsd = self.input.__dict__["enviSed"]
                # print 'Sediment Function:', envsd
                # print 'envsd shape:', envsd.shape
                self.input.__dict__["communityMatrix"] = self.opt_cMatrix
                self.input.__dict__["malthusParam"] = self.opt_malthusParam
            elif (flowsim == True) and (sedsim == False):
                self.input.__dict__["enviFlow"] = self.opt_Flow
                # envwf = self.input.__dict__["enviFlow"]
                # print 'envwf:', envwf
                # envsd = self.input.__dict__["enviSed"]
                # print 'Sediment Function:', envsd
                self.input.__dict__["communityMatrix"] = self.opt_cMatrix
                self.input.__dict__["malthusParam"] = self.opt_malthusParam
            elif (sedsim == True) and (flowsim == True):
                self.input.__dict__["enviSed"] = self.opt_Sed
                envsd = self.input.__dict__["enviSed"]
                self.input.__dict__["enviFlow"] = self.opt_Flow
                envwf = self.input.__dict__["enviFlow"]
                self.input.__dict__["communityMatrix"] = self.opt_cMatrix
                self.input.__dict__["malthusParam"] = self.opt_malthusParam
            cmmat= self.input.__dict__["communityMatrix"]
            mtpar= self.input.__dict__["malthusParam"]
            # print 'Community matrix:', cmmat
            # print 'malthus:', mtpar
    	##########################################################################################

        # Seed the random number generator consistently on all nodes
        seed = None
        if self._rank == 0:
            # limit to max uint32
            seed = np.random.mtrand.RandomState().tomaxint() % 0xFFFFFFFF
        seed = self._comm.bcast(seed, root=0)
        np.random.seed(seed)
        self.iter = 0
        self.layID = 0

        # Initialise environmental forcing conditions
        self.force = enviForce.enviForce(input=self.input)
        # print 'self_dict_force', self.force.__dict__.keys()

        # Initialise core data
        self.core = coreData.coreData(input=self.input)

        # Environmental forces functions
        self.core.seatime = self.force.seatime
        self.core.sedtime = self.force.sedtime
        self.core.flowtime = self.force.flowtime
        self.core.seaFunc = self.force.seaFunc
        self.core.sedFunc = self.force.sedFunc
        self.core.flowFunc = self.force.flowFunc
        self.core.sedfctx = self.force.plotsedy
        self.core.sedfcty = self.force.plotsedx
        self.core.flowfctx = self.force.plotflowy
        self.core.flowfcty = self.force.plotflowx

        # Initialise plotting functions
        self.plot = modelPlot.modelPlot(input=self.input)

        return self.initial_sed, self.initial_flow

    def run_to_time(self, tEnd, showtime=10, profile=False, verbose=False):
        """
        Run the simulation to a specified point in time (tEnd).

        If profile is True, dump cProfile output to /tmp.
        """

        timeVerbose = self.tNow+showtime

        if profile:
            pid = os.getpid()
            pr = cProfile.Profile()
            pr.enable()

        if self._rank == 0:
            print 'tNow = %s [yr]' %self.tNow

        if tEnd > self.input.tEnd:
            tEnd = self.input.tEnd
            print 'Requested time is set to the simulation end time as defined in the XmL input file'

        if self.tNow == self.input.tStart:
            # Initialise Generalized Lotka-Volterra equation
            self.coral = coralGLV.coralGLV(input=self.input)

        # Perform main simulation loop
        # NOTE: number of iteration for the ODE during a given time step, could be user defined...
        N = 100

        # Define environmental factors
        dfac = np.ones(self.input.speciesNb,dtype=float)
        sfac = np.ones(self.input.speciesNb,dtype=float)
        ffac = np.ones(self.input.speciesNb,dtype=float)
        while self.tNow < tEnd:
            # Initial coral population
            if self.tNow == self.input.tStart:
                self.coral.population[:,self.iter] = self.input.speciesPopulation

            # Store accomodation space through time
            self.coral.accspace[self.iter] = max(self.core.topH,0.)

            # Get sea-level
            if self.input.seaOn:
                tmp = self.core.topH
                self.core.topH, dfac = self.force.getSea(self.tNow, tmp)
                if self.tNow == self.input.tStart:
                    self.core.sealevel[self.layID] = self.force.sealevel
                else:
                    self.core.sealevel[self.layID+1] = self.force.sealevel

            # Get sediment input
            if self.input.sedOn:
                sedh, sfac = self.force.getSed(self.tNow, self.core.topH)
                self.core.sedinput[self.layID] = self.force.sedlevel
            else:
                sedh = 0.

            # Get flow velocity
            if self.input.flowOn:
                ffac = self.force.getFlow(self.tNow, self.core.topH)
                self.core.waterflow[self.layID] = self.force.flowlevel

            # Limit species activity from environmental forces
            tmp = np.minimum(dfac, sfac)
            fac = np.minimum(ffac, tmp)
            self.coral.epsilon = self.input.malthusParam * fac

            # Initialise RKF conditions
            self.odeRKF = self.coral.solverGLV()
            self.odeRKF.set_initial_condition(self.coral.population[:,self.iter])

            # Define coral evolution time interval and time stepping
            self.tCoral += self.input.tCarb
            tODE = np.linspace(self.tNow, self.tCoral, N+1)
            self.dt = tODE[1]-tODE[0]

            # Solve the Generalized Lotka-Volterra equation
            coral,t = self.odeRKF.solve(tODE)
            population = coral.T
            tmppop = np.copy(population[:,-1])
            # maxpop
            tmppop[tmppop>100.] = 100.
            population[:,-1] = tmppop

            # Update coral population
            self.iter += 1
            ids = np.where(self.coral.epsilon==0.)[0]
            population[ids,-1] = 0.
            ids = np.where(np.logical_and(fac>=0.5,population[:,-1]==0.))[0]
            population[ids,-1] = 1.

            self.coral.population[:self.input.speciesNb,self.iter] = population[:,-1]

            # In case there is no accomodation space
            if self.core.topH <= 0.:
                population[ids,-1] = 0.
                self.coral.population[:self.input.speciesNb,self.iter] = 0.

            # Compute carbonate production and update coral core characteristics
            self.core.coralProduction(self.layID, self.coral.population[:,self.iter],
                                      self.coral.epsilon, sedh)
            # Update time step
            self.tNow = self.tCoral

            # Update stratigraphic layer ID
            if self.tLayer <= self.tNow :
                self.tLayer += self.input.laytime
                self.layID += 1

            if self._rank == 0 and self.tNow>=timeVerbose:
                timeVerbose = self.tNow+showtime
                # print 'tNow = %s [yr]' %self.tNow

        #print "Did not return from this", "...I have my doubts"

        # Update plotting parameters
        self.plot.pop = self.coral.population
        self.plot.timeCarb = self.coral.iterationTime
        self.plot.depth = self.core.thickness
        self.plot.sedH = self.core.coralH
        self.plot.timeLay = self.core.layTime
        self.plot.surf = self.core.topH
        self.plot.sealevel = self.core.sealevel
        self.plot.sedinput = self.core.sedinput
        self.plot.waterflow = self.core.waterflow
        self.plot.accspace = self.coral.accspace

        return

    def ncpus(self):
        """
        Return the number of CPUs used to generate the results.
        """

        return 1

    def convert_vector(self, communities, input_vector, sedsim, flowsim, verbose=False):
        # print 'input vector: ', input_vector
        new_shape = communities*4
        if (sedsim == True) and (flowsim == False):
            self.opt_Sed = input_vector[0:new_shape].reshape(4,communities)
            self.opt_Sed = self.opt_Sed.T
            x = input_vector[new_shape]
            y = input_vector[new_shape+1]
            diagmat = np.zeros((communities, communities))
            np.fill_diagonal(diagmat, x)
            for i in range(0, communities - 1):
                diagmat[i][i + 1] = y
                diagmat[i + 1][i] = y
            self.opt_cMatrix = diagmat
            tempParam= float(input_vector[new_shape+2])
            self.opt_malthusParam = np.full(communities, tempParam)
            # self.opt_malthusParam = np.ones(communities)
            # for i in range(0,self.opt_malthusParam.shape[0]):
            #     self.opt_malthusParam[i] = float(tempParam)
        elif (flowsim == True) and (sedsim == False):
            self.opt_Flow = input_vector[0:new_shape].reshape(4,communities)
            self.opt_Flow = self.opt_Flow.T
            x = input_vector[new_shape]
            y = input_vector[new_shape+1]
            diagmat = np.zeros((communities, communities))
            np.fill_diagonal(diagmat, x)
            for i in range(0, communities - 1):
                diagmat[i][i + 1] = y
                diagmat[i + 1][i] = y
            self.opt_cMatrix = diagmat
            tempParam= float(input_vector[new_shape+2])
            self.opt_malthusParam = np.full(communities, tempParam)
        elif (sedsim == True) and (flowsim == True):
            self.opt_Sed = input_vector[0:new_shape].reshape(4,communities)
            self.opt_Flow = input_vector[new_shape:(new_shape*2)].reshape(4,communities)
            self.opt_Sed = self.opt_Sed.T
            self.opt_Flow = self.opt_Flow.T
            x = input_vector[new_shape*2]
            y = input_vector[(new_shape*2)+1]
            diagmat = np.zeros((communities, communities))
            np.fill_diagonal(diagmat, x)
            for i in range(0, communities - 1):
                diagmat[i][i + 1] = y
                diagmat[i + 1][i] = y
            self.opt_cMatrix = diagmat
            tempParam= float(input_vector[(new_shape*2)+2])
            self.opt_malthusParam = np.full(communities, tempParam)
        optf = self.opt_Flow
        opts = self.opt_Sed
        optCM = self.opt_cMatrix
        optMP = self.opt_malthusParam

        print 'New parameters:'
        # print '\t Sed:\n', opts
        # print '\t Flow:\n', optf
        print '\t Matrix main:', x, 'and sub-/super:', y
        print '\t Malthus.:', tempParam
        
        return 

    def convert_core(self, communities, output_core, core_depths):
        # print 'output_core', output_core
        predicted_core = np.zeros(core_depths.size)        
        if (communities == 3): 
            comm_1 = output_core[0,:].flatten()
            comm_2 = output_core[1,:].flatten()
            comm_3 = output_core[2,:].flatten()
            sed = output_core[3,:].flatten()
            comms_at_depth = np.zeros((core_depths.size, communities+1))
            for i in range(0,core_depths.size):
                comms_at_depth[i,0] = comm_1[i]
                comms_at_depth[i,1] = comm_2[i]
                comms_at_depth[i,2] =comm_3[i]
                comms_at_depth[i,3] = sed[i]
            for n in range(0,core_depths.size):
                index, value = max(enumerate(comms_at_depth[n]), key=operator.itemgetter(1))
                # print 'no:', n, 'index',index,'value',value
                if index == 0 and value != 0:
                    predicted_core[n] = 0.143
                elif index == 1:
                    predicted_core[n] = 0.286
                elif index == 2:
                    predicted_core[n] = 0.429
                elif index == 3:
                    predicted_core[n] = 0.571
                else:
                    predicted_core[n] = 0
            # print 'predicted_core', predicted_core
        
        elif (communities == 6): # and (self.opt_malthusParam != []):
            comm_1 = output_core[0,:].flatten()
            comm_2 = output_core[1,:].flatten()
            comm_3 = output_core[2,:].flatten()
            comm_4 = output_core[3,:].flatten()
            comm_5 = output_core[4,:].flatten()
            comm_6 = output_core[5,:].flatten()
            sed = output_core[6,:].flatten()
            comms_at_depth = np.zeros((core_depths.size, communities+1)) #initialise array 
            
            for i in range(0,core_depths.size):
                comms_at_depth[i,0] = comm_1[i]
                comms_at_depth[i,1] = comm_2[i]
                comms_at_depth[i,2] =comm_3[i]
                comms_at_depth[i,3] = comm_4[i]
                comms_at_depth[i,4] = comm_5[i]
                comms_at_depth[i,5] = comm_6[i]
                comms_at_depth[i,6] = sed[i]
            # print 'Final comms_at_depth', comms_at_depth
            
            for n in range(0,core_depths.size):
                index, value = max(enumerate(comms_at_depth[n]), key=operator.itemgetter(1))
                # print 'no:', n, 'index',index,'value',value
                if index == 0 and value != 0:
                    predicted_core[n] = 0.143
                elif index == 1:
                    predicted_core[n] = 0.286
                elif index == 2:
                    predicted_core[n] = 0.429
                elif index == 6:
                    predicted_core[n] = 0.571
                elif index == 3:
                    predicted_core[n] = 0.714
                elif index == 4:
                    predicted_core[n] = 0.857
                elif index == 5:
                    predicted_core[n] = 1.0
                else:
                    predicted_core[n] = 0
            # print 'predicted_core', predicted_core

        # elif (communities == 6) and (self.opt_malthusParam == []):
        #      for n in range(0,core_depths.size):
        #         index, value = max(enumerate(comms_at_depth[n]), key=operator.itemgetter(1))
        #         # print 'no:', n, 'index',index,'value',value
        #         if index == 0 and value != 0:
        #             predicted_core[n] = 0.143
        #         elif index == 1:
        #             predicted_core[n] = 0.286
        #         elif index == 2:
        #             predicted_core[n] = 0.429
        #         elif index == 3:
        #             predicted_core[n] = 0.571
        #         elif index == 4:
        #             predicted_core[n] = 0.714
        #         elif index == 5:
        #             predicted_core[n] = 0.857
        #         elif index == 6:
        #             predicted_core[n] = 1.0
        #         else:
        #             predicted_core[n] = 0
        #     print 'predicted_core', predicted_core
        else:
            print 'No predicted core made'

        return predicted_core
    
