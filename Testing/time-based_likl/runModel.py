
# coding: utf-8


import matplotlib.pyplot as plt
import numpy as np
import os
import fnmatch
import shutil
from PIL import Image



# # ReefCore library
# 
# **ReefCore**  is a 1D model which simulates evolution of mixed carbonate-siliciclastic system under environmental forcing conditions (e.g. sea-level, water flow, siliciclastic input).
# 
# The carbonate production model simulates the logistic growth and interaction among species based on **Generalized Lotka-Volterra** equations. This equation is mainly formed by two parts, the logistic growth/decay of a species and its interaction with the other species,
# 
# $$\frac{dx_i}{dt} = \epsilon_i x_i + \sum_{j=1}^{N_s} \alpha_{ij}x_ix_j$$
# 
# where $x_i$ is the population density of species _i_; $\epsilon_i$ is the intrinsic rate of increase/decrease of a population of species _i_ (also called **Malthusian** parameter); $\alpha_{ij}$ is the interaction coefficient among the species association _i_ and _j_, (a particular case is $\alpha_{ii}$, the interaction of one species association with itself); and _t_ is time. 
# 
# In **ReefCore** the equation is solved using **Runge-Kutta-Fehlberg** method (_RKF45_ or _Fehlberg_ as defined in the odespy library).


from pyReefCore.model import Model


# On the library has been loaded, the model initialisation is done using the following command:


# Initialise model
reef = Model()


# # Build environmental forcing curves
# 
# 


# reef.enviforcing.buildCurve(timeExt=[0,500],timeStep=10,
#                             funcExt=[0,20],ampExt=[2,5],
#                             periodExt=[100,150])



#reef.enviforcing.plotCurves(size=(3,6), lwidth = 3, title='Sea-level curve [m]', 
#                            color='slateblue', font=8, dpi=80, figName = None)



# reef.enviforcing.exportCurve(nameCSV='data/sealevel.csv')


# # XmL input file
# 
# The next step consists in defining the initial conditions for our simulation. This is done by using an **XmL** input file which set the parameters to be used, such as:
# 
# - the initial species population number $X0$
# - the intrinsic rate of a population species $\epsilon$
# - the interaction coefficients among the species association $\alpha$

filename = 'input_synth.xml'

reef.load_xml(filename, False, False)

run_nb =0
while os.path.exists('model-%s' % (run_nb)):
      run_nb+=1

if not os.path.exists('model-%s' % (run_nb)):
    os.makedirs('model-%s' % (run_nb))


# Visualise the initial conditions of your model run can be done using the following command:


reef.core.initialSetting(size=(8,2.5), size2=(8,4.5), dpi=300, fname='-%s.pdf' % (run_nb))
#write initial settings somewhere 


# # Model simulation
# 
# The core of the code consist in solving the system of ODEs from the **GLV** equations using the **RKF** method.
# 
# Once a species association population is resolved, carbonate production is calculated using a carbonate production factor. Production factors are specified for the maximum population, and linearly scaled to the actual population following the relation
# $$ \frac{dP}{dt} = R_{max}\frac{x_i}{K_i}$$
# 
# where $P$ is the carbonate production, $t$ is time, $R_{max}$ is the carbonate production factor when population is at its maximum, and $K_i$ is the maximum population of species _i_, computed as
# 
# $$K_i=\frac{\epsilon_i}{\alpha_{ii}}$$
# 
# To run the model for a given time period [years], the following function needs to be called:


reef.run_to_time(8500,showtime=100.)


# # Results
# 
# All the output from the model run can be plotted on the notebook using a series of internal functions presented below.
# 
# First one can specify a colormap to use for the plot using one of the matplotlib predefined colormap proposed here: 
# - [colormaps_reference](http://matplotlib.org/examples/color/colormaps_reference.html)


from matplotlib.cm import terrain, plasma
nbcolors = len(reef.core.coralH)+10 #14
print 'nbcolors',nbcolors
colors = terrain(np.linspace(0, 1.8, nbcolors))

nbcolors = len(reef.core.layTime)+3 #174
print 'nbcolors', nbcolors
colors2 = plasma(np.linspace(0, 1, nbcolors))


# ## Species population evolution
# 
# - with time: `reef.plot.speciesTime`
# - with depth: `reef.plot.speciesDepth`


reef.plot.speciesTime(colors=colors, size=(8,4), font=8, dpi=100,fname=('apop_t-%s.pdf' % (run_nb)))
reef.plot.speciesDepth(colors=colors, size=(8,4), font=8, dpi=100, fname =('apop_d-%s.pdf' % (run_nb)))
reef.plot.accomodationTime(size=(8,4), font=8, dpi=100, fname =('acc_t-%s.pdf' % (run_nb)))


# ## Coral synthetic core
# 
# The main output of the model consists in the synthetic core which shows the evolution of the coral stratigraphic architecture obtained from the interactions among species and with their environment. The plot is obtained using the following function: 
# - `reef.plot.drawCore`
# 
# The user has the option to save: 
# - the figure using the `figname` parameter (`figname` could either have a _.png_ or _.pdf_ extension)
# - the model output as a _CSV_ file using the `filename` parameter. This will dump all output dataset for further analysis if required.


reef.plot.drawCore(lwidth = 3, colsed=colors, coltime = colors2, size=(9,8), font=8, dpi=380, 
                   figname=('core-%s.pdf' % (run_nb)), filename=('core-%s.csv' % (run_nb)), sep='\t')



def gen_find(filepat,top):
    for path, dirlist, filelist in os.walk(top):
        for name in fnmatch.filter(filelist,filepat):
            yield os.path.join(path,name)
src = os.getcwd()
print src
dst = '%s/model-%s' % (src,run_nb)

shutil.copyfile(src+'/'+filename, dst+'/input-%s.xml' % (run_nb))

filesToMove = gen_find(('*-%s*' % (run_nb)),src)
for name in filesToMove:
    shutil.move(name, dst)


# np.set_printoptions(threshold='nan')
# communities = 3
# # synth_core = reef.convert_core(communities,output_core, core_depths)
# synth_core, timelay = reef.plot.convertTimeStructure()
# print 'synth core', synth_core
# src = 'data/synthetic_core'
# with file(('%s/rawsynth-%s.txt' % (src, run_nb)), 'w') as outfile:
#     for h in range(timelay.size):
#         rev = -1-h
#         depth_str = str(timelay[h])
#         outfile.write('{0}\t'.format(depth_str))
#         for c in range(communities+1):
#             oc_string = str(synth_core[h,c])
#             outfile.write('{0}\t'.format(oc_string))
#         outfile.write("\n")