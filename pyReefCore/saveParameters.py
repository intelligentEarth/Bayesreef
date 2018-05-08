##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
##                                                                                   ##
##  This file forms part of the BayesReef modelling software                         ##
##                                                                                   ##
##  For full license and copyright information, please refer to the LICENSE.md file  ##
##  located at the project root, or contact the authors.                             ##
##                                                                                   ##
##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
"""
This script is intended to save records of posteriors of parameters in BayesReef.
"""
import os
import csv
import numpy as np

def saveParameters(fname, sedsim, flowsim, naccept, pos_m, pos_ax, pos_ay, 
    pos_sed1, pos_sed2, pos_sed3, pos_sed4, 
    pos_flow1, pos_flow2, pos_flow3, pos_flow4, pos_diff, pos_likl, pos_samples, proposal):
    if sedsim == True:
        if not os.path.isfile(('%s/pos_sed.csv' % (fname))):
            with file(('%s/pos_sed.csv' % (fname)),'wb') as outfile:
                writer = csv.writer(outfile, delimiter=',')
                data = [naccept,np.ndarray.tolist(pos_sed1),np.ndarray.tolist(pos_sed2),np.ndarray.tolist(pos_sed3),np.ndarray.tolist(pos_sed4)]
                writer.writerow(data)
        else:
            with file(('%s/pos_sed.csv' % (fname)),'ab') as outfile:
                writer = csv.writer(outfile, delimiter=',')
                data = [naccept,np.ndarray.tolist(pos_sed1),np.ndarray.tolist(pos_sed2),np.ndarray.tolist(pos_sed3),np.ndarray.tolist(pos_sed4)]
                writer.writerow(data)
    
    if flowsim == True:
        if not os.path.isfile(('%s/pos_flow.csv' % (fname))):
            with file(('%s/pos_flow.csv' % (fname)),'wb') as outfile:
                writer = csv.writer(outfile, delimiter=',')
                data = [naccept,np.ndarray.tolist(pos_flow1),np.ndarray.tolist(pos_flow2),np.ndarray.tolist(pos_flow3),np.ndarray.tolist(pos_flow4)]
                writer.writerow(data)
        else:
            with file(('%s/pos_flow.csv' % (fname)),'ab') as outfile:
                writer = csv.writer(outfile, delimiter=',')
                data = [naccept,np.ndarray.tolist(pos_flow1),np.ndarray.tolist(pos_flow2),np.ndarray.tolist(pos_flow3),np.ndarray.tolist(pos_flow4)]
                writer.writerow(data)

    if not os.path.isfile(('%s/pos_m.csv' % (fname))):
        with file(('%s/pos_m.csv' % (fname)),'wb') as outfile:
            writer = csv.writer(outfile, delimiter=',')
            data = [naccept,pos_m]
            writer.writerow(data)
    else:
        with file(('%s/pos_m.csv' % (fname)),'ab') as outfile:
            writer = csv.writer(outfile, delimiter=',')
            data = [naccept,pos_m]
            writer.writerow(data)

    if not os.path.isfile(('%s/pos_aij.csv' % (fname))):
        with file(('%s/pos_aij.csv' % (fname)),'wb') as outfile:
            writer = csv.writer(outfile, delimiter=',')
            data = [naccept,pos_ax,pos_ay]
            writer.writerow(data)
    else:
        with file(('%s/pos_aij.csv' % (fname)),'ab') as outfile:
            writer = csv.writer(outfile, delimiter=',')
            data = [naccept,pos_ax,pos_ay]
            writer.writerow(data)

    if not os.path.isfile(('%s/pos_diff.csv' % (fname))):
        with file(('%s/pos_diff.csv' % (fname)),'wb') as outfile:
            writer = csv.writer(outfile, delimiter=',')
            data = [naccept,pos_diff]
            writer.writerow(data)
    else:
        with file(('%s/pos_diff.csv' % (fname)),'ab') as outfile:
            writer = csv.writer(outfile, delimiter=',')
            data = [naccept,pos_diff]
            writer.writerow(data)

    if not os.path.isfile(('%s/pos_likl.csv' % (fname))):
        with file(('%s/pos_likl.csv' % (fname)),'wb') as outfile:
            writer = csv.writer(outfile, delimiter=',')
            data = [naccept,pos_likl]
            writer.writerow(data)
    else:
        with file(('%s/pos_likl.csv' % (fname)),'ab') as outfile:
            writer = csv.writer(outfile, delimiter=',')
            data = [naccept,pos_likl]
            writer.writerow(data)

    # Save accepted samples
    if not os.path.isfile(('%s/pos_samples.csv' % (fname))):
        with file(('%s/pos_samples.csv' % (fname)),'wb') as outfile:
            writer = csv.writer(outfile, delimiter=',')
            data = [naccept,np.ndarray.tolist(pos_samples)]
            writer.writerow(data)
    else:
        with file(('%s/pos_samples.csv' % (fname)),'ab') as outfile:
            writer = csv.writer(outfile, delimiter=',')
            data = [naccept,np.ndarray.tolist(pos_samples)]
            writer.writerow(data)
    
    # Save accepted proposals
    if not os.path.isfile('%s/pos_proposals.csv' % (fname)):
        with file(('%s/pos_proposals.csv' % (fname)), 'wb') as outfile:
            writer = csv.writer(outfile, delimiter=',')
            writer.writerow(proposal)
    else:
        with file(('%s/pos_proposals.csv' % (fname)), 'ab') as outfile:
            writer = csv.writer(outfile, delimiter=',')
            writer.writerow(proposal)