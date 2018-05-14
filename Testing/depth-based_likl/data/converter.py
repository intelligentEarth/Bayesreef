import os
import math
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab




def main():
	datafile = 'data_probstic_08.txt'
	core_depths, a1,a2,a3,cs = np.genfromtxt(datafile, usecols=(0,1,2,3,4), unpack=True)


	idx=0
	# write multinomial matrix
	with file('synth_core_prop_08.txt', 'wb') as outfile:
		for x in range(core_depths.size):
			slc=[]
			rev = x#-1-x
			# print 'a1[x],a2[x],a3[x],cs[x]',a1[x],a2[x],a3[x],cs[x]
			slc = np.append(slc, (a1[x],a2[x],a3[x],cs[x]))
			facies_idx = np.argmax(slc)
			vector = np.zeros(4)
			vector[facies_idx] = 1
			outfile.write(('{0}\t'.format(core_depths[rev])))
			for y in range(4):
				outfile.write('{0}\t'.format(vector[y]))
			outfile.write('\n')
	with file('synth_core_vec_08.txt', 'wb') as outfile:
		for x in range(core_depths.size):
			slc=[]
			slc = np.append(slc, (a1[x],a2[x],a3[x],cs[x]))
			facies_idx = np.argmax(slc)
			outfile.write('{0}\t{1}\n'.format(core_depths[x],facies_idx+1))
	
	# data = np.loadtxt("synth_core.txt")
	# core_depths = data[:,0]
	# print 'core_depths', core_depths
	# print 'core_depths.size',core_depths.size
	# core_data = data[:,1]
	# print 'core_data',core_data
	# pred_core = np.zeros((core_depths.size,4))

	# for n in range(0,core_depths.size):
	# 		if core_data[n] == 0.571:
	# 			pred_core[n,3] = 1 
	# 		if core_data[n] == 0.429:
	# 			pred_core[n,2] = 1 
	# 		if core_data[n] == 0.286:
	# 			pred_core[n,1] = 1 
	# 		if core_data[n] == 0.143:
	# 			pred_core[n,0] = 1 

	# print pred_core

	# pred_core_ = str(pred_core)

	# with file('testing.txt','wb') as outfile:
	# 	outfile.write ('')

	# with file('testing.txt','ab') as outfile:
	# 	for x in range(0,core_depths.size):
	# 		for y in range(0,4):
	# 			val = str(int(pred_core[x,y]))
	# 			rev = -1-x
	# 			depth_str = str(core_depths[rev])
	# 			outfile.write('{0}\t{1}\n'.format(depth_str, val))

if __name__ == "__main__": main()

#Need to load what is output by the RunModel
