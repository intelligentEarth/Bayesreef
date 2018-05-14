import os
import math
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab




def main():
	src = 'synthetic_core'
	datafile = '%s/data_timestructure_08.txt' % src
	intervals, a1,a2,a3,cs = np.genfromtxt(datafile, usecols=(0,1,2,3,4), unpack=True)
	prop_mat = np.loadtxt(datafile, usecols=(1,2,3,4))
	# print 'prop matrix', prop_mat
	# ids_list = []
	# for n in range(intervals.shape[0]):
	# 	print n
	# 	a = np.where(prop_mat[n,:] ==0.)[0]
	# 	ids_list = np.append(ids_list, a)
	# print ids_list
	idx=0
	# write multinomial matrix
	with file('data_timestructure_08_2.txt', 'wb') as prop_file:
		with file('data_timestructure_08_vec.txt', 'wb') as vec_file:
			for x in range(intervals.size):
				vector = np.zeros(4)
				slc=[]
				rev = -1-x
				prop_file.write(('{0}\t'.format(intervals[rev])))
				slc = np.append(slc, (a1[x],a2[x],a3[x],cs[x]))
				if not all(v == 0 for v in slc):
					facies_idx = np.argmax(slc) #finds the dominant assemblage in a slice
					vector[facies_idx] = 1
					for y in range(4):
						prop_file.write('{0}\t'.format(vector[y]))
					vec_file.write('{0}\t{1}\n'.format(intervals[rev],facies_idx+1))
					prop_file.write('\n')

				else:
					for y in range(4):
						prop_file.write('{0}\t'.format(vector[y]))
					vec_file.write('{0}\t0.\n'.format(intervals[rev]))
					prop_file.write('\n')

				# for x in range(intervals.size):
				# 	slc=[]
				# 	slc = np.append(slc, (a1[x],a2[x],a3[x],cs[x]))
				# 	facies_idx = np.argmax(slc)
				# 	vec_file.write('{0}\t{1}\n'.format(intervals[x],facies_idx+1))
			
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
