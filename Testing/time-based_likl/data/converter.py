import os
import math
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab


def noGrowthColumn(intervals, prop_mat):
	"""create function that puts a 1 in a 1d array (x) when the growth is zero"""
	v_nogrowth = np.zeros((intervals.size,1))

	for a in range(intervals.size):
		if np.amax(prop_mat[a,:]) == 0.:
			v_nogrowth[a,:] = 1.
	prop_mat = np.append(prop_mat,v_nogrowth,axis=1)
	return prop_mat

def main():
	datafile = 'synthetic_core/rawsynth-10.txt'
	intervals, a1,a2,a3,cs = np.genfromtxt(datafile, usecols=(0,1,2,3,4), unpack=True)
	prop_mat = np.loadtxt(datafile, usecols=(1,2,3,4))
	prop_mat = noGrowthColumn(intervals, prop_mat)
	idx=0
	no_growth_val = 0.
	# write multinomial matrix
	with file('synthdata_t_prop_08_x.txt', 'wb') as prop_file:
		with file('synthdata_t_vec_08_x.txt', 'wb') as vec_file:
			for x in range(intervals.size):
				vector = np.zeros(prop_mat.shape[1])
				slc=[]
				rev = -1-x
				prop_file.write(('{0}\t'.format(intervals[rev])))
				slc = np.append(slc, prop_mat[x,:])
				# slc = np.append(slc, (a1[x],a2[x],a3[x],cs[x]))
				if not all(v == 0 for v in slc):
					facies_idx = np.argmax(slc) #finds the dominant assemblage in a slice
					vector[facies_idx] = 1.
					for y in range(prop_mat.shape[1]):
						prop_file.write('{0}\t'.format(vector[y]))
					vec_file.write('{0}\t{1}\n'.format(intervals[rev],facies_idx+1))
					prop_file.write('\n')

				else:
					for y in range(prop_mat.shape[1]):
						vector=np.full(prop_mat.shape[1],no_growth_val)
						prop_file.write('{0}\t'.format(vector[y]))
					vec_file.write('{0}\t{1}\n'.format(intervals[rev],no_growth_val))
					prop_file.write('\n')
	print 'Finished conversion.'

if __name__ == "__main__": main()