import os
import math
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab




def main():
	datafile = 'synthetic_core/data_timestructure_08.txt'
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
	with file('data_timestructure_08_prop_3.txt', 'wb') as prop_file:
		with file('data_timestructure_08_vec_3.txt', 'wb') as vec_file:
			for x in range(intervals.size):
				vector = np.zeros(4)
				slc=[]
				rev = -1-x
				prop_file.write(('{0}\t'.format(intervals[rev])))
				slc = np.append(slc, prop_mat[x,:])
				# slc = np.append(slc, (a1[x],a2[x],a3[x],cs[x]))
				if not all(v == 0 for v in slc):
					facies_idx = np.argmax(slc) #finds the dominant assemblage in a slice
					vector[facies_idx] = 1.
					for y in range(4):
						prop_file.write('{0}\t'.format(vector[y]))
					vec_file.write('{0}\t{1}\n'.format(intervals[rev],facies_idx+1))
					prop_file.write('\n')

				else:
					for y in range(4):
						vector=np.full(4,-1.)
						prop_file.write('{0}\t'.format(vector[y]))
					vec_file.write('{0}\t{1}\n'.format(intervals[rev],-1.))
					prop_file.write('\n')

if __name__ == "__main__": main()