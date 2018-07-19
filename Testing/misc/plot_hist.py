
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats 


def plotHistogram():
    sed1=[0.0009, 0.0015, 0.0023]
    sed2=[0.0015, 0.0017, 0.0024]
    sed3=[0.0016, 0.0028, 0.0027]
    sed4=[0.0017, 0.0031, 0.0043]
    flow1=[0.055, 0.008 ,0.]
    flow2=[0.082, 0.051, 0.]
    flow3=[0.259, 0.172, 0.058] 
    flow4=[0.288, 0.185, 0.066] 
    nb_bins=30
    font = 11
    width = 1

    # pos_val = np.loadtxt('pos_f2.txt')
    point = 1
    true_val = 0.008
    axis_title = r'$f_{flow}^1$'
    print 'true value', true_val

    pos = np.genfromtxt('pos_burnin_flow.csv', delimiter=",", usecols=(1,4))
    print 'pos', pos.shape
    pos_ = pos[:,point-1]
    print 'pos_', pos_.shape
    long_title = 'Moderate-deep assemblage, %s' % axis_title
    slen = np.arange(0,pos.shape[0], 1)

    fig = plt.figure(figsize=(5,6))
    ax = fig.add_subplot(111)
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    ax.set_title('%s' % long_title, fontsize= font+2)#, y=1.02)
    
    ax1 = fig.add_subplot(211)
    ax1.set_facecolor('#f2f2f3')
    ax1.hist(pos_, bins=25, alpha=0.5, facecolor='steelblue', normed=True)
    ax1.axvline(true_val, linestyle='-', color='black', linewidth=1,label='True value')
    ax1.grid(True)
    ax1.set_ylabel('Density',size=font+1)
    ax1.set_xlabel('%s' % axis_title, size=font+1)

    ax2 = fig.add_subplot(212)
    ax2.set_facecolor('#f2f2f3')
    ax2.plot(slen,pos_,linestyle='-', linewidth=width, color='k', label=None)
    ax2.set_title(r'Trace of %s' % axis_title,size=font+2)
    ax2.set_xlabel('Samples',size=font+1)
    ax2.set_ylabel('%s' % axis_title, size=font+1)
    ax2.set_xlim([0,np.amax(slen)])
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    plt.savefig('pos_flow_%s.png' % point, bbox_inches='tight', dpi=300, transparent=False)
    plt.clf()

def main():
    plotHistogram()

if __name__ == "__main__": main()
