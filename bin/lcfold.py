#!/usr/bin/env python

import numpy as np
from matplotlib import pyplot as plt
import sys
import lctips
import copy

fname   = sys.argv[1]
period  = float(sys.argv[2])

data        = np.loadtxt(fname, comments='#').T
#data        = np.array((data[0], data[2]))

#sp_lc       = lctips.split_discon(data)
#dl = []
#for lc in sp_lc:
#    det_lc  = lctips.detrend_lc(lc, npoint=10)
#    dl.append(det_lc)
#data    = np.hstack(dl)
data_fold   = copy.copy(data)
data_fold[0]= data_fold[0]%period

means   = lctips.bin_lc(data_fold)

plt.scatter(data_fold[0], data_fold[1],s=3)
plt.scatter(means[0], means[1],s=5, c='black')
plt.show()
exit()




plt.show()


