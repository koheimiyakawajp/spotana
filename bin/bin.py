#!/usr/bin/env python

import lctips
import sys
import numpy as np
from matplotlib import pyplot as plt

fname   = sys.argv[1]
data    = np.loadtxt(fname, comments='#').T
lc      = np.array((data[0], data[2]))

lc_bin  = lctips.bin_lc(lc, 14)
plt.scatter(lc[0], lc[1], s = 2)
plt.scatter(lc_bin[0], lc_bin[1], s=2, c='black')
plt.show()
