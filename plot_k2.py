#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import sys

fk2sff  =   "lightcurves/"+sys.argv[1]+"_k2sff.dat"
fk2sap  =   "lightcurves/"+sys.argv[1]+"_k2sap.dat"

dk2sff  =   np.loadtxt(fk2sff, dtype='f8').T
dk2sap  =   np.loadtxt(fk2sap, dtype='f8').T

plt.scatter(dk2sff[0], dk2sff[1], s=0.5, label='K2SFF')
plt.scatter(dk2sap[0], dk2sap[1], s=0.5, label='K2SAP')
plt.legend()
plt.show()