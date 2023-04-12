#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import sys
import copy

def sekibun(data):
    dtime   = data[0,1:] - data[0,:-1]
    f   = data[1,1:]
    val     = np.sum(f*dtime)
    return val

spdata  = np.loadtxt(sys.argv[1], dtype='f8').T
bpdata  = spdata[:,((6000<spdata[0])&(spdata[0]<7000))]
d   = 0.001
#plt.scatter(bpdata[0], bpdata[1], s=0.1)

Hal     = spdata[:,((6562<spdata[0])&(spdata[0]<6564))]
Hals    = spdata[:,((6562+d<spdata[0])&(spdata[0]<6564+d))]
plt.plot(Hal[0], Hal[1], lw=2, c='red')
plt.plot(Hals[0], Hals[1], lw=2, c='blue')
plt.show()
a   = sekibun(Hal)
b   = sekibun(Hals)
print(a,b,a/b)

