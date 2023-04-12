#!/usr/bin/env python3

import numpy as np

f1  = 'hyades.lst'
f2  = 'praesepe.lst'
f3  = 'pleiades.lst'
f4  = 'usco.lst'

d1  = np.loadtxt(f1,dtype='unicode').flatten()
d2  = np.loadtxt(f2,dtype='unicode').flatten()
d3  = np.loadtxt(f3,dtype='unicode').flatten()
d4  = np.loadtxt(f4,dtype='unicode').flatten()

elist   = np.hstack((d1,d2,d3,d4))
mdata   = np.loadtxt("k2ticxmatch_20210831.csv", delimiter=',', usecols=(0,1,2,3), dtype='unicode')

radec   = []
for eid in elist:
    radec.append(mdata[(mdata[:,1]==eid)])
radec   = np.vstack(radec)
print(radec)

np.savetxt("epicradec.lst", radec, fmt='%s')
