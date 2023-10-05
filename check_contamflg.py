#!/usr/bin/env python3

import numpy as np
from bin.getlc import check_contaminant as cc

import sys

flist   = sys.argv[1]
data    = np.loadtxt(flist, delimiter=',', dtype='unicode')[1:]
epicid  = np.array(data[:,0])
print(epicid)
result_ar   = []
for eid in epicid:
    print(eid)
    tmp    = cc(eid)
    if len(tmp) == 2:
        cwd,ffs = tmp
        print(cwd,ffs)
        result_ar.append([eid, cwd, ffs])
    else:
        result_ar.append([eid, np.nan, np.nan])
result_ar   = np.array(result_ar, dtype='unicode')
np.savetxt(flist.split(".csv")[0] + "_crowd.dat", result_ar, fmt="%s")