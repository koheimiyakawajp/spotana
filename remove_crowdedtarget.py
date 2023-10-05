#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import sys

def plothist(args):
    fcontam = args[1]
    dcontam = np.loadtxt(fcontam, dtype='unicode')
    crwd    = np.array(dcontam[1:,1], dtype='float')
    plt.hist(crwd)
    plt.show()


def main(args):
    ftarlist    = args[0]
    fcontam     = args[1]

    dlist       = np.loadtxt(ftarlist, delimiter=',', dtype='unicode')
    dcontam     = np.loadtxt(fcontam, dtype='unicode')
    
    flg_ar      = [True]
    for dline in dlist[1:]:
        epicid  = dline[0]
        contam_val  = dcontam[(dcontam[:,0]==epicid)]
        if np.ndim(contam_val) ==2:
            contam_val   = contam_val[0]
            print(contam_val)
            cwd     = float(contam_val[1])
            if cwd > 0.8:
                flg_ar.append(True)
            else:
                flg_ar.append(False)
        else:
            flg_ar.append(False)
    np.savetxt(ftarlist.split(".")[0] + "_rmcnt.csv", dlist[flg_ar], delimiter=',', fmt="%s")

#plothist(sys.argv[1:])
#exit()
main(sys.argv[1:])