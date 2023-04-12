#!/usr/bin/env python

import numpy as np
from matplotlib import pyplot as plt
import copy
import sys

import getlc, fft, lctips, period


def lc_detrend(data):
    data[1]     = data[1] - np.median(data[1])
    sp_lc       = lctips.split_discon(data)

    result  = []
    dl = []
    for lc in sp_lc:
        det_lc  = lctips.detrend_lc(lc, npoint=10)
        dl.append(det_lc)
    y2  = np.hstack(dl)
    y2  = lctips.remove_flare(y2, 4)
    y2  = lctips.remove_lowest(y2, 2)

    #plt.scatter(data[0],data[1],s=5)
    #plt.scatter(dl_st[0], dl_st[1],s=3)
    #plt.show()
    #exit()

    #fp,_,freq,pgram    = period.lomb_scargle(y2, N=1000)
    #y2      = fft.peak_filter(y2, fp)
    y2[0]   = copy.copy(data[0])
    #plt.scatter(y2[0], y2[1], s=3)
    #plt.show()
    #exit()

    dt  = data[0,1] - data[0,0]
    n_window    =  int(1//dt)
    med1    = lctips.median1d(y2[1], n_window)
    #plt.scatter(y2[0], y2[1]+1, s=3)
    y2[1]   = y2[1] - med1 + 1
    #plt.scatter(y2[0], y2[1], s=3)
    #plt.scatter(y2[0], med1 + 1, s=2)
    #plt.show()
    #exit()

    return y2

def lc_bin(data, nbin):
    data[1]     = data[1] - np.median(data[1])
    sp_lc       = lctips.split_discon(data)

    result  = []
    bl = []
    for lc in sp_lc:
        bin_lc = lctips.bin_lc(lc,nbin)
        bl.append(bin_lc)
    binlc = np.hstack((bl))

    return  binlc



epicid  = "211414619"
#lc      = getlc.k1lc_byepic(epicid)
#lc      = getlc.tesslc_byepic(epicid)

datak2      = np.loadtxt(sys.argv[1], dtype='f8', comments='#').T
datatess    = np.loadtxt(sys.argv[2], dtype='f8', comments='#').T
lck2    = np.array((datak2[0], datak2[1]))
lctess  = np.array((datatess[0], datatess[2]))

nbin    = int((lck2[0,1] - lck2[0,0])//(lctess[0,1] - lctess[0,0]))
lc_tmp  = lc_bin(lctess, nbin)
lctess  = copy.copy(lc_tmp)

print(lctess)

resk2   = lc_detrend(lck2)
restess = lc_detrend(lctess)

resk2[0]    += 2454833
#print(resk2)
#print(restess)

result  = np.hstack((resk2, restess))
result2 = period.mask_transit(result, 4.01799, 0.01, 2458100.068145902)

plt.scatter(result[0], result[1], s=3)
plt.scatter(result2[0], result2[1], s=3)
plt.show()
exit()
#result = resk2
#result = restess

tls_spc = period.tls_periodogram(result2, rad_star=0.1965, mas_star=0.1635)
sde  = tls_spc.SDE
pgram  = np.array((tls_spc.periods, tls_spc.power))
TO  = tls_spc.T0
depth = tls_spc.depth
duration = tls_spc.duration

mt = tls_spc.model_lightcurve_time
mf = tls_spc.model_lightcurve_model

print("#sde, TO, depth, duration")
print(sde, TO, depth, duration)
#plt.plot(pgram[0], pgram[1])
#plt.show()
plt.savefig("tls_periodgram.png", dpi=200)

np.savetxt("det_lc.dat", result.T)
np.savetxt("pgram.dat", pgram.T)
np.savetxt("model.dat", np.array((mt, mf)).T)


