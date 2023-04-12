#!/usr/bin/env python

import numpy as np
from matplotlib import pyplot as plt

import sys

import getlc, fft, lctips, period

epicid  = "247122957"
#epicid  = "210721261"
#epicid  = "206036749"

lck2    = getlc.k2lc_byepic(epicid)
lctess  = getlc.tesslc_byepic(epicid)
#plt.plot(lck2[0], lck2[1])
#plt.show()
#exit()

lck2[1]     = lck2[1] - np.median(lck2[1])
lctess[1]   = lctess[1] - np.median(lctess[1])


#sp_lc       = lctips.split_discon(lctess)
sp_lc       = lctips.split_discon(lck2)
result  = []
dl = []
for lc in sp_lc:
    det_lc  = lctips.detrend_lc(lc)
    dl.append(det_lc)

dl_st   = np.hstack(dl)
fp,_,freq,pgram    = period.lomb_scargle(dl_st, N=1000)

#for det_lc in dl:
#    lc_new  = fft.peak_filter(det_lc,fp)
#    result.append(lc_new)
#result  = np.hstack(result)

result  = fft.peak_filter(dl_st, fp)
#plt.plot(lctess[0], result[1])
#plt.plot(lctess[0], lctess[1])
#plt.plot(lctess[0], lctess[1] - result[1])
plt.plot(dl_st[0], result[1], lw = 2)
plt.plot(dl_st[0], dl_st[1], lw =1)
plt.plot(dl_st[0], dl_st[1] - result[1])
plt.show()
exit()

#fft.plot_freq(lck2)
test_lc     = fft.whitenoise_sigma(lctess,3)

#fft.plot_freq(test_lc)

plt.plot(test_lc[0]+lctess[0,0], test_lc[1])
plt.plot(lctess[0], lctess[1])
plt.show()
