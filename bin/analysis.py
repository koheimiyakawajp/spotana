#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt

import sys
import requests

import getlc, fft, lctips, period, citeID
import spectra_analysis as sa

URL="http://www.lamost.org/dr8/v2.0/spectrum/fits2csv/"
epicid_array = np.loadtxt(sys.argv[1], dtype='i8').flatten()
lmst_idmjd  = citeID.lamostid_byepic(epicid_array)

for line in lmst_idmjd:
    lmstid  = str(int(line[3]))
    tarurl  = URL+lmstid
    sp_req  = requests.get(tarurl)
    page    = sp_req.text
    sp1     = page.split('\n')
    sp1     = sp1[1:-1]
    spec    = []
    for l1  in sp1:
        spec.append(l1.split(","))
    spec    = np.array(spec, dtype='f8')
    CaII    = sa.ext_CaIIHK(spec)
    Halp    = sa.ext_Halpha(spec)
    FeII    = sa.ext_FeII(spec)
    #plt.plot(CaII[:,0], CaII[:,1])
    #plt.plot(Halp[:,0], Halp[:,1])
    plt.plot(FeII[:,0], FeII[:,1])
    plt.show()
    print(spec)
    exit()
#for epicid in epicid_array:
#    citeID.lamostid_byepic(epicid)
exit()
flg     = getlc.EPIC_to_TIC(epicid)
print(flg)
exit()

lck2    = getlc.k2lc_byepic(epicid)
lctess  = getlc.tesslc_byepic(epicid)

lck2[1]     = lck2[1] - np.median(lck2[1])
lctess[1]   = lctess[1] - np.median(lctess[1])


lc_array    = (lck2, lctess)
for lc in lc_array:
    sp_lc       = lctips.split_discon(lc)

    dl = []
    for lc_parts in sp_lc:
        det_lc  = lctips.detrend_lc(lc_parts)
        cln_lc  = lctips.remove_outlier(det_lc)
        dl.append(cln_lc)

    #result   = np.hstack(dl)
    dl_st   = np.hstack(dl)
    fp,_,freq,pgram    = period.lomb_scargle(dl_st, N=1000)
    #print(fp)
    #exit()
    result  = fft.peak_filter(dl_st, fp)
    fft.plot_freq(dl_st)
    exit()
    plt.plot(dl_st[0], result[1], lw = 2)
    plt.plot(dl_st[0], dl_st[1], lw = 2)
    plt.show()
    exit()

#fft.plot_freq(lck2)
test_lc     = fft.whitenoise_sigma(lctess,3)

#fft.plot_freq(test_lc)

plt.plot(test_lc[0]+lctess[0,0], test_lc[1])
plt.plot(lctess[0], lctess[1])
plt.show()
