#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.interpolate import interp1d
import copy


def get_f_perT(fluxmodel, FeH, logg):
    FeHlist     = np.sort(np.unique(fluxmodel[1]))
    logglist    = np.sort(np.unique(fluxmodel[2]))
    print(FeHlist)
    print(logglist)

    i   = 0 
    for fi in FeHlist:
        for loggi in logglist:
            fap     = fluxmodel[:,((fluxmodel[1]==fi)&(fluxmodel[2]==loggi))].T
            if i == 0:
                far = copy.copy(fap)
            else:
                far = np.vstack((far, fap))
            i   += 1
    print(far)

def amplitude(fsp, feff, S):
    fph     = (2*feff - fsp*S)/(2-S)
    h       = (1 - fsp/fph)*S/2

    return h




fmodel  = np.loadtxt("./bin/data/fluxdata/flux.dat", dtype='f8', comments='#').T
#get_f_perT(fmodel, 0, 0)
#exit()

fF0g4   = fmodel[:,((fmodel[1]==0.0)&(fmodel[2]==4))]
#fF0g4   = fmodel[:,((fmodel[1]==0.5)&(fmodel[2]==4))]

Kp      = interp1d(fF0g4[0], fF0g4[3], kind='cubic')
T       = interp1d(fF0g4[0], fF0g4[4], kind='cubic')

teff_list   = [3400., 3800., 4200., 4600., 5000.]
plt.figure(figsize=(5.5,5))
plt.rcParams["font.family"] = "Arial"   # 使用するフォント
for s in [0.01, 0.05]:#, 0.1, 0.5]:
    for teff in teff_list:
        tf_sp   = np.arange(3000,teff,10)

        #hKp = (1 - Kp(tf_sp)/Kp(teff))*0.05
        #hT  = (1 - T(tf_sp)/T(teff))*0.05
        hKp = amplitude(Kp(tf_sp), Kp(teff), s)
        hT  = amplitude(T(tf_sp), T(teff),s)

        #plt.plot(tf_sp, hT/hKp, label=str(int(teff))+' K')
        plt.plot(tf_sp, hT, label=str(int(teff))+' K')
        plt.plot(tf_sp, hKp, label=str(int(teff))+' K')

#plt.title("FeH:0.0, log(g):4.0", fontsize=14)
plt.xlabel("$T_{spot}$ [K]", fontsize=12)
plt.ylabel("$h_{T}/h_{Kp}$", fontsize=12)
plt.legend(fontsize=10)
plt.show()
#plt.savefig("FeH00logg40.png", dpi=200)
