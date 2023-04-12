#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
import sys

def mass(MISTdata, Teff):
    con1    = ((MISTdata[:,3]<2)&(10**MISTdata[:,4]<6000)&(10**MISTdata[:,6]<3))
    mass_ar = MISTdata[con1,3]
    teff_ar = 10**MISTdata[con1,4]

    funci   = interp1d(teff_ar, mass_ar, kind='linear',fill_value="extrapolate")
    return  funci(Teff)

def logg(MISTdata, Teff):
    con1    = ((MISTdata[:,3]<2)&(10**MISTdata[:,4]<6000)&(10**MISTdata[:,6]<3))
    logg_ar = MISTdata[con1,5]
    teff_ar = 10**MISTdata[con1,4]

    funci   = interp1d(teff_ar, logg_ar, kind='linear',fill_value="extrapolate")
    return  funci(Teff)

def rad(MISTdata, Teff):
    con1    = ((MISTdata[:,3]<2)&(10**MISTdata[:,4]<6000)&(10**MISTdata[:,6]<3))
    L_ar    = 10**MISTdata[con1,6]
    teff_ar = 10**MISTdata[con1,4]

    teff_s  = 5772.
    funci   = interp1d(teff_ar, L_ar, kind='linear',fill_value="extrapolate")
    L       = funci(Teff)
    R       = (Teff/teff_s)**(-2.) * (L)**0.5

    return R

def main():
    fname   = sys.argv[1]
    data    = np.loadtxt(fname, dtype='f8', comments='#')

    con1    = ((data[:,3]<2)&(10**data[:,4]<6000)&(10**data[:,6]<3))
    fig     = plt.figure(figsize=(8,4))
    ax1     = fig.add_subplot(1,2,1)
    ax2     = fig.add_subplot(1,2,2)
    ax1.plot(10**data[con1,4], data[con1,5])
    ax2.plot(10**data[con1,4],data[con1,3])

    plt.show()


if __name__=='__main__':
    #main()
    #exit()
    fname   = sys.argv[1]
    data    = np.loadtxt(fname, dtype='f8', comments='#')
    print(logg(data, 3000))
    print(mass(data, 3000))
    print(logg(data, 3200))
    print(mass(data, 3200))
    print(logg(data, 3500))
    print(mass(data, 3500))
    print(rad(data, 3500))
    print(logg(data, 4000))
    print(mass(data, 4000))
    print(rad(data, 4000))
    print(logg(data, 5000))
    print(mass(data, 5000))
    print(rad(data, 5000))