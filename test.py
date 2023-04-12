#!/usr/bin/env python3

import astropy.io.fits as fits
import numpy as np
import matplotlib.pyplot as plt

import sys

fname1   = sys.argv[1]
hdulist = fits.open(fname1)
hdu     = hdulist[0]
data    = hdu.data

fname2   = sys.argv[2]
hdulist2 = fits.open(fname2)
hdu2    = hdulist2[0]
wave  = hdu2.data
plt.plot(wave, data)
plt.show()