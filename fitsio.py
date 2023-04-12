#!/usr/bin/env python3

from urllib.error import URLError
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import sys
import numpy as np

l1="ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/"
l2="Z-0.0/lte03000-0.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"

def getprime(fname):
    hdulist = fits.open(fname)
    hdu     = hdulist[0]
    return hdu.data

if __name__=='__main__':
    if len(sys.argv) == 3:
        ffits   = sys.argv[1]
        wfits   = sys.argv[2]

        f       = getprime(ffits)
        w       = getprime(wfits)
        
        plt.scatter(w,f, s=0.3)
        plt.show()

        exit()

        print(hdulist.info())
        print(hdulist["PRIMARY"])
        #print(hdu.keys())