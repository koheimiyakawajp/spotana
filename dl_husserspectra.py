#!/usr/bin/env python3

from gzip import FNAME
import os
from multiprocessing import Pool
from typing import no_type_check_decorator
from urllib.error import URLError
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import numpy as np
import subprocess
from scipy.interpolate import interp1d
import copy

link2   =   "ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS//WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"

l1="ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z"
"ftp://phoenix.astro.physik.uni-goettingen.de/MedResFITS/A1FITS/PHOENIX-ACES-AGSS-COND-2011_A1FITS_Z-0.0.zip"
l2="Z-0.0/lte03000-0.00-0.0"
l3=".PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"

#FeHlist =   ["-2.0","-1.5","-1.0","-0.5","-0.0","+0.5", "+1.0"]
FeHlist =   ["-0.0", "+0.5"]
#teflist =   np.linspace(3000, 5000, 21, dtype='i8')
teflist =   np.linspace(5100, 5600, 6, dtype='i8')
#teflist =   np.linspace(3000, 4000, 6, dtype='i8')
teflist =   np.array(teflist, dtype='unicode')
glist   =   np.linspace(3.0,6.0,7, dtype='unicode')
#glist   =   ["4.0"]

linklist    = []
savelist    = []
setlist     = []
for FeH in FeHlist:
    for tef in teflist:
        for g in glist:
            link    = l1+FeH+"/lte0"+tef+"-"+g+"0"+FeH+l3
            savef   = "lte0"+tef+"-"+g+"0"+FeH+"_husser2011.dat"
            set_ar  = [FeH, tef, g]
            linklist.append(link)
            savelist.append(savef)
            setlist.append(set_ar)

def dl_fits(link):
    fname   = "fits/"+link.split("/")[-1]
    res     = subprocess.run("curl -s "+link+" -o "+fname, shell=True)
    if res.returncode == 0:
        data    = getprime(fname)
        return data
    else:
        return -1

def getprime(fname):
    hdulist = fits.open(fname)
    hdu     = hdulist[0]
    return hdu.data

def cal_flux(specdata, filter):
    spec_n  = copy.copy(specdata[:,\
        ((filter[0,0]<specdata[0])&(specdata[0]<filter[0,-1]))])
    wav     = spec_n[0]
    ffunc   = interp1d(filter[0], filter[1])
    flt_v   = ffunc(wav)

    f_obs   = flt_v*spec_n[1]

    val     = f_obs[:-1]*(wav[1:]-wav[:-1])*wav[:-1]
    n_phot  = np.sum(val)

    return n_phot

SPECDIR="./bin/data/spectra/"

def cal_flux_per_spec(input_list):
    index,wave,Kpfilter,Tfilter  = input_list

    res     = subprocess.run("ls "+SPECDIR+" | grep "+savelist[index], shell=True)
    flg     = res.returncode
    #if flg==0:
    #    data        = np.loadtxt(SPECDIR+savelist[index], dtype='f8')
    #else:
    if flg != 0:
        data    = dl_fits(linklist[index])
        print("downloading "+savelist[index])
        if len(data) > 1:
            #if flg==0:
            #    specdata    = data.T
            #else:
            #    specdata    = np.array((wave,data), dtype='f8')
            specdata    = np.array((wave,data), dtype='f8')
            #fKp     = cal_flux(specdata, Kpfilter)
            #fT      = cal_flux(specdata, Tfilter)
            #pr      = setlist[index]
            np.savetxt(SPECDIR+savelist[index], specdata.T)

        #return [pr[0], pr[1], pr[2], fKp, fT]


if __name__=='__main__':
    Kpfilter    = np.loadtxt("./bin/data/filter/Kp.tsv", dtype='f8', comments="#").T
    Kpfilter[0] = Kpfilter[0]*10 #angstrom
    Tfilter     = np.loadtxt("./bin/data/filter/T.csv", dtype='f8', comments='#', delimiter=',').T
    Tfilter[0]  = Tfilter[0]*10 #angstrom

    wave    = dl_fits(link2)
    #data_list   = [(x, wave) for x in arange(len(linklist))]
    data_list   = [(x, wave, Kpfilter, Tfilter) for x in range(len(linklist))]
    p   = Pool(os.cpu_count())
    #outlist     = p.map(cal_flux_per_spec, data_list)
    p.map(cal_flux_per_spec, data_list)
    p.close()
    #outlist     = np.array(outlist, dtype='f8')
    #np.savetxt("./bin/data/flux.dat", outlist)

    #outlist = []
    #for i in range(len(linklist)):
    #    res     = subprocess.run("ls "+SPECDIR+" | grep "+savelist[i], shell=True)
    #    flg     = res.returncode
    #    if flg==0:
    #        data        = np.loadtxt(SPECDIR+savelist[i], dtype='f8')
    #    else:
    #        data    = dl_fits(linklist[i])
    #        print("downloading "+savelist[i])
    #    if len(data) > 1:
    #        if flg==0:
    #            specdata    = data.T
    #        else:
    #            specdata    = np.array((wave,data), dtype='f8')
    #        fKp     = cal_flux(specdata, Kpfilter)
    #        fT      = cal_flux(specdata, Tfilter)
    #        pr      = setlist[i]
    #        outlist.append([pr[0], pr[1], pr[2], fKp, fT])
    #        np.savetxt(SPECDIR+savelist[i], specdata.T)


