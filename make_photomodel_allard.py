#!/usr/bin/env python3

import copy
import os
import re
import subprocess
import sys
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d, interp2d

SPECDIR="./bin/data/spectra/models_1680746720/bt-settl/"

def cal_flux_KpT(specfile_list):

    Kpfilter    = np.loadtxt("./bin/data/filter/Kp.tsv", dtype='f8', comments="#").T
    Kpfilter[0] = Kpfilter[0]*10 #angstrom
    Tfilter     = np.loadtxt("./bin/data/filter/T.csv", dtype='f8', comments='#', delimiter=',').T
    Tfilter[0]  = Tfilter[0]*10 #angstrom

    arg_list    = [(x, Kpfilter, Tfilter) for x in specfile_list]
    
    p   = Pool(os.cpu_count())
    res = p.map(cal_flux, arg_list)
    p.close()

    return np.array(res)

def cal_n_phot(specdata, filter):
    spec_n  = copy.copy(specdata[:,\
        ((filter[0,0]<specdata[0])&(specdata[0]<filter[0,-1]))])
    wav     = spec_n[0]
    ffunc   = interp1d(filter[0], filter[1])
    flt_v   = ffunc(wav)

    f_obs   = flt_v*spec_n[1]*1e8

    val     = f_obs[:-1]*(wav[1:]-wav[:-1])*wav[:-1]
    n_phot  = np.sum(val)

    return '{:.5g}'.format(n_phot)

def cal_flux(arg_list):
    specfile, Kpfilter, Tfilter    = arg_list
    specdata    = np.loadtxt(SPECDIR+specfile, dtype='f8').T

    np_Kp   = cal_n_phot(specdata, Kpfilter)    
    np_T    = cal_n_phot(specdata, Tfilter)    

    FeH     = '{:.2f}'.format(float(specfile[10:14]))
    logg    = '{:.2f}'.format(float(specfile[7:10]))

    return [FeH, logg, np_Kp, np_T]

def main():
    #teflist =   np.linspace(2800, 5500, 28, dtype='i8')
    teflist =   np.linspace(25, 60, 36, dtype='i8')
    res_mtx = np.array([[0,0,0,0,0]])
    for teff in teflist:
        print("processing " + str(teff) +"00 K.")
        subpro  = subprocess.run("ls "+SPECDIR+" | grep lte0"+ str(teff), \
            shell=True, stdout=subprocess.PIPE)
        speclist    = str(subpro.stdout)
        speclist    = speclist[2:-1]
        spec_array  = speclist.split("\\n")
        spec_array  = spec_array[:-1]
        #tmp = spec_array[0]
        #print(tmp)
        #print(tmp[7:10])
        #print(tmp[10:14])
        #exit()

        res     = cal_flux_KpT(spec_array)
        vlen    = len(res[:,0])
        t_ar    = np.full(vlen, '{:.2f}'.format(float(teff*100)))
        t_ar    = t_ar.reshape((-1,1))


        res     = np.hstack((t_ar, res))
        res_mtx = np.vstack((res_mtx, res))

    #print(res_mtx)
    np.savetxt("flux.dat", res_mtx[1:], fmt="%s")
    print("result is saved in flux.dat")



if __name__=='__main__':

    if len(sys.argv) == 1:
        main()
    else:
        print("make photomodel")
        print(sys.argv[0])