#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import sys

from scipy.interpolate import interp1d, interp2d
import bin.iso_utils as iu

def makefluxmodel_FeH(fluxmodel, FeH, FeHa=0.0, FeHb=0.5):

    modF00  = fluxmodel[(fluxmodel[:,1]==FeHa)]
    modF05  = fluxmodel[(fluxmodel[:,1]==FeHb)]
    #print(len(modF00))
    #print(len(modF05))
    #exit()
    
    Tefflist    = np.unique(modF00[:,0])
    logglist    = np.unique(modF00[:,2])

    res = []
    for it in Tefflist:
        res_t   = []
        for ig in logglist:
            f00 = modF00[((modF00[:,0]==it)&(modF00[:,2]==ig)),-2:]
            f05 = modF05[((modF05[:,0]==it)&(modF05[:,2]==ig)),-2:]
            if (len(f00) > 0) & (len(f05) > 0):
                f00 = f00[0]
                f05 = f05[0]

                c1  = (FeH  - FeHa)
                c2  = (FeHb - FeH)
                c3  = (FeHb - FeHa)
                ftmp    = (c2*f00 + c1*f05)/c3
            else:
                ftmp    = 0,0
            restmp  = np.hstack((it, FeH, ig,ftmp))
            #res.append(restmp)
            res_t.append(restmp)
        
        res_t  = np.vstack(res_t)
        con     = (res_t[:,3]!=0.)
        f1      = interp1d(res_t[con,2], res_t[con,3], fill_value='extrapolate')
        f2      = interp1d(res_t[con,2], res_t[con,4], fill_value='extrapolate')
        for ig in logglist:
            restmp  = np.hstack((it, FeH, ig, f1(ig), f2(ig)))
            res.append(restmp)

    return np.vstack(res)

def makefluxmodel_grid(fluxmodel):
    tf_ar   = np.unique(fluxmodel[:,0])
    lg_ar   = np.unique(fluxmodel[:,2])
    f_redef = []
    for tf in tf_ar:
        cd  = (fluxmodel[:,0]==tf)
        y3  = interp1d(fluxmodel[cd,2], fluxmodel[cd,3], kind='linear', fill_value='extrapolate')
        y4  = interp1d(fluxmodel[cd,2], fluxmodel[cd,4], kind='linear', fill_value='extrapolate')
        for lg in lg_ar:
            f_redef.append([tf, 0.0, lg, y3(lg), y4(lg)])
    f_redef = np.array(f_redef)
    return f_redef

def makefluxmodel_Tefflogg(fluxmodel, Teff, logg):
    tf_ar   = np.unique(fluxmodel[:,0])
    lg_ar   = np.unique(fluxmodel[:,2])
    n       = len(tf_ar)
    m       = len(lg_ar)
    fKp     = fluxmodel[:,3].reshape(n,m)
    fT      = fluxmodel[:,4].reshape(n,m)

    #func_Kp = interp2d(lg_ar, tf_ar, fKp, kind='linear', fill_value=True) 
    #func_T  = interp2d(lg_ar, tf_ar, fT,  kind='linear', fill_value=True) 
    func_Kp = interp2d(lg_ar, tf_ar, fKp, kind='cubic', fill_value=True) 
    func_T  = interp2d(lg_ar, tf_ar, fT,  kind='cubic', fill_value=True) 
    res     = []
    for lg, tf in zip(logg, Teff):
        fKpout  = func_Kp(lg, tf)[0]
        fTout   = func_T(lg, tf)[0]
        res.append([[fKpout, fTout]])
    
    res     = np.vstack(res)

    return res

def amplitude(Fph, Fsp, S=0.1):
    h   = (1. - Fsp/Fph)*S/2.
    return h

def main(MISTdata, fluxmodel, Thost=3500., Tspot=3400.):
    logg    = iu.logg(MISTdata, Thost)
    Fph_Kp,Fph_T  = makefluxmodel_Tefflogg(fluxmodel, Thost, logg)
    Fsp_Kp,Fsp_T  = makefluxmodel_Tefflogg(fluxmodel, Tspot, logg)

    hKp     = amplitude(Fph_Kp, Fsp_Kp)
    hT      = amplitude(Fph_T, Fsp_T)

    print("T_ph   :", Thost)
    print("T_sp   :", Tspot)
    print("logg   :", logg)
    print("hT/hKp :", hT/hKp)


if __name__=='__main__':
    cmdfile     = sys.argv[1]
    fluxfile    = "bin/data/fluxdata/flux.dat"
    fluxmodel0  = np.loadtxt(fluxfile)
    fluxmodel   = makefluxmodel_FeH(fluxmodel0, FeH=0.3)

    MISTdata    = np.loadtxt(cmdfile, comments='#', dtype='f8')
    main(MISTdata, fluxmodel)