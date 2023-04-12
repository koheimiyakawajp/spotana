#!/usr/bin/env python3

from matplotlib import pyplot as plt
import numpy as np
import lightkurve as lk
import pandas as pd
import sys

def obj_to_f(obj):
    time_f  = []
    for t in obj:
        t   = float(str(t))
        time_f.append(t)
    time    = np.array(time_f, dtype='f8')

    return time

def flux_relative(flux, flux_er):
    median  = np.median(flux[(flux>0)])
    f   = flux/median
    er  = flux_er/median

    return f,er

def remove_nan(data):
    data    = data[:,(data[1]>0)]
    return data

def dl_lightcurve(sr_object):
    lcf     = sr_object.download()
    time    = lcf.time
    time    = obj_to_f(time)
    flux,er     = flux_relative(lcf.flux, lcf.flux_err)

    data    = np.array((time,flux,er), dtype='f8')
    return data

def merge_lightcurves(search_result, author):
    TESS_search_result  = search_result[(search_result.author==author)]

    time_tot  = []
    flux_tot  = []
    eror_tot  = []
    for sr  in TESS_search_result:
        data    = dl_lightcurve(sr)
        time_tot.extend(data[0])
        flux_tot.extend(data[1])
        eror_tot.extend(data[2])
    data    = np.vstack((time_tot,flux_tot,eror_tot))
    data    = remove_nan(data)
    return data

def EPIC_to_TIC(EPICid):
    df      = pd.read_csv("./k2ticxmatch_20210831.csv",dtype='unicode')
    target  = df[df['epic']==EPICid]
    if target=='':
        return -1
    else:
        tid     = target.iat[0,0]
        return "TIC "+tid

def tesslc_byepic(epicid):
    TIC     = EPIC_to_TIC(epicid)
    search_result = lk.search_lightcurve(TIC)
    data    = merge_lightcurves(search_result, 'SPOC')
    return data

def k2lc_byepic(epicid):
    search_result = lk.search_lightcurve("EPIC "+epicid)
    data    = merge_lightcurves(search_result, 'K2SFF')
    return data

if __name__=='__main__':
    epicid  = sys.argv[1]
    data    = k2lc_byepic(epicid)

    #data    = tesslc_byepic(epicid)
    plt.errorbar(data[0], data[1], yerr=data[2], fmt='.', ms=1, color='gray')
    plt.show()

