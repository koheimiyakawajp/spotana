#!/usr/bin/env python3

from matplotlib import pyplot as plt
import numpy as np
import lightkurve as lk
import pandas as pd
import sys
import math

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

def dl_lightcurve(sr_object, fpdc=True):
    lcf     = sr_object.download()
    print(lcf)
    t_ofs   = lcf.meta['BJDREFI']
    #print(lcf.meta["CROWDSAP"])
    #print(lcf.meta["FLFRCSAP"])
    time    = lcf.time
    time    = obj_to_f(time)
    if fpdc:
        flux_rw = lcf.flux
        er_rw   = lcf.flux_err
    else:
        flux_rw = lcf.sap_flux
        er_rw   = lcf.sap_flux_err

    flux,er     = flux_relative(flux_rw, er_rw)
    data    = np.array((time+t_ofs, flux, er, flux_rw), dtype='f8')
    return data

def merge_lightcurves(search_result, author, texp, fpdc=True):
    TESS_sr1  = search_result[(search_result.author==author)]
    print(TESS_sr1.mission)

    if len(TESS_sr1) != 0:
        exp_ar  = np.array(TESS_sr1.exptime,dtype='f8')
        TESS_search_result  = TESS_sr1[(exp_ar==float(texp))]
        if len(TESS_search_result) == 0:
            return [0]
    else:
        return [0]
    
    time_tot  = []
    flux_tot  = []
    eror_tot  = []
    fraw_tot  = []
    for sr  in TESS_search_result:
        data    = dl_lightcurve(sr, fpdc=fpdc)
        time_tot.extend(data[0])
        flux_tot.extend(data[1])
        eror_tot.extend(data[2])
        fraw_tot.extend(data[3])
    data    = np.vstack((time_tot, flux_tot, eror_tot, fraw_tot))
    data    = remove_nan(data)
    return data

def EPIC_to_TIC(EPICid):
    df      = pd.read_csv("./k2ticxmatch_20210831.csv",dtype='unicode')
    target  = df[df['epic']==EPICid]
    #if math.isnan(float(target.iat[0,0])):
    if len(target) == 1:
        tid     = target.iat[0,0]
        return "TIC "+tid
    else:
        return -1

def tesslcTESSSPOC_byepic(epicid, fpdc=True):
    TIC     = EPIC_to_TIC(epicid)
    search_result = lk.search_lightcurve(TIC)
    if np.any(search_result.author=='SPOC'):
        tarname = np.unique(search_result.target_name[(search_result.author=='SPOC')])[0]
    else:
        tarname = np.unique(search_result.target_name[(search_result.author=='TESS-SPOC')])[0]
        
    search_result = search_result[(search_result.target_name==tarname)]
    data    = merge_lightcurves(search_result, 'TESS-SPOC', '600.', fpdc=fpdc)

    return data

def tesslcQLP_byepic(epicid, fpdc=True):
    TIC     = EPIC_to_TIC(epicid)
    #print(TIC)
    search_result = lk.search_lightcurve(TIC)
    #print(search_result)
    if np.any(search_result.author=='SPOC'):
        tarname = np.unique(search_result.target_name[(search_result.author=='SPOC')])[0]
    elif np.any(search_result.author=='QLP'):
        tarname = np.unique(search_result.target_name[(search_result.author=='QLP')])[0]
    else:
        return [0]
        
    #print(tarname)
    search_result = search_result[(search_result.target_name==tarname)]
    data    = merge_lightcurves(search_result, 'QLP', '600.', fpdc=fpdc)

    return data

def tesslc_byepic(epicid, fpdc=True):
    TIC     = EPIC_to_TIC(epicid)
    search_result = lk.search_lightcurve(TIC)
    search_result = search_result[(search_result.target_name==TIC[4:])]
    if np.all(search_result.author!='SPOC'):
        return [0]
    
    data    = merge_lightcurves(search_result, 'SPOC', '120', fpdc=fpdc)

    return data

def check_contaminant(epicid):
    TIC     = EPIC_to_TIC(epicid)
    search_result = lk.search_lightcurve(TIC)
    search_result = search_result[(search_result.target_name==TIC[4:])]
    if np.all(search_result.author!='SPOC'):
        return [0]
    else:
        search_result   = search_result[(search_result.author=='SPOC')]
    
    
    flgs    =[]
    nsr     = 0
    for sr in search_result:
        #lcf     = search_result.download()
        lcf     = sr.download()
        cs  = lcf.meta["CROWDSAP"]
        fs  = lcf.meta["FLFRCSAP"]
        flgs.append([cs, fs])
        nsr += 1
    flgs    = np.array(flgs)
    if nsr > 1:
        flgs    = np.mean(flgs,axis=0)
    
    return flgs
    



def k2lc_byepic(epicid, author='K2SFF', fpdc=True):
    search_result = lk.search_lightcurve("EPIC "+epicid)
    data    = merge_lightcurves(search_result, author,'1800', fpdc=fpdc)
    return data

if __name__=='__main__':
    epicid  = sys.argv[1]
    #data   = k2lc_byepic(epicid, author='K2', fpdc=True)
    #data2  = k2lc_byepic(epicid, author='K2', fpdc=False)
    #data3  = k2lc_byepic(epicid, author='K2SFF')
    #data2   = tesslcQLP_byepic(epicid)
    print(check_contaminant(epicid))
    exit()
    data    = tesslc_byepic(epicid, fpdc=False)
    #data3   = tesslc_byepic(epicid)

    plt.scatter(data[0], data[1], s=1, color='black', zorder=3)
    #plt.scatter(data[0], data[1], s=1, color='black', zorder=3)
    #plt.scatter(data2[0], data2[1], s=1, color='red', zorder=4)
    #plt.scatter(data3[0], data3[1], s=1, color='pink', zorder=5)
    #print(data3[1])
    plt.show()



