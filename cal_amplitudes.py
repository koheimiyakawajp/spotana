#!/usr/bin/env python3

import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from copy import copy
from scipy.signal import argrelextrema, medfilt

import bin.getlc as gl
import bin.lctips as lt
import bin.fft as ft
import bin.period as pr

def ylimits(ydata):
    med     = np.median(ydata)
#    mad     = np.median(np.abs(np.median(ydata) - ydata))
    ysort   = np.sort(ydata)
    dlen    = len(ydata)
    sig1    = ysort[int(dlen*0.1585)] - med
    sig2    = ysort[-int(dlen*0.1585)] - med

    return med+4*sig1, med+4*sig2

def axplot_wn(ax, toff, data, data_rot, data_red):
    res     = data[1] - data_rot[1]
    dtot    = int(len(res))
    ax.scatter(data[0]-toff ,res, s=0.5,c='dimgrey')
    ax.scatter(data[0]-toff ,data_red[1], s=0.3,c='orangered')
    std_th  = np.std(res)
    med     = np.median(res)
    res_srt = np.sort(res)
    std_ms0 = res_srt[-int(0.1585*dtot)]
    std_ms1 = res_srt[int(0.1585*dtot)]
    #ax.axhline(med - std_th, ls=':', c='blue', lw=0.8)
    #ax.axhline(med + std_th, ls=':', c='blue', lw=0.8)
    #ax.axhline(std_ms1, ls=':', c='red', lw=0.8)
    #ax.axhline(std_ms0, ls=':', c='red', lw=0.8)

    ax.set_ylim((med - std_th*4, med + std_th*4))


def plotfunc(lc ,lc_1, lc_rot, lc_red, k2id, tid):
    lck2    = lc[0]
    toff    = int(lck2[0,0])
    fig,axes    = plt.subplots(4,2,tight_layout=True, figsize=(13,6.5))
    #plt.rcParams["font.family"] = "cmss10"   # 使用するフォント
    plt.rcParams["font.size"] = 10  
    tarray  = ["K2SFF", "K2 SAP", "TESS SAP", "TESS QLP"]
    for i in range(4):
        if lc[i] is not np.nan:

            lcp     = lc[i]
            lcp_1   = lc_1[i]
            lcp_rot = lc_rot[i]
            lcp_red = lc_red[i]

            axes[i,0].scatter(lcp[0]-toff,lcp[1],s=0.7,c="black")
            axes[i,0].scatter(lcp_1[0]-toff,lcp_1[1]+1,s=0.5,c="dimgrey")
            axes[i,0].scatter(lcp_1[0]-toff,lcp_rot[1]+1,s=0.1,c="dodgerblue")
            axplot_wn(axes[i,1], toff, lcp_1, lcp_rot, lcp_red)

            m,u,l = mes_amplitude(lcp_rot[1])
            #axes[i,0].axhline(m+1, c='black',ls='--',lw=0.5)
            #axes[i,0].axhline(u+1, c='black',ls=':',lw=0.5)
            #axes[i,0].axhline(l+1, c='black',ls=':',lw=0.5)
            axes[i,0].set_ylim((ylimits(lcp[1])))
    fig.suptitle("EPIC "+ k2id+ " / "+ tid)
    fig.supxlabel('time - '+str(toff)+" [d]");fig.supylabel('relative flux')
    #plt.show()
    plt.savefig("figure/"+k2id+"lightcurve.png", dpi=200)
    plt.clf();plt.close()

def lc_clean(lc, timesep=1, bsep=10):
    sp_lc   = lt.split_discon(lc, timesep=timesep)
    dl      = []
    for lc_parts in sp_lc:
        binsep  = bsep #day
        trange  = lc_parts[0,-1] - lc_parts[0,0]
        npoint  = int(trange/binsep)
        #if npoint>2:
        #    det_lc  = lt.detrend_lc(lc_parts, npoint=npoint)
        #else:
        #    det_lc  = lc_parts
        #    det_lc[1]  -= np.median(det_lc[1])

        #det_lc  = lt.detrend_lc_quad(lc_parts)
        det_lc  = lt.detrend_lc_linear(lc_parts)

        #print("***** ", np.median(det_lc[1]))
        cln_lc  = lt.remove_outlier(det_lc, nsigma=5)
        dl.append(cln_lc)
    dl_st   = np.hstack(dl)
    return dl_st

def mes_amplitude(flux):
    fsort   = np.sort(flux)
    flen    = len(fsort)
    fu      = fsort[int(flen*0.95)]
    fl      = fsort[int(flen*0.05)]
    return np.median(flux), fl, fu

def roop_mes(data_nn, pbest):
    dlen    = data_nn[0,-1] - data_nn[0,0]
    roopn   = int(dlen/pbest) + 1
    t_beg   = data_nn[0,0]

    amp_ar  = []
    for i in range(roopn):
        t_a     = t_beg + i*pbest
        t_b     = t_beg + (i+1)*pbest
        #plt.axvline(t_b)
        con     = ((t_a<=data_nn[0])&(data_nn[0]<t_b))
        d_i     = data_nn[:,con]
        if np.any(con):
            #plt.scatter(d_i[0], d_i[1], s=2)
            if (d_i[0,-1] - d_i[0,0]) > 0.9*pbest:
                _,b,c   = mes_amplitude(d_i[1])
                amp     = np.abs(b-c)/2.
                amp_ar.append(amp)
    if len(amp_ar) == 0:
        return np.nan, np.nan
    elif len(amp_ar) == 1:
        return amp, np.nan
    else:
        return np.mean(amp_ar), np.std(amp_ar)

def mes_wrap(data, pres, wsigma=3):
    if pres.ndim == 2:
        pbest   = pres[0,0]
        prange  = pres[:,-2:]
    elif pres.ndim == 1:
        pbest   = pres[0]
        prange  = pres[-2:]
    data_nn,_   = ft.rm_whitenoise(data,wsigma)
    #data_rednoise   = ft.peakrange_filter(data_nn, prange)
    data_wn         = np.array((data[0], data[1] - data_nn[1]))
    #data_rednoise   = ft.highpass_filter(data_nn, 0.1)
    data_rednoise   = ft.highpass_filter(data_nn, max(0.1, pbest/5.))
    data_rotation   = np.array((data_rednoise[0], data_nn[1] - data_rednoise[1]))

    #fig = plt.figure()
    #ax1 = fig.add_subplot(2,1,1)
    #ft.plot_freq(data_rednoise, ax1, color="red")
    #ft.plot_freq(data_wn, ax1, color="gray")
    #ft.plot_freq(data_rotation, ax1, color="blue")
    #ax2 = fig.add_subplot(2,1,2)
    #ax2.scatter(data_wn[0], data_wn[1], c="gray", s=1)
    #ax2.scatter(data_rednoise[0], data_rednoise[1], c="red", s=1)
    #ax2.scatter(data_rotation[0], data_rotation[1], c="blue", s=1)
    #plt.show()

    rotamp,rotamp_er    = roop_mes(data_rotation, pbest)
    data_wn_sort        = np.sort(data_wn[1])
    dlen    = len(data_wn_sort)
    wnamp   = np.abs(data_wn_sort[-int(0.1585*dlen)] - data_wn_sort[int(0.1585*dlen)])/2.

    _,l,u   = mes_amplitude(data_rednoise[1])
    redamp  = np.abs(u-l)/2.
    print("rotamp", rotamp, rotamp_er)
    print("redamp", redamp)
    print("wnamp", wnamp)

    return data_rotation, data_rednoise, data_wn, rotamp, rotamp_er, redamp, wnamp

def peri_error_single(p,pow,p_best):
    i_p     = int(np.where(p==p_best)[0])
    i_min   = i_p
    i_max   = i_p
    for i in range(i_p):
        if (pow[i_max-1] >= pow[i_max]):
            break
        else:
            i_max -= 1
    for i in range(len(p) - i_p-1):
        if (pow[i_min+1] >= pow[i_min]):
            break
        else:
            i_min += 1

    return p[i_min], p[i_max]

def peri_error_thres(p,pow,p_best,thres):
    i_p     = int(np.where(p==p_best)[0])
    i_min   = i_p
    i_max   = i_p
    for i in range(i_p):
        if (pow[i_max-1] < thres):
            break
        else:
            i_max -= 1
    for i in range(len(p) - i_p-1):
        if (pow[i_min+1] < thres):
            break
        else:
            i_min += 1

    return p[i_min]-p_best, p[i_max]-p_best

def remove_harmonics(presult):
    inrange = []
    i       = 0
    presult_filtered    = []
    for pr in presult:
        #print(pr)
        if i==0:
            peri,_,_,_,lim1,lim2  = pr
            inrange.append([lim1,lim2])
            presult_filtered.append(pr)
        else:
            peri,_,er1,er2,lim1,lim2  = pr
            flg     = 0
            for rg in inrange:
                con21    = ((rg[0]/2.<peri+er2)&(peri+er1<rg[1]/2.))
                con22    = ((rg[0]*2.<peri+er2)&(peri+er1<rg[1]*2.))
                con31    = ((rg[0]/3.<peri+er2)&(peri+er1<rg[1]/3.))
                con32    = ((rg[0]*3.<peri+er2)&(peri+er1<rg[1]*3.))
                con41    = ((rg[0]/4.<peri+er2)&(peri+er1<rg[1]/4.))
                con42    = ((rg[0]*4.<peri+er2)&(peri+er1<rg[1]*4.))
                if con21|con22|con31|con32|con41|con42:
                    flg     = 1
                    break
            if flg == 0:
                presult_filtered.append(pr)
                inrange.append([lim1,lim2])

        i+=1
    return np.array(presult_filtered)

def match_K2peri(pgm_peak_ok, pbest_k2):
    pgm_bypwr   = copy(pgm_peak_ok)
    pgm_bypwr   = pgm_bypwr[:,np.argsort(pgm_peak_ok[1])[::-1]]
    p_resid     = np.abs(pgm_bypwr[0] - pbest_k2)
    plike_tess  = float(pgm_bypwr[0,(p_resid==np.min(p_resid))])

    return plike_tess

def period_analysis(data, title='--', k2input=0):
    #trange      = data[0,-1] - data[0,0]
    #pmax        = trange//2
    pmax        = 30
    pmin        = 0.1
    ngrid       = pmax//0.005
    _,_,p,pow   = pr.lomb_scargle(data,N=int(ngrid),pmin=pmin,pmax=pmax)
    pgm         = np.array((p,pow))
    print("calculating sigmin val.")
    sigmin      = pr.sigmin_bootstrap(data,N=int(ngrid),pmin=pmin,pmax=pmax,nboot=100, seed=300)
    #sigmin  = 1e-3
    peaks   = argrelextrema(pow, np.greater)

    pgm_peak_ok = copy(pgm[:,peaks])
    #print(pgm_peak_ok)
    pgm_peak_ok = pgm_peak_ok[:,(pgm_peak_ok[1]>sigmin)]#abovethres

    if len(pgm_peak_ok[0]) == 0:
        return np.nan
    if k2input!=0:
        p1      = match_K2peri(pgm_peak_ok, k2input)
    else:
        p1      = pgm_peak_ok[0,(pgm_peak_ok[1]==np.max(pgm_peak_ok[1]))]#best period
        print("K2max", p1)
    pgm_peak20  = pgm_peak_ok[:,((pgm_peak_ok[0]>=p1*0.80)&(pgm_peak_ok[0]<=p1*1.2))]#pm20% 
    pgm_peakan  = pgm_peak_ok[:,((pgm_peak_ok[0]<p1*0.80)|(pgm_peak_ok[0]>p1*1.2))]#outer than20% 
    #plt.plot(pgm[1],pgm[1],lw=1)
    #plt.scatter(pgm_peak_ok[0],pgm_peak_ok[1])
    #plt.scatter(pgm_peak20[0],pgm_peak20[1])
    #plt.scatter(pgm_peakan[0],pgm_peakan[1])
    #plt.show()
    #exit()

    pow_sort    = np.hstack((np.sort(pgm_peak20[1])[::-1],np.sort(pgm_peakan[1])[::-1]))
    #print(pow_sort[0])
    presult     = []
    for ps in pow_sort:
        if np.any(pgm_peak_ok[1] == ps):
            p_best      = float(pgm_peak_ok[0,(pgm_peak_ok[1]==ps)])
            #print(p_best)
            
            per1,per2 = peri_error_thres(pgm[0],pgm[1],p_best,ps/2.)
            lim1,lim2 = peri_error_single(pgm[0],pgm[1],p_best)
            presult.append([p_best,ps,per1,per2,lim1,lim2])

    presult     = np.array(presult)
    presult2    = remove_harmonics(presult)
    #print(presult2)
    #exit()

    plt.figure(figsize=(5,3))
    #plt.rcParams["font.family"] = "cmss10"   # 使用するフォント
    plt.rcParams["font.size"] = 10  
    plt.plot(pgm[0],pgm[1],lw=1.,c='black')
    plt.scatter(presult[:,0], presult[:,1],c='royalblue',s=10)
    plt.scatter(presult2[:,0], presult2[:,1],c='orangered',s=10)
    plt.axhline(sigmin, c='blue',ls=':', lw=1)
    plt.xscale('log')
    plt.title(title);plt.xlabel("Period [d]");plt.ylabel("LS Power")
    
    #plt.show()
    tword   = title.split(" ")
    plt.savefig("figure/"+tword[0]+tword[1]+"_pdgram.png", dpi=200)
    plt.clf();plt.close()

    return presult2

def lcbin(data):
    t   = copy(data[0])
    f   = copy(data[1])
    sp  = 0.02
    ilen    = int((t[-1] - t[0])/sp)
    #print(ilen)
    mean_ar = []
    for i in range(ilen):
        r1  = i*sp + t[0]
        r2  = (i+1)*sp + t[0]
        con = ((r1<=t)&(t<r2))
        #print(len(f[con]))
        if len(t[con]) >= 3:
            f_mean  = np.mean(f[con])
            t_mean  = np.mean(t[con])
            mean_ar.append((t_mean, f_mean))
    
    return np.vstack(mean_ar).T

        
if __name__=='__main__':
    fname   = sys.argv[1]
    dlist   = np.loadtxt(fname, dtype='unicode',comments='#',delimiter=',')
    epiclist    = dlist[1:,0]
    out_array1  = np.array(["k2id", "ampk2", "erk2", "redk2", "wnk2", \
                            "ampk2sp", "erk2sp", "redk2sp", "wnk2sp",\
                            "amptess", "ertess", "redtess", "wntess",\
                            "amptqlp", "ertqlp", "redtqlp", "wntqlp", \
                            "shotk2", "shottess", "pbest", "pbest_tqlp",\
                            "k2time", "tesstime"], dtype='unicode')
    i = 0
    np.set_printoptions(precision=4, floatmode='maxprec')

    for k2id in epiclist:
        tid     = gl.EPIC_to_TIC(k2id)
        k2time  = np.nan
        tesstime= np.nan
        if tid != -1:
            print("EPIC "+k2id, tid)
            fkey    = "lightcurves/"+k2id
            flg     = 0
            if os.path.isfile(fkey+"_k2sff.dat"): #---------------------- K2SFF
                print("loading k2 lightcurve.--------------")
                lck2    = np.loadtxt(fkey+"_k2sff.dat", dtype='f8').T
                if len(lck2)!=0:
                    print("processing K2 data.")
                    lck2_1      = lc_clean(lck2, timesep=1)
                    k2time      = (lck2[0,-1] + lck2[0,0])/2.
                    #ft.plot_freq(lck2_1)

                    fileperi    = "period/"+k2id+"_k2sff.dat"
                    if os.path.isfile(fileperi):
                        print("period file for K2 data already exists.")
                        pres_k2     = np.loadtxt(fileperi, dtype='f8')
                    else:
                        print("running period analysis for K2 data.")
                        lck2_1_ar   = lt.split_discon(lck2_1, timesep=100)
                        #pres_k2     = period_analysis(lck2_1,k2id + " K2")
                        pres_k2     = period_analysis(lck2_1_ar[0],k2id + " K2")
                        if pres_k2 is not np.nan:
                            np.savetxt(fileperi, pres_k2)
                    if pres_k2.ndim == 2:
                        pbest   = pres_k2[0,0]
                    elif pres_k2.ndim == 1:
                        pbest   = pres_k2[0]
                    print("measuring amplitude for K2 data.")
                    lck2_rot,lck2_red,lck2_wn,ampk2,erk2,redk2,wnk2 = mes_wrap(lck2_1, pres_k2, wsigma=3)
                    print("P_k2", pbest)

                    flg += 1
                
            if os.path.isfile(fkey+"_k2sap.dat"): #------------------ K2 SAP
                print("loading k2sap lightcurve.---------------")
                lck2sp  = np.loadtxt(fkey+"_k2sap.dat", dtype='f8').T
                if len(lck2sp)!=0:
                    
                    print("cal shotnoise")
                    phk2    = np.median(lck2sp[-1])*1800
                    shotk2  = phk2**(-0.5)

                    print("processing K2SAP data.")
                    lck2sp_1    = lc_clean(lck2sp, timesep=1)
                    #print("running period analysis for TESS data.")
                    #pres_tess   = period_analysis(lctess_1,k2id + " TESS")
                    print("measuring amplitude for K2SAP data.")
                    lck2sp_rot,lck2sp_red,lck2sp_wn,ampk2sp,erk2sp,redk2sp,wnk2sp = mes_wrap(lck2sp_1, pres_k2, wsigma=3)
                    #print(ampk2sp, erk2sp)
                    
                    flg += 1

            else:
                lck2sp,lck2sp_1,lck2sp_rot,lck2sp_red,lck2sp_wn,ampk2sp,erk2sp,redk2sp,wnk2sp,shotk2 \
                    = np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan

            pbest_tqlp  = np.nan
            if os.path.isfile(fkey+"_tess_qlp.dat"): #--------------- TESS QLP
                print("loading tess_qlp lightcurve--------------.")
                lctqlp  = np.loadtxt(fkey+"_tess_qlp.dat", dtype='f8').T
                if len(lctqlp)!=0:

                    print("processing TESS QLP data.")
                    lctqlp_1    = lc_clean(lctqlp, timesep=1)
                    fileperi    = "period/"+k2id+"_tess_qlp.dat"
                    if os.path.isfile(fileperi):
                        print("period file for TESS QLP data already exists.")
                        pres_tqlp   = np.loadtxt(fileperi, dtype='f8')
                    else:
                        print("running period analysis for TESS QLP data.")
                        pres_tqlp   = period_analysis(lctqlp_1,k2id + " TESS_QLP", k2input=pbest)
                        if pres_tqlp is not np.nan:
                            np.savetxt(fileperi, pres_tqlp)
                    if pres_tqlp.ndim == 2:
                        pbest_tqlp  = pres_tqlp[0,0]
                    elif pres_tqlp.ndim == 1:
                        pbest_tqlp  = pres_tqlp[0]
                    print("P_tqlp", pbest_tqlp)

                    print("measuring amplitude for TESS QLP data.")
                    lctqlp_rot,lctqlp_red,lctqlp_wn,amptqlp,ertqlp,redtqlp,wntqlp = mes_wrap(lctqlp_1, pres_tqlp, wsigma=3)

            else:
                lctqlp,lctqlp_1,lctqlp_rot,lctqlp_red,lctqlp_wn,amptqlp,ertqlp,redtqlp,wntqlp\
                    = np.nan,np.nan,np.nan,np.nan,np.nan, np.nan,np.nan,np.nan,np.nan

            if os.path.isfile(fkey+"_tess.dat"): #------------------ TESS SAP
                print("loading tess lightcurve.------------")
                lctess  = np.loadtxt(fkey+"_tess.dat", dtype='f8').T
                if len(lctess)!=0:
                    
                    tesstime = (lctess[0,0] + lctess[0,-1])/2.
                    print("cal shotnoise")
                    phtess   = np.median(lctess[-1])*120
                    shottess = phtess**(-0.5)

                    print("processing TESS data.")
                    lctess_1    = lc_clean(lctess, timesep=0.5)
                    #print("running period analysis for TESS data.")
                    #pres_tess   = period_analysis(lctess_1,k2id + " TESS")
                    if pbest_tqlp is np.nan:
                        lctess_bin  = lcbin(lctess_1)
                        fileperi    = "period/"+k2id+"_tess_sap.dat"
                        if os.path.isfile(fileperi):
                            print("period file for TESS SAP data already exists.")
                            pres_tqlp   = np.loadtxt(fileperi, dtype='f8')
                        else:
                            print("running period analysis for TESS SAP data.")
                            pres_tqlp   = period_analysis(lctess_bin,k2id + " TESS_SAP", k2input=pbest)
                            if pres_tqlp is not np.nan:
                                np.savetxt(fileperi, pres_tqlp)
                        if pres_tqlp.ndim == 2:
                            pbest_tqlp  = pres_tqlp[0,0]
                        elif pres_tqlp.ndim == 1:
                            pbest_tqlp  = pres_tqlp[0]
                    print("measuring amplitude for TESS data.")
                    lctess_rot,lctess_red,lctess_wn,amptess,ertess,redtess,wntess = mes_wrap(lctess_1, pres_tqlp, wsigma=3)
                    print("shotnoise", shottess)
                    flg  += 1

            else:
                lctess,lctess_1,lctess_rot,lctess_red,lctess_wn,amptess,ertess,redtess,wntess,shottess\
                    = np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan

        if flg == 3:
            output      = np.array([k2id, ampk2, erk2, redk2, wnk2, \
                            ampk2sp, erk2sp, redk2sp, wnk2sp,\
                            amptess, ertess, redtess, wntess,\
                            amptqlp, ertqlp, redtqlp, wntqlp, \
                            shotk2, shottess, pbest, pbest_tqlp,\
                            k2time, tesstime], dtype='f8')
            #outp_f4     = [round(x, 4) for x in output]
            outp_f4     = ['{:.3g}'.format(x) for x in output[1:-2]]
            tarname     = np.array(output[0],dtype='i8')
            tartime     = ['{:.2f}'.format(x) for x in output[-2:]]
            output      = np.array(np.hstack((tarname,outp_f4,tartime)), dtype='unicode')

            out_array1  = np.vstack((out_array1,output))

            plotfunc([lck2sp,lck2,lctess,lctqlp],[lck2sp_1,lck2_1,lctess_1,lctqlp_1],\
                [lck2sp_rot,lck2_rot,lctess_rot,lctqlp_rot],\
                    [lck2sp_red,lck2_red,lctess_red,lctqlp_red], k2id, tid)
        print(" ")
    
    outfilename = fname.split(".")[0] + "_out.dat"
    np.savetxt(outfilename, out_array1, fmt='%s')
    print("fin.")

