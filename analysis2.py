#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import sys

from scipy.interpolate import interp1d
import bin.model_utils as mu
import bin.iso_utils as iu
import bin.mcmcplot as mp

import emcee
from copy import copy

dir         = "./mcmc/"


#def d_temp(Tph, c):
#    dT  = np.full(np.shape(Tph), 0.)
#    for i in range(len(c)):
#        dT  += c[i]*(Tph - 3000.)**i
#    return dT
#    
#def Tsp(Tph, c):
#    Tspot   = Tph - d_temp(Tph, c)
#
#    return Tspot

def Tsp(Tph, c):
    Tspot   = Tph*(1.- c)
    return Tspot

def Tph_SBlaw(Thost, c, S1, S2, fluxmodel, logg):
    Fhost   = mu.makefluxmodel_Tefflogg(fluxmodel, Thost, logg)
    Fspot   = mu.makefluxmodel_Tefflogg(fluxmodel, Tsp(Thost, c), logg)
    if (np.all(Fspot!=1) &(np.all(Fspot<Fhost))):
        Fphot       = (2.*Fhost - Fspot*(2.*S1 + S2))/(2.-2.*S1- S2)
        Tphot       = Thost.reshape(-1,1) * (Fphot/Fhost)**0.25
        return np.mean(Tphot, axis=1)
    else:
        return np.array((0,0,0))


def h_model(Thost, c, S1, S2, fluxmodel, logg):
    Fhost   = mu.makefluxmodel_Tefflogg(fluxmodel, Thost, logg)
    Fspot   = mu.makefluxmodel_Tefflogg(fluxmodel, Tsp(Thost, c), logg)
    if (np.all(Fspot!=1) &(np.all(Fspot<Fhost))):
        Fphot       = (2.*Fhost - Fspot*(2.*S1 + S2))/(2.-2.*S1- S2)
        h   = (Fphot - Fspot)*S2/Fhost/2.
        hKp = h[:,0]
        hT  = h[:,1]

        return hKp, hT
    else:
        return np.array((0,0,0))


def hThKp_model(Thost, c, fluxmodel, logg, S1=0.1, S2=0.1):
    resh    = h_model(Thost, c, S1, S2, fluxmodel, logg)
    if np.any(resh!=0):
        hThKp   = resh[1]/ resh[0]
        return hThKp, resh
    else:
        return np.array((0,0,0)), resh

def lnfn_unit(x, args):
    teff        = args[0]
    #hThKp       = args[1]
    #err         = args[2]
    fluxmodel   = args[3]
    logg        = args[4]
    
    hKp         = args[5]
    hKp_e       = args[6]
    hT          = args[7]
    hT_e        = args[8]

    S1flg       = args[9]
    #if S1flg :
    #    if (np.min(x[:1])<0.) | (1<=np.max(x[:1])) | (1.<np.sum(x[:1])):
    #        return -np.inf
    #    else:
    #        hmodel,resh = hThKp_model(teff, x[2:], fluxmodel, logg, S1=x[0], S2=x[1])
    #else :
    if (x[0]<0.) | (1<=x[0]) :
        return -np.inf
    else:
        hmodel,resh = hThKp_model(teff, x[1], fluxmodel, logg, S1=0, S2=x[0])

    if np.all(hmodel==0.0):
        return -np.inf
    else:
        hKp_m   = resh[0]
        hT_m    = resh[1]
        lnL     = -(np.sum(((hKp - hKp_m)**2)/ (2*hKp_e**2)))\
                    -(np.sum(((hT - hT_m)**2)/ (2*hT_e**2)))

        return lnL

def lnfn(x, *args):
    teff        = args[0]
    #hThKp       = args[1]
    #err         = args[2]
    fluxmodel   = args[3]
    logg        = args[4]

    hKp         = args[5]
    hKp_e       = args[6]
    hT          = args[7]
    hT_e        = args[8]

    #S1flg       = args[9]
    S1flg       = False
    nbin        = args[10]
    Tthres      = np.linspace(np.min(teff), np.max(teff), nbin+1)
    #nx          = 3 if S1flg else 2
    nx          = int(len(x)/nbin)

    lnfn_total  = 0
    for i in range(nbin):
        cond    = ((Tthres[i]<=teff)&(teff<Tthres[i+1]))
        args_u  = [teff[cond], 0, 0, fluxmodel, logg[cond],\
            hKp[cond], hKp_e[cond], hT[cond], hT_e[cond], S1flg]
        x_u     = x[i*nx:(i+1)*nx]
        lnfn_total  += lnfn_unit(x_u, args_u)

    return lnfn_total

def make_init(x0, ndim, nwalker):
    res     = []
    for i in range(ndim):
        x   = x0[int(i%2)]
        tmp = np.random.uniform(x[0], x[1], nwalker)
        res.append(tmp)

    return np.vstack(res).T

from multiprocessing import Pool
def emcee_fit(param, x0, ndim=2, nwalker=10, nstep=1000, dfrac=0.3, plot=False, fkey="tmp"):

    S1flg   = param[-2]
    xy0 = make_init(x0, ndim, nwalker)
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalker,ndim,lnfn,args=param, pool=pool)
        sampler.run_mcmc(xy0, nstep, progress=True)
    vid     = (sampler.get_log_prob()[-1]!=-np.inf)
    samples = sampler.get_chain()[int(nstep*dfrac):, vid]

    if plot:
        mp.plot_chain(samples, fkey=fkey, S1flg=S1flg)
        mp.plot_corner(samples,fkey=fkey, S1flg=S1flg)
        #mp.plot_corner(sampler,fkey=fkey, S1flg=S1flg)

    #return np.median(samples[:,:,0]), np.median(samples[:,:,1])
    return sampler

def rm_outlier(idata):
    odata       = copy(idata)
    sid         = np.argsort(odata[:,1])
    odata       = odata[sid]
    return odata[3:-3]

def calstat(flatchain):
    dlen    = len(flatchain[0])
    res     = []
    for fc in flatchain:
        sc  = np.sort(fc)
        loe = sc[int(dlen*0.176)]
        upe = sc[-int(dlen*0.176)]
        med = sc[int(dlen/2)]
        
        res.append([med, loe-med, upe-med])
    
    return np.vstack(res)

def form_result(statres, BIC, redchi, S1flg=True):
    res = {}
    if S1flg:
        name_ar = ["S1", "S2","c0", "c1", "c2", "c3", "c4"]
    else:
        name_ar = ["S2","c0", "c1", "c2", "c3", "c4"]

    for i in range(len(statres[:,0])):
        name1  = name_ar[i]
        name2  = name_ar[i] + "_er1"
        name3  = name_ar[i] + "_er2"
        name4  = name_ar[i] + "_str"
        res[name1]  = '{:.2f}'.format(statres[i,0]) if abs(statres[i,0]) > 1 else '{:.2e}'.format(statres[i,0])
        res[name2]  = '{:.2f}'.format(statres[i,1]) if abs(statres[i,1]) > 1 else '{:.2e}'.format(statres[i,1])
        res[name3]  = '{:.2f}'.format(statres[i,2]) if abs(statres[i,2]) > 1 else '{:.2e}'.format(statres[i,2])
        res[name4]  = res[name1]+"_{"+res[name2]+"}^{"+res[name3]+"}"
    res["bic"]      = '{:.2f}'.format(BIC)
    res["redchi"]   = '{:.2f}'.format(redchi)

    return res

import json
import subprocess
#def main(vdata, MISTdata, fluxmodel, fkey="tmp", ndim=2, nwalker=10, nstep=1000, S1flg=False, nbin=3):
def main(vdata, MISTdata, fluxmodel, fkey="tmp", nwalker=10, nstep=1000, S1flg=False, nbin=3):
    udata       = rm_outlier(vdata)
    logg_ar     = iu.logg(MISTdata, udata[:,0])
    ndim        = nbin*2

    #print(np.linspace(2500, 5500, nbin+2))
    #print(np.linspace(np.min(udata[:,0]), np.max(udata[:,0]), nbin+2))
    #exit()
    if S1flg:
        stail   = "_s2"
    else:
        stail   = "_s1"

    args    = [udata[:,0], udata[:,1], udata[:,2], fluxmodel, logg_ar, \
                udata[:,3], udata[:,4],\
                    udata[:,5], udata[:,6], S1flg, nbin]
    
    fparam  = (dir + fkey).split("_w")[0]
    spres   = subprocess.run("ls -tr "+fparam+"*"+stail+"_result.json | tail -n 1",\
        shell=True, capture_output=True,text=True).stdout
    if len(spres) != 0:
        fparam  = spres[:-1]
        jo  = open(fparam, 'r')
        prm = json.load(jo)
        prmv    = list(prm.values())
        x0      = []
        for i in range(ndim):
            md  = float(prmv[i*4])
            e1  = float(prmv[i*4+1])
            e2  = float(prmv[i*4+2])
            x0.append([md+e1, md+e2])
    else:
        x0      = [ [0.001, 0.1],\
                    [0., 0.1]]
    #print(lnfn([0.001, 0.4, 0.001, 0.2, 0.2, 0.4], *args))
    #exit()
    #            

    sampler = emcee_fit(args, x0, ndim=ndim, nwalker=nwalker, nstep=nstep, plot=True, fkey=fkey)
    s_flat  = sampler.get_chain(flat=True)

    vid         = (sampler.get_log_prob()[-1]!=-np.inf)
    samples     = sampler.get_chain()[int(nstep/3.):, vid]
    ns,nw,nd    = np.shape(samples)
    s_flat      = samples.reshape(ns*nw, nd)

    statres     = calstat(s_flat.T)
    bestlnfn    = max(sampler.get_log_prob(flat=True))
    BIC         = -2*bestlnfn + ndim*np.log(len(udata[:,0])*2)
    redchi      = -2*bestlnfn/(len(udata[:,0])*2 - ndim)
    
    #hmodel      = mp.plot_scatter(vdata, udata, statres[:,0],\
    #    MISTdata, fluxmodel, hThKp_model, fkey=fkey, S1flg=S1flg)
    #np.savetxt(dir+fkey+"_scatter_model.dat", np.vstack((udata.T,hmodel)).T)

    result_dict = form_result(statres, BIC, redchi, S1flg)
    tf          = open(dir+fkey+"_result.json","w")
    json.dump(result_dict, tf)
    tf.close()
    
    #newx    = np.linspace(min(udata[:,0]), max(udata[:,0]), 100)
    #logg    = iu.logg(MISTdata, newx)
    #if S1flg:
    #    Tphot   = Tph_SBlaw(newx, statres[2:,0],statres[0,0],statres[0,1], fluxmodel, logg)
    #    Tplot   = np.array((Tphot, Tphot - Tsp(newx, statres[2:,0])))
    #else:
    #    Tphot   = Tph_SBlaw(newx, statres[1:,0],0,statres[0,1], fluxmodel, logg)
    #    Tplot   = np.array((Tphot, Tphot - Tsp(newx, statres[1:,0])))

    #np.savetxt(dir+fkey+"_Tplot.dat", Tplot.T)

def clean_vdata(vdata):
    con1    = (2600.<= vdata[:,0])&(vdata[:,0]<=6000.)
    con2    = np.logical_not(np.isnan(vdata[:,2]))
    return vdata[(con1&con2)]


if __name__=='__main__':

    if (len(sys.argv) == 4) | (len(sys.argv) == 7) | (len(sys.argv)==8):

        vfname      = sys.argv[1]
        cmdfname    = sys.argv[2]
        FeH         = float(sys.argv[3])
        flxfname    = "./bin/data/fluxdata/flux.dat"   

        vdata       = np.loadtxt(vfname, dtype='f8', comments='#')
        vdata       = clean_vdata(vdata)
        MISTdata    = np.loadtxt(cmdfname, dtype='f8', comments='#')
        fluxmodel0  = np.loadtxt(flxfname, dtype='f8', comments='#')
        fluxmodel   = mu.makefluxmodel_grid(fluxmodel0[(fluxmodel0[:,1]==0.0)])
        #exit()
        #fluxmodel   = mu.makefluxmodel_FeH(fluxmodel0, FeH, FeHb=0.3)

        if len(sys.argv) >= 7:
            nbin    = int(sys.argv[4])
            nwalker = int(sys.argv[5])
            nstep   = int(sys.argv[6])
            #if len(sys.argv) == 8:
            #    if sys.argv[7] == "False":
            #        S1flg   = False
            #    elif sys.argv[7] == "True":
            #        S1flg   = True
            #    else:
            #        print("7th arg is True or False.")
            #        exit()
            #else:
            #    S1flg   = True
        else:
            nbin    = 3
            nwalker = 10
            nstep   = 1000
            S1flg   = True
        #if S1flg:
        #    stail   = "_s2"
        #else:
        #    stail   = "_s1"

        fkey    = vfname.split("/")[-1]
        fkey    = fkey.split("_")[0] + "_d"+str(int(nbin*2))+"_w"+str(nwalker)+"_s"+str(nstep)
        #main(vdata, MISTdata, fluxmodel, fkey=fkey, ndim=ndim, nwalker=nwalker, nstep=nstep, S1flg=False)
        main(vdata, MISTdata, fluxmodel, fkey=fkey, nwalker=nwalker, nstep=nstep, S1flg=False, nbin=nbin)
    else:
        print("args) [hT/hKp file] [MISTcmd file] [FeH]")