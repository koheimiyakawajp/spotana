#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import sys

from scipy.interpolate import interp1d
import bin.model_utils as mu
import bin.iso_utils as iu

import emcee
from copy import copy

dir         = "./mcmc/"

def d_temp(Tph, c):
    dT  = np.full(np.shape(Tph), 0.)
    for i in range(len(c)):
        dT  += c[i]*(Tph - 3000.)**i
    return dT
    
def Tsp(Tph, c):
    Tspot   = Tph - d_temp(Tph, c)

    return Tspot

def h_model(Thost, c, S, fluxmodel, logg):
    Fhost   = mu.makefluxmodel_Tefflogg(fluxmodel, Thost, logg)
    Fspot   = mu.makefluxmodel_Tefflogg(fluxmodel, Tsp(Thost, c), logg)
    if (np.all(Fspot!=1) &(np.all(Fspot<Fhost))):
        Fhost_Kp    = Fhost[:,0]
        Fhost_T     = Fhost[:,1]
        Fspot_Kp    = Fspot[:,0]
        Fspot_T     = Fspot[:,1]

        hKp = (1 - Fspot_Kp/Fhost_Kp)*S
        hT  = (1 - Fspot_T/Fhost_T)*S
        return hKp, hT
    else:
        return np.array((0,0,0))


def hThKp_model(Thost, c, fluxmodel, logg, S=1):
    resh    = h_model(Thost, c, S, fluxmodel, logg)
    if np.any(resh!=0):
        hThKp   = resh[1]/ resh[0]
        return hThKp, resh
    else:
        return np.array((0,0,0)), resh

def lnfn(x, *args):
    teff        = args[0]
    hThKp       = args[1]
    err         = args[2]
    fluxmodel   = args[3]
    logg        = args[4]
    
    hKp         = args[5]
    hKp_e       = args[6]
    hT          = args[7]
    hT_e        = args[8]

    hmodel,resh = hThKp_model(teff, x[1:], fluxmodel, logg, S=x[0])
    if np.all(hmodel==0.0):
        return -np.inf
    else:
        #lnL     = -(np.sum(((hThKp - hmodel)**2)/ (2*err**2)))
        hKp_m   = resh[0]
        hT_m    = resh[1]
        #SKp     = 2*hKp/hKp_m
        #ST      = 2*hT/hT_m
        #if (x[0]<0.001) | (1<=x[0]):
        #    return -np.inf
        #else:
        lnL     = -(np.sum(((hKp - hKp_m)**2)/ (2*hKp_e**2)))\
                    -(np.sum(((hT - hT_m)**2)/ (2*hT_e**2)))
                    #    +np.log(1./(x[0]*1e3*np.log(1000./1.)))

        return lnL

def line_sigma(chain):
    schain  = np.sort(chain)
    nc      = len(chain)

    return schain[int(nc*0.175)], schain[int(nc/2)], schain[-int(nc*0.175)]

def limit_chain(chain):
    schain  = np.sort(chain)
    nc      = len(chain)

    return schain[int(nc*0.05):-int(nc*0.05)]

def plot_corner(samples, fkey="tmp", save=True):
    nshape  = np.shape(samples)
    nstep   = nshape[0]
    nwalker = nshape[1]
    ndim    = nshape[2]

    samples_flt = samples.reshape(nwalker*nstep, ndim)
    plt.figure(figsize=(2.1*ndim, 2.*ndim))

    labels = ["S", "c0", "c1", "c2", "c3", "c4"]
    for i in range(ndim):
        ch_i        = limit_chain(samples_flt[:,i])
        ys1,ym,ys2  = line_sigma(samples_flt[:,i])
        for j in range(i+1):
            plt.subplot(ndim,ndim,1+i*ndim+j)
            if i==j:
                ch_j        = limit_chain(samples_flt[:,j])
                plt.xlim((ch_j[0], ch_j[-1]))
                xs1,xm,xs2  = line_sigma(samples_flt[:,j])
                plt.hist(ch_j, 50, color='orange')
                plt.axvline(xs1, ls=':', c='black', lw=0.5)
                plt.axvline(xs2, ls=':', c='black', lw=0.5)
                plt.axvline(xm,  ls='--',c='black', lw=0.5)
                plt.tick_params('y', labelleft=False)
            else:
                ch_j        = limit_chain(samples_flt[:,j])
                plt.xlim((ch_j[0], ch_j[-1]))
                plt.ylim((ch_i[0], ch_i[-1]))
                xs1,xm,xs2  = line_sigma(samples_flt[:,j])
                plt.hist2d(ch_j, ch_i, bins=50, cmap="Oranges")
                plt.axvline(xs1, ls=':', c='black', lw=0.5)
                plt.axvline(xs2, ls=':', c='black', lw=0.5)
                plt.axvline(xm,  ls='--',c='black', lw=0.5)
                plt.axhline(ys1, ls=':', c='black', lw=0.5)
                plt.axhline(ys2, ls=':', c='black', lw=0.5)
                plt.axhline(ym,  ls='--',c='black', lw=0.5)
                if j == 0:
                    plt.ylabel(labels[i])
                else:
                    plt.tick_params('y', labelleft=False)
            if i == ndim-1:
                plt.xlabel(labels[j])
            else:
                plt.tick_params('x', labelbottom=False)

    plt.tight_layout()
    if save:
        plt.savefig(dir+fkey+"_corner.png", dpi=300)
        plt.clf()
        plt.close()
    else:
        plt.show()



def plot_chain(samples, fkey="tmp", save=True):
    ndim    = np.shape(samples)[-1]
    fig, axes = plt.subplots(ndim, 1, figsize=(6, 1.5*ndim), sharex=True)
    labels = ["c0", "c1", "c2", "c3", "c4"]
    for i in range(ndim):
        ax1 = axes[i]
        ax1.plot(samples[:, :, i], "k", alpha=0.3, lw=1)
        ax1.plot(np.median(samples[:, :, i], axis=1), c="darkorange")
        ax1.set_xlim(0, len(samples))
        ax1.set_ylabel(labels[i])
    axes[-1].set_xlabel("n steps")
    #if len(samples[:,:,0]) > 1e3:
    #    plt.xscale("log")
    plt.tight_layout()
    if save:
        plt.savefig(dir+fkey+"_chain.png", dpi=300)
        plt.clf()
        plt.close()
    else:
        plt.show()

def plot_scatter(vdata, udata, args, MISTdata, fluxmodel, fkey="tmp", save=True):
    
    newx    = np.linspace(min(vdata[:,0]), max(udata[:,0]), 300)
    newlogg = iu.logg(MISTdata, newx)

    fig     = plt.figure(figsize=(5.5,4.5))

    ax1     = fig.add_subplot(3,1,(1,2))
    hThKp,_ = hThKp_model(newx, args[1:], fluxmodel, newlogg, S=args[0])
    ax1.plot(newx, hThKp, c='darkorange', zorder=3)
    ax1.scatter(vdata[:,0], vdata[:,1], s=10, c='lightgrey', zorder=1)
    ax1.errorbar(udata[:,0], udata[:,1], yerr=udata[:,2], fmt='o', ms=3.5, elinewidth=0.5, c='black', zorder=2)
    ax1.axhline(1., lw=1, c='black', ls=':', zorder=0)
    ax1.set_yscale("log")
    ax1.axes.xaxis.set_visible(False)
    ax1.set_ylabel("$h_{T}$/$h_{Kp}$")
    ax1.set_ylim((5e-2, 2e1))

    hThKp_s,_   = hThKp_model(udata[:,0], args[1:], fluxmodel, newlogg, S=args[0])
    ax2     = fig.add_subplot(3,1,3, sharex=ax1)
    ax2.errorbar(udata[:,0], udata[:,1] - hThKp_s, yerr=udata[:,2], ms=3.5, elinewidth=0.5, fmt='o', c='black', zorder=0)
    ax2.axhline(0., lw=1, c='black', ls=':')
    ax2.set_ylim((-1.8,1.8))
    ax2.set_ylabel("residuals")
    ax2.set_xlabel("effective temperature [K]")
    plt.subplots_adjust(left=0.15,hspace=0.,top=0.9, bottom=0.15)

    if save :
        plt.savefig(dir+fkey+"_scatter.png", dpi=300)
        plt.clf()
        plt.close()
    else:
        plt.show()

def make_init(x0, ndim, nwalker):
    res     = []
    for i in range(ndim):
        x   = x0[i]
        tmp = np.random.uniform(x[0], x[1], nwalker)
        res.append(tmp)

    return np.vstack(res).T

from multiprocessing import Pool
def emcee_fit(param, x0, ndim=2, nwalker=10, nstep=1000, dfrac=0.3, plot=False, fkey="tmp"):

    xy0 = make_init(x0, ndim, nwalker)
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalker,ndim,lnfn,args=param, pool=pool)
        sampler.run_mcmc(xy0, nstep, progress=True)
    vid     = (sampler.get_log_prob()[-1]!=-np.inf)
    samples = sampler.get_chain()[int(nstep*dfrac):, vid]

    if plot:
        plot_chain(samples, fkey=fkey)
        plot_corner(samples,fkey=fkey)

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

def form_result(statres, BIC, redchi):
    res = {}
    for i in range(len(statres[:,0])):
        name1  = "c" + str(i)
        name2  = "c" + str(i) + "_er1"
        name3  = "c" + str(i) + "_er2"
        name4  = "c" + str(i) + "_str"
        res[name1]  = '{:.2f}'.format(statres[i,0]) if abs(statres[i,0]) > 1 else '{:.2e}'.format(statres[i,0])
        res[name2]  = '{:.2f}'.format(statres[i,1]) if abs(statres[i,1]) > 1 else '{:.2e}'.format(statres[i,1])
        res[name3]  = '{:.2f}'.format(statres[i,2]) if abs(statres[i,2]) > 1 else '{:.2e}'.format(statres[i,2])
        res[name4]  = res[name1]+"_{"+res[name2]+"}^{"+res[name3]+"}"
    res["bic"]      = '{:.2f}'.format(BIC)
    res["redchi"]   = '{:.2f}'.format(redchi)

    return res

import json
def main(vdata, MISTdata, fluxmodel, fkey="tmp", ndim=2, nwalker=10, nstep=1000):
    udata       = rm_outlier(vdata)
    logg_ar     = iu.logg(MISTdata, udata[:,0])
    #x   = np.linspace(1.,1000,int(1e4))
    #x   = np.linspace(1.e-3,1.,int(1e4))
    ##y   = 1./(x*np.log(1000./1.))
    ##y   = 1./(x*np.log(x[-1]/x[0]))
    #y   = np.full(int(1e4),1./(x[-1] - x[0]))
    #print(np.sum(y))
    #plt.plot(x,y)
    #plt.xscale('log')
    #plt.show()
    #exit()

    #plt.scatter(udata[:,3], udata[:,5])
    #plt.show()
    #exit()
    #c_test      = [27.46, -4.30e-02, 3.25e-05]
    #print(udata[:,1])
    #_, resh     = hThKp_model(udata[:,0], c_test, fluxmodel, logg_ar, S=1)
    #print(resh)
    #Sspot   = 2* udata[:,3]/resh[0]
    #SspotT  = 2* udata[:,5]/resh[1]
    #plt.hist(Sspot  , 50, range=(0,2))
    #plt.hist(SspotT , 50, range=(0,2), alpha=0.5)
    #plt.show()
    #exit()

    args    = [udata[:,0], udata[:,1], udata[:,2], fluxmodel, logg_ar, udata[:,3], udata[:,4],\
        udata[:,5], udata[:,6]]
    x0      = [ [0.01, 0.1],\
                [0., 3000.],\
                [-1., 1.],\
                [-1.e-3, 1.e-3],\
                [-1e-8, 1e-8] ]

    sampler = emcee_fit(args, x0, ndim=ndim, nwalker=nwalker, nstep=nstep, plot=True, fkey=fkey)
    s_flat  = sampler.get_chain(flat=True)

    vid         = (sampler.get_log_prob()[-1]!=-np.inf)
    samples     = sampler.get_chain()[int(nstep/3.):, vid]
    ns,nw,nd    = np.shape(samples)
    s_flat      = samples.reshape(ns*nw, nd)

    statres     = calstat(s_flat.T)
    bestlnfn    = max(sampler.get_log_prob(flat=True))
    BIC         = -2*bestlnfn + ndim*np.log10(len(udata[:,0]))
    redchi      = -2*bestlnfn/len(udata[:,0])
    
    
    plot_scatter(vdata, udata, statres[:,0], MISTdata, fluxmodel, fkey=fkey)

    result_dict = form_result(statres, BIC, redchi)
    tf          = open(dir+fkey+"_result.json","w")
    json.dump(result_dict, tf)
    tf.close()
    #exit()
    #hmodel,resh = hThKp_model(udata[:,0], statres[:,0], fluxmodel, logg_ar)
    #SKp         = 2.*udata[:,3]/resh[0]
    #ST          = 2.*udata[:,5]/resh[1]

    #plt.figure(figsize=(5,5))
    #plt.scatter(udata[:,-1], ST/SKp, c=udata[:,-1])
    #plt.scatter(udata[:,0], ST/SKp, c=udata[:,-1])
    #plt.scatter(SKp, ST, c=udata[:,-1])
    #plt.plot(np.linspace(0,1,100), np.linspace(0,1,100), ls='--', c='black', lw=0.5)
    #plt.xlim((2e-3,2e-1))
    #plt.ylim((2e-3,2e-1))
    #plt.xscale("log") 
    #plt.yscale("log") 
    #plt.show()
    

    newx    = np.linspace(min(vdata[:,0]), max(vdata[:,0]), 300)
    plt.figure(figsize=(5,5))
    plt.plot(newx, newx, ls='--', lw=0.5, c='black', zorder=0)
    plt.plot(newx, Tsp(newx, statres[1:,0]), zorder=2)
    plt.show()
    exit()

def clean_vdata(vdata):
    con1    = (2600.<= vdata[:,0])&(vdata[:,0]<=6000.)
    con2    = np.logical_not(np.isnan(vdata[:,2]))
    return vdata[(con1&con2)]


if __name__=='__main__':

    if (len(sys.argv) == 4) | (len(sys.argv) == 7):

        vfname      = sys.argv[1]
        cmdfname    = sys.argv[2]
        FeH         = float(sys.argv[3])
        flxfname    = "./bin/data/fluxdata/flux.dat"   

        vdata       = np.loadtxt(vfname, dtype='f8', comments='#')
        vdata       = clean_vdata(vdata)
        MISTdata    = np.loadtxt(cmdfname, dtype='f8', comments='#')
        fluxmodel0  = np.loadtxt(flxfname, dtype='f8', comments='#')
        fluxmodel   = mu.makefluxmodel_FeH(fluxmodel0, FeH, FeHb=0.3)
        if len(sys.argv) == 7:
            ndim    = int(sys.argv[4])
            nwalker = int(sys.argv[5])
            nstep   = int(sys.argv[6])
        else:
            ndim    = 3
            nwalker = 10
            nstep   = 1000

        fkey    = vfname.split("/")[-1]
        fkey    = fkey.split("_")[0] + "_d"+str(ndim)+"_w"+str(nwalker)+"_s"+str(nstep)
        main(vdata, MISTdata, fluxmodel, fkey=fkey, ndim=ndim, nwalker=nwalker, nstep=nstep)
    else:
        print("args) [hT/hKp file] [MISTcmd file] [FeH]")