#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import sys

import bin.iso_utils as iu
dir         = "./mcmc/"

def line_sigma(chain):
    schain  = np.sort(chain)
    nc      = len(chain)

    return schain[int(nc*0.175)], schain[int(nc/2)], schain[-int(nc*0.175)]

def limit_chain(chain):
    schain  = np.sort(chain)
    nc      = len(chain)

    return schain[int(nc*0.05):-int(nc*0.05)]

from scipy.stats import gaussian_kde
import matplotlib.cm as cm

def bound(xarray):
    xlen    = len(xarray)
    unm     = int(xlen*0.90)
    lnm     = int(xlen*0.10)
    xsort   = np.sort(xarray)
    #print(xarray)
    #print(xsort)

    return xsort[lnm],xsort[unm],(xsort[unm] - xsort[lnm])/20.

def plot_corner(samples, fkey="tmp", save=True,
    labels = ["S1", "c1", "S2", "c2", "S3", "c3", "S4", "c4"]):
#def plot_corner(sampler, fkey="tmp", save=True, S1flg=True):
    #nshape  = np.shape(sampler.get_chain())
    nshape  = np.shape(samples)
    nstep   = nshape[0]
    nwalker = nshape[1]
    ndim    = nshape[2]

    samples_flt = samples.reshape(nwalker*nstep, ndim)
    #samples_flt = sampler.flatchain
    plen    = 1.4*ndim + 1.
    plt.figure(figsize=(plen, plen))


    #if S1flg:
    #    labels = ["S1", "S2", "c0", "c1", "c2", "c3", "c4"]
    #else:
    #    labels = ["S2", "c0", "c1", "c2", "c3", "c4"]

    for i in range(ndim):
        ch_i        = limit_chain(samples_flt[:,i])
        ys1,ym,ys2  = line_sigma(samples_flt[:,i])
        for j in range(i+1):
            plt.subplot(ndim,ndim,1+i*ndim+j)
            if i==j:
                ch_j        = limit_chain(samples_flt[:,j])
                plt.xlim((ch_j[0], ch_j[-1]))
                xs1,xm,xs2  = line_sigma(samples_flt[:,j])
                xstr    = ['{:.2f}'.format(i) if abs(i) > 1 else str('{:.2e}'.format(i)) for i in (xs1-xm,xm,xs2-xm)]
                plt.hist(ch_j, 50, color='orange')
                plt.axvline(xs1, ls=':', c='black', lw=0.5)
                plt.axvline(xs2, ls=':', c='black', lw=0.5)
                plt.axvline(xm,  ls='--',c='black', lw=0.5)
                plt.tick_params('y', left=False, labelleft=False)
                #plt.title("$"+xstr[1]+"_"+xstr[0]+"$")
                plt.title(xstr[1]+"$_{"+xstr[0]+"}^{+"+xstr[2]+"}$", fontsize=8)
            else:
                ch_j        = limit_chain(samples_flt[:,j])

                plt.xlim((ch_j[0], ch_j[-1]))
                plt.ylim((ch_i[0], ch_i[-1]))
                nbin    = 20
                dx  = (ch_j[-1] - ch_j[0])/nbin
                dy  = (ch_i[-1] - ch_i[0])/nbin
                xx,yy = np.mgrid[ch_j[0]:ch_j[-1]+dx:dx,\
                    ch_i[0]:ch_i[-1]+dy:dy]

                positions = np.vstack([xx.ravel(),yy.ravel()])
                value = np.vstack([ch_j,ch_i])
                kernel = gaussian_kde(value)

                f = np.reshape(kernel(positions).T, xx.shape)
                plt.contourf(xx,yy,f,cmap=cm.Oranges, levels=10)

                xs1,xm,xs2  = line_sigma(samples_flt[:,j])
                plt.axvline(xs1, ls=':', c='black', lw=0.5)
                plt.axvline(xs2, ls=':', c='black', lw=0.5)
                plt.axvline(xm,  ls='--',c='black', lw=0.5)
                plt.axhline(ys1, ls=':', c='black', lw=0.5)
                plt.axhline(ys2, ls=':', c='black', lw=0.5)
                plt.axhline(ym,  ls='--',c='black', lw=0.5)
                if j == 0:
                    plt.ylabel(labels[i])
                    ytl,labs   = plt.yticks()
                    ystr    = ['{:.2f}'.format(i) if abs(i) > 1 else str('{:.2e}'.format(i)) for i in ytl[1:-1]]
                    plt.yticks(ytl[1:-1],ystr, fontsize=8, rotation=45)

                else:
                    plt.tick_params('y', bottom=False, left=False, labelleft=False)
            if i == ndim-1:
                plt.xlabel(labels[j])
                xtl,_   = plt.xticks()
                xstr    = ['{:.2f}'.format(i) if abs(i) > 1 else str('{:.2e}'.format(i)) for i in xtl[1:-1]]
                plt.xticks(xtl[1:-1],xstr, fontsize=8, rotation=45)
            else:
                plt.tick_params('x', bottom=False, labelbottom=False)

    #plt.tight_layout()
    mg  = 0.8
    mgsc    = mg/plen
    plt.subplots_adjust(left=mgsc,hspace=0.0,wspace=0.0,top=1-(mgsc/2), bottom=mgsc)
    if save:
        plt.savefig(dir+fkey+"_corner.png", dpi=300)
        plt.clf()
        plt.close()
    else:
        plt.show()



def plot_chain(samples, fkey="tmp", save=True, S1flg=True,
    labels = ["S1", "c1", "S2", "c2", "S3", "c3", "S4", "c4"]):
    ndim    = np.shape(samples)[-1]
    fig, axes = plt.subplots(ndim, 1, figsize=(6, 1.5*ndim), sharex=True)
    #if S1flg:
    #    labels = ["S1", "S2", "c0", "c1", "c2", "c3", "c4"]
    #else:
    #    labels = ["S2", "c0", "c1", "c2", "c3", "c4"]
    #labels = ["S1", "c1", "S2", "c2", "S3", "c3", "S4", "c4", "S5", "c5"]
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

def plot_scatter2(udata, statres, ndim, MISTdata, fluxmodel, hmodelfunc, fkey="tmp", save=True):

    nbin    = int(ndim[0])
    logg_ar = iu.logg(MISTdata, udata[:,0])
    fig     = plt.figure(figsize=(5.5,4.5))
    ax1     = fig.add_subplot(3,1,(1,2))
    ax2     = fig.add_subplot(3,1,3,sharex=ax1)

    amp_av      = (udata[:,3]*udata[:,4]**(-2) \
        + udata[:,5]*udata[:,6]**(-2))/(udata[:,4]**(-2) + udata[:,6]**(-2))
    tmpar       = np.linspace(np.log10(np.min(amp_av)*0.9), np.log10(np.max(amp_av)*1.1), ndim[1]+1)
    Sthres      = 10**tmpar
    S_ar        = np.full(len(udata[:,0]), 0.01)
    for i in range(int(ndim[1])):
        cond    = ((Sthres[i]<=amp_av)&(amp_av<Sthres[i+1]))
        S_ar[cond]  = statres[i+int(ndim[0]),0]


    Tthres  = np.linspace(np.min(udata[:,0]), np.max(udata[:,0]), nbin+1)
    zansa   = []
    omodel  = []
    for i in range(nbin):
        cond    = ((Tthres[i]<=udata[:,0])&(udata[:,0]<Tthres[i+1]))
        ax1.errorbar(udata[cond,0], udata[cond,1], yerr=udata[cond,2], fmt='o', ms=3.5, elinewidth=0.5, c='black', zorder=2)
        xnew    = np.linspace(Tthres[i],Tthres[i+1],50)
        lnew    = iu.logg(MISTdata, xnew)
        hmodel,_    = hmodelfunc(xnew, statres[i,0], fluxmodel, lnew, S1=0, S2=0.1)
        hmodel_p,_  = hmodelfunc(udata[cond,0], statres[i,0], fluxmodel, logg_ar[cond],\
             S1=0, S2=S_ar[cond])
        ax1.plot(xnew, hmodel, lw=1.5, c='orangered', zorder=3)
        ax2.errorbar(udata[cond,0], udata[cond,1]-hmodel_p,\
            yerr=udata[cond,2], fmt='o', ms=3.5, elinewidth=0.5, c='black', zorder=2)
        ax2.axhline(0,ls=':',lw=0.8,zorder=0,c='black')
        ax1.axhline(1,ls=':',lw=0.8,zorder=0,c='black')
        zansa.append((udata[cond,0], udata[cond,1]-hmodel_p))

        omodel.append([np.mean(udata[cond,0]), np.std(udata[cond,0]), \
            statres[i,0], statres[i,1], statres[i,2]])
                #statres[i*2,0], statres[i*2,1], statres[i*2,2]])
        if i < nbin-1:
            ax1.axvline(Tthres[i+1], ls='--', lw=0.5, zorder=0, c='black')
            ax2.axvline(Tthres[i+1], ls='--', lw=0.5, zorder=0, c='black')
    ax1.axes.xaxis.set_visible(False)
    ax1.set_ylabel("$h_{T}$/$h_{Kp}$")
    ax1.set_ylim((0, 2.))
    ax1.set_yticks([0.2,0.6,1.0,1.4,1.8])
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

    omodel  = ['{:.2f}'.format(i) if abs(i) > 1 else str('{:.2e}'.format(i)) for i in sum(omodel,[])]
    omodel  = np.array(omodel, dtype='unicode')
    
    return omodel.reshape(nbin,5), np.hstack(zansa)


def plot_scatter(udata, statres, MISTdata, fluxmodel, hmodelfunc, dir="./mcmc/", fkey="tmp", save=True):

    nbin    = int(np.shape(statres)[0]/2)
    logg_ar = iu.logg(MISTdata, udata[:,0])
    plt.rcParams["font.family"]     = "Arial"
    plt.rcParams["font.size"]     = 13
    fig     = plt.figure(figsize=(5.5,4.5))
    ax1     = fig.add_subplot(3,1,(1,2))
    ax2     = fig.add_subplot(3,1,3,sharex=ax1)

    Tthres  = np.linspace(np.min(udata[:,0]), np.max(udata[:,0]), nbin+1)
    zansa   = []
    omodel  = []
    for i in range(nbin):
        cond    = ((Tthres[i]<=udata[:,0])&(udata[:,0]<Tthres[i+1]))
        ax1.errorbar(udata[cond,0], udata[cond,1], yerr=udata[cond,2], fmt='o', ms=2.8,
            elinewidth=0.5, c='black', zorder=2, alpha=0.5)
        xnew    = np.linspace(Tthres[i],Tthres[i+1],50)
        lnew    = iu.logg(MISTdata, xnew)
        hmodel,_    = hmodelfunc(xnew, statres[i*2+1,0], fluxmodel, lnew, S1=0, S2=statres[i*2,0])
        hmodel_p,_  = hmodelfunc(udata[cond,0], statres[i*2+1,0], fluxmodel, logg_ar[cond],\
             S1=0, S2=statres[i*2,0])
        ax1.plot(xnew, hmodel, lw=2, c='red', zorder=3)
        stud    = np.sort(udata[cond,1])
        nsd     = len(stud)
        er1     = np.abs(stud[int(nsd*0.159)] - stud[int(nsd*0.5)])
        er2     = np.abs(stud[int(nsd*0.841)] - stud[int(nsd*0.5)])
        er      = np.array((er1,er2))
        #ax1.scatter(np.mean(udata[cond,0]), np.average(udata[cond,1], weights=udata[cond,2]))
        #ax1.scatter(np.median(udata[cond,0]), np.median(udata[cond,1]), s=50, zorder=3, c='deepskyblue', marker='o',
        #    ec='black', linewidth=0.5)
        ax1.errorbar(np.median(udata[cond,0]), np.median(udata[cond,1]), yerr=er.reshape(-1,1),
                     ms=10, mfc='skyblue', fmt='o', mec='black', ecolor='skyblue', elinewidth=2)
        #, zorder=3, c='deepskyblue', marker='o',
            #ec='black', linewidth=0.5)
        ax2.errorbar(udata[cond,0], udata[cond,1]-hmodel_p,\
            yerr=udata[cond,2], fmt='o', ms=2.8, elinewidth=0.5, c='black', zorder=2, alpha=0.5)
        ax2.axhline(0,ls=':',lw=0.8,zorder=0,c='black')
        ax1.axhline(1,ls=':',lw=0.8,zorder=0,c='black')
        zansa.append((udata[cond,0], udata[cond,1]-hmodel_p))

        omodel.append([np.mean(udata[cond,0]), np.std(udata[cond,0]), \
            statres[i*2+1,0], statres[i*2+1,1], statres[i*2+1,2],\
                statres[i*2,0], statres[i*2,1], statres[i*2,2]])
        if i < nbin-1:
            ax1.axvline(Tthres[i+1], ls='-', lw=0.8, zorder=0, c='black')
            ax2.axvline(Tthres[i+1], ls='-', lw=0.8, zorder=0, c='black')
    ax1.axes.xaxis.set_visible(False)
    ax1.set_ylabel("$h_{T}$/$h_{Kp}$")
    ax1.set_ylim((0, 2.))
    ax1.set_yticks([0.2,0.6,1.0,1.4,1.8])
    ax2.set_ylim((-1.8,1.8))
    #ax2.set_ylim((-10,10))
    #ax2.set_yscale("log")
    ax2.set_ylabel("residuals")
    ax2.set_xlabel("effective temperature [K]")
    plt.subplots_adjust(left=0.15,hspace=0.,top=0.9, bottom=0.15)

    if save :
        plt.savefig(dir+fkey+"_scatter.png", dpi=300)
        plt.clf()
        plt.close()
    else:
        plt.show()

    omodel  = ['{:.2f}'.format(i) if abs(i) > 1 else str('{:.2e}'.format(i)) for i in sum(omodel,[])]
    omodel  = np.array(omodel, dtype='unicode')

    return omodel.reshape(nbin,8), np.hstack(zansa)
    
def plot_scatter_H(udata, statres, MISTdata, fluxmodel, hmodelfunc, s_flat,
                   dir="./mcmc/", fkey="tmp", save=True):

    nbin    = int(np.shape(statres)[0]/2)
    logg_ar = iu.logg(MISTdata, udata[:,0])
    plt.rcParams["font.family"]     = "Arial"
    plt.rcParams["font.size"]     = 13
    fig     = plt.figure(figsize=(5.5,4.5))
    ax1     = fig.add_subplot(3,1,(1,2))
    ax2     = fig.add_subplot(3,1,3,sharex=ax1)

    Tthres  = np.linspace(np.min(udata[:,0]), np.max(udata[:,0]), nbin+1)
    zansa   = []
    omodel  = []
    for i in range(nbin):
        cond    = ((Tthres[i]<=udata[:,0])&(udata[:,0]<Tthres[i+1]))
        #ax1.errorbar(udata[cond,0], udata[cond,1], yerr=udata[cond,2], fmt='o', ms=2.8,
        #    elinewidth=0.5, c='black', zorder=2, alpha=0.5)
        xnew    = np.linspace(Tthres[i],Tthres[i+1],50)
        lnew    = iu.logg(MISTdata, xnew)
        hmodel_p,resh_p = hmodelfunc(udata[cond,0], statres[i*2+1,0], fluxmodel, logg_ar[cond],\
             S1=0, S2=statres[i*2,0])
        if i==0:
            ax1.errorbar(udata[cond,0], udata[cond,3], yerr=udata[cond,4], 
                        fmt='o', ms=2.8, elinewidth=0.5, c='blue', alpha=0.5, label='$Kp$')
            ax1.errorbar(udata[cond,0], udata[cond,5], yerr=udata[cond,6], 
                        fmt='o', ms=2.8, elinewidth=0.5, c='orangered', alpha=0.5, label='$T$')
        else:
            ax1.errorbar(udata[cond,0], udata[cond,3], yerr=udata[cond,4], 
                        fmt='o', ms=2.8, elinewidth=0.5, c='blue', alpha=0.5)
            ax1.errorbar(udata[cond,0], udata[cond,5], yerr=udata[cond,6], 
                        fmt='o', ms=2.8, elinewidth=0.5, c='orangered', alpha=0.5)
        for k in s_flat:
            x0  = k[i*2+1]
            x1  = k[i*2]
            _,resh   = hmodelfunc(xnew, x0, fluxmodel, lnew, S1=0, S2=x1)
            if len(resh)!=3:
                ax1.plot(xnew, resh[0], c='royalblue', alpha=0.08, lw=1)
                ax1.plot(xnew, resh[1], c='orangered', alpha=0.08, lw=1)
        
        _,resh   = hmodelfunc(xnew, statres[i*2+1,0], fluxmodel, lnew, S1=0, S2=statres[i*2,0])
        ax1.plot(xnew, resh[0], c='mediumblue', lw=1.5, zorder=5)
        ax1.plot(xnew, resh[1], c='red', lw=1.5, zorder=5)

        ax2.errorbar(udata[cond,0], udata[cond,3]-resh_p[0],\
            yerr=udata[cond,4], fmt='o', ms=2.8, elinewidth=0.5, c='royalblue', zorder=2, alpha=0.5)
        ax2.errorbar(udata[cond,0], udata[cond,5]-resh_p[1],\
            yerr=udata[cond,6], fmt='o', ms=2.8, elinewidth=0.5, c='orangered', zorder=2, alpha=0.5)
        ax2.axhline(0,ls=':',lw=0.8,zorder=0,c='black')
        ax1.axhline(1,ls=':',lw=0.8,zorder=0,c='black')
        zansa.append((udata[cond,0], udata[cond,1]-hmodel_p))

        omodel.append([np.mean(udata[cond,0]), np.std(udata[cond,0]), \
            statres[i*2+1,0], statres[i*2+1,1], statres[i*2+1,2],\
                statres[i*2,0], statres[i*2,1], statres[i*2,2]])
        if i < nbin-1:
            ax1.axvline(Tthres[i+1], ls='-', lw=0.8, zorder=0, c='black')
            ax2.axvline(Tthres[i+1], ls='-', lw=0.8, zorder=0, c='black')
    ax1.set_ylim((1e-3,3.5e-2))
    ax1.axes.xaxis.set_visible(False)
    ax1.set_ylabel("$h$")
    ax2.set_ylim((-0.02,0.02))
    ax2.set_ylabel("residuals")
    ax2.set_xlabel("effective temperature [K]")
    ax1.legend(loc='upper right', fontsize=10)
    plt.subplots_adjust(left=0.15,hspace=0.,top=0.9, bottom=0.15)
    #plt.show()
    plt.savefig(dir+fkey+"_scatter_inH.png", dpi=300)