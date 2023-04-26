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

def plot_corner(samples, fkey="tmp", save=True, S1flg=True):
#def plot_corner(sampler, fkey="tmp", save=True, S1flg=True):
    #nshape  = np.shape(sampler.get_chain())
    nshape  = np.shape(samples)
    nstep   = nshape[0]
    nwalker = nshape[1]
    ndim    = nshape[2]

    samples_flt = samples.reshape(nwalker*nstep, ndim)
    #samples_flt = sampler.flatchain
    plt.figure(figsize=(2.1*ndim, 2.*ndim))

    #if S1flg:
    #    labels = ["S1", "S2", "c0", "c1", "c2", "c3", "c4"]
    #else:
    #    labels = ["S2", "c0", "c1", "c2", "c3", "c4"]
    labels = ["S1", "c1", "S2", "c2", "S3", "c3", "S4", "c4"]

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



def plot_chain(samples, fkey="tmp", save=True, S1flg=True):
    ndim    = np.shape(samples)[-1]
    fig, axes = plt.subplots(ndim, 1, figsize=(6, 1.5*ndim), sharex=True)
    #if S1flg:
    #    labels = ["S1", "S2", "c0", "c1", "c2", "c3", "c4"]
    #else:
    #    labels = ["S2", "c0", "c1", "c2", "c3", "c4"]
    labels = ["S1", "c1", "S2", "c2", "S3", "c3", "S4", "c4", "S5", "c5"]
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

def plot_scatter(vdata, udata, args, MISTdata, fluxmodel, hmodelfunc, fkey="tmp", save=True, S1flg=True):

    newx    = np.linspace(min(vdata[:,0]), max(udata[:,0]), 300)
    newlogg = iu.logg(MISTdata, newx)

    fig     = plt.figure(figsize=(5.5,4.5))

    ax1     = fig.add_subplot(3,1,(1,2))
    if S1flg:
        hThKp,_ = hmodelfunc(newx, args[2:], fluxmodel, newlogg, S1=args[0], S2=args[1])
    else:
        hThKp,_ = hmodelfunc(newx, args[1:], fluxmodel, newlogg, S1=0, S2=args[1])

    ax1.plot(newx, hThKp, c='darkorange', zorder=3)
    ax1.scatter(vdata[:,0], vdata[:,1], s=10, c='lightgrey', zorder=1)
    ax1.errorbar(udata[:,0], udata[:,1], yerr=udata[:,2], fmt='o', ms=3.5, elinewidth=0.5, c='black', zorder=2)
    ax1.axhline(1., lw=1, c='black', ls=':', zorder=0)
    #ax1.set_yscale("log")
    ax1.axes.xaxis.set_visible(False)
    ax1.set_ylabel("$h_{T}$/$h_{Kp}$")
    #ax1.set_ylim((5e-2, 2e1))
    ax1.set_ylim((0, 2.))
    ax1.set_yticks([0.2,0.6,1.0,1.4,1.8])

    if S1flg:
        hThKp_s,_   = hmodelfunc(udata[:,0], args[2:], fluxmodel, newlogg, S1=args[0], S2=args[1])
    else:
        hThKp_s,_   = hmodelfunc(udata[:,0], args[1:], fluxmodel, newlogg, S1=0, S2=args[1])

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

    return hThKp_s