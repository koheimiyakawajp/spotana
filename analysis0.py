#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import sys
from copy import copy
import matplotlib.colors as colors

def plot_vj_MK(prpdata, ax=np.nan):

    k2id   = np.array(prpdata[1:,0], dtype='f8')
    j   = np.array(prpdata[1:,1], dtype='f8')
    jer = np.array(prpdata[1:,2], dtype='f8')
    k   = np.array(prpdata[1:,5], dtype='f8')
    ker = np.array(prpdata[1:,6], dtype='f8')
    v   = np.array(prpdata[1:,7], dtype='f8')
    rw  = np.array(prpdata[1:,11], dtype='f8')
    ga  = np.array(prpdata[1:,12], dtype='f8')
    d   = np.array(prpdata[1:,13], dtype='f8')
    plx = prpdata[1:,8]
    per = prpdata[1:,9]
    
    k2id   = k2id[(plx!=('--'))]
    j   = j[(plx != '--')]
    jer = jer[(plx != '--')]
    k   = k[(plx != '--')]
    ker = ker[(plx != '--')]
    v   = v[(plx != '--')]
    rw  = rw[(plx != '--')]
    ga  = ga[(plx != '--')]
    d   = d[(plx != '--')]
    vj  = v-j
    per = per[(plx!='--')]
    plx = plx[(plx!='--')]
    plx = np.array(plx, dtype='f8')
    per = np.array(per, dtype='f8')
    MK  = k + 5 + 5*np.log10(plx*1e-3)
    MKp = k+ker + 5 + 5*np.log10((plx-per)*1e-3)
    MKm = k-ker + 5 + 5*np.log10((plx+per)*1e-3)
    MKe = np.abs((MKp - MKm)/2.)

    mksar   = np.array((np.sort(vj), MK[(np.argsort(vj))], jer[(np.argsort(vj))],\
         MKe[(np.argsort(vj))],np.arange(len(vj))[np.argsort(vj)]))
    mkmed   = medfilt(mksar[1],kernel_size=11)
    resid   = np.abs(mksar[1] - mkmed)

    con0    = ((1.2<=vj)&(vj<=7))
    con1    = ((rw< 1.4)&((ga< 20)|(d< 5)))
    con2    = np.array([not x for x in con1])
    con0    = con0[np.argsort(vj)]
    con1    = con1[np.argsort(vj)]
    #con2    = ((rw>=1.4)|((ga>=20)&(d>=5)))
    #con2    = np.array(map(lambda x: not x, con1))
    con2    = con2[np.argsort(vj)]
    #plt.scatter(mksar[0],resid)
    #plt.show()
    contot  = (((resid<1)&con1)&con0)
    contot_not  = np.array([not x for x in contot])

    ms      = 4
    if ax is not np.nan:
        #mkok    = copy(mksar[:,((resid< 1)|con1)&(con0)])
        #mkng    = copy(mksar[:,((resid>=1)|con2)&(con0)])
        #mkok    = copy(mksar[:,((resid< 1)&con1)&(con0)])
        #mkng    = copy(mksar[:,((resid>=1)|con2)&(con0)])
        mkok    = copy(mksar[:,contot])
        mkng    = copy(mksar[:,contot_not])
        ax.errorbar(mkok[0], mkok[1], xerr=mkok[2], yerr=mkok[3], c='orange', fmt='o', capsize=0, \
             ecolor='black', markeredgewidth=0.5,markeredgecolor='black', markersize=ms, elinewidth=0.5)
        ax.errorbar(mkng[0], mkng[1], xerr=mkng[2], yerr=mkng[3], c='black',  fmt='o', capsize=0,\
             ecolor='black', markeredgewidth=0.5,markeredgecolor='black', markersize=ms, elinewidth=0.5)
        ax.set_xlabel("$V~-~J$")
        ax.set_xlim((0.2,6.5)); ax.set_ylim((0.1,9.3))
        ax.set_ylabel("M$_K$")
    
        ax.grid(ls='--', lw=0.5)
    flg     = np.where(((resid >=1)|(con2)|(vj<1.2)|(7<vj)), 1, 0)
    flgsort = flg[np.argsort(mksar[-1])]
    return  np.array(np.array(k2id[(flgsort==1)],dtype='i8'), dtype='unicode')

def plot_teff_mass(ax, tf, ms, tf_er, ms_er):
    ax.scatter(tf,ms,c="orange",ec='black',s=15,zorder=3,label="RUWE<1.4", linewidth=0.5)
    ax.errorbar(tf,ms,xerr=tf_er,yerr=ms_er,fmt='.',c='black',capsize=0,zorder=1,\
        elinewidth=0.5)#,c=rw,s=30,cmap=plt.cm.jet,ec='gray',vmin=1,vmax=1.5)
    ax.set_xlabel("Effective Temperature [K]")
    #ax.set_ylabel("Stellar Mass [M$_{\odot}$]")
    #ax.set_ylabel("M$_K$")
    ax.grid(ls='--', lw=0.5)
    ax.set_xlim((2500,5800))
    ax.set_ylim((0.1,9.3))

def cal_err_wari(a,b,ae,be):
    er  = ((ae/b)**2+(a*be/b**2)**2)**0.5
    return er

if __name__=='__main__':

    print("hello world")

    if len(sys.argv) == 2:
        prpfile     = sys.argv[1]

        prpdata     = np.loadtxt(prpfile, dtype='unicode', comments='#', delimiter=',') 
        #print(prptit)
        #exit()
        fig     = plt.figure(figsize=(7,3))

        #plt.rcParams["font.family"] = "Arial"

        ax1     = fig.add_subplot(1,2,1)
        rmids   = plot_vj_MK(prpdata,ax1)
        for rmid in rmids:
            prpdata = prpdata[(prpdata[:,0] != rmid)]
        
        con1    = (prpdata[:,16]!='nan')&(prpdata[:,17]!='nan')
        con2    = (prpdata[:,14]!='nan')&(prpdata[:,15]!='nan')
        prpdata     = prpdata[con1&con2]
        np.savetxt(prpfile.split(".cs")[0]+"_selected.csv", prpdata, delimiter=',', fmt='%s')
        
        prpval      = prpdata[1:]
        print("valid targets", len(prpval))

        k2id   = np.array(prpval[:,0], dtype='f8')

        k   = np.array(prpval[:,5], dtype='f8')
        ker = np.array(prpval[:,6], dtype='f8')
        plx = np.array(prpval[:,8], dtype='f8')
        per = np.array(prpval[:,9], dtype='f8')
        MK  = k + 5 + 5*np.log10(plx*1e-3)
        MKp = k+ker + 5 + 5*np.log10((plx-per)*1e-3)
        MKm = k-ker + 5 + 5*np.log10((plx+per)*1e-3)
        MKe = np.abs((MKp - MKm)/2.)

        tf  = np.array(prpval[:,16], dtype='f8')
        ms  = np.array(prpval[:,14], dtype='f8')
        tf_er  = np.array(prpval[:,17], dtype='f8')
        ms_er  = np.array(prpval[:,15], dtype='f8')
        rw  = np.array(prpval[:,11], dtype='f8')
        ga  = np.array(prpval[:,12], dtype='f8')
        d   = np.array(prpval[:,13], dtype='f8')
        #ax1.scatter(tf[(rw<1.4)],ms[(rw<1.4)],c="orange",s=30,cmap=plt.cm.jet,lw=1,ec='gray',vmin=1,vmax=1.5,zorder=2)
        ax2     = fig.add_subplot(1,2,2, sharey=ax1)
        #plot_teff_mass(ax2, tf, ms, tf_er, ms_er)
        plot_teff_mass(ax2, tf, MK, tf_er, MKe)
        plt.subplots_adjust(hspace=.0)
        plt.tight_layout()
        #plt.show()
        #exit()
        plt.savefig(prpfile.split(".cs")[0]+".png", dpi=200)