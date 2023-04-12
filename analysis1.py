#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import sys
from copy import copy
import matplotlib.colors as colors

def cal_err_wari(a,b,ae,be):
    er  = ((ae/b)**2+(a*be/b**2)**2)**0.5
    return er

if __name__=='__main__':

    print("hellow world")

    if len(sys.argv) == 3:
        ampfile     = sys.argv[1]
        prpfile     = sys.argv[2]

        ampdata     = np.loadtxt(ampfile, dtype='unicode', comments='#')
        ampval      = np.array(ampdata[1:], dtype='f8')
        epicid      = ampval[:,0]
        _,con0      = np.unique(epicid, return_index=True)
        ampval      = ampval[con0]

        amptit      = ampdata[0]
        prpdata     = np.loadtxt(prpfile, dtype='unicode', comments='#', delimiter=',') 
        prpval      = np.array(prpdata[1:], dtype='f8')
        prptit      = prpdata[0]
        #print(prptit)
        #exit()

        #print(ampdata,prpdata)
        res     = []
        for dline in ampval:
            print(dline)
            #exit()
            epic    = dline[0]
            con     = (np.array(prpval[:,0], dtype='f8') == epic)
            if np.any(con) :
                prop    = prpval[(np.array(prpval[:,0], dtype='f8') == epic)][0]

                teff    = float(prop[(prptit=="teff")][0])
                teff_er = float(prop[(prptit=="teff_er")][0])
                mass    = float(prop[(prptit=="mass")][0])
                mass_er = float(prop[(prptit=="mass_er")][0])
                ruwe    = float(prop[(prptit=="ruwe")][0])
                hmag    = float(prop[(prptit=="hmag")][0])

                # ---------------
                if dline[(amptit=="wnk2")][0] < dline[(amptit=="wnk2sp")][0]:
                    ampk2   = dline[(amptit=="ampk2")][0]
                    erk2    = dline[(amptit=="erk2")][0]
                    wnk2    = dline[(amptit=="wnk2")][0]
                    rnk2    = dline[(amptit=="redk2")][0]
                else:
                    ampk2   = dline[(amptit=="ampk2sp")][0]
                    erk2    = dline[(amptit=="erk2sp")][0]
                    wnk2    = dline[(amptit=="wnk2sp")][0]
                    rnk2    = dline[(amptit=="redk2sp")][0]
                
                amptess     = dline[(amptit=="amptess")][0]
                ertess      = dline[(amptit=="ertess")][0]
                wntess      = dline[(amptit=="wntess")][0]
                rntess      = dline[(amptit=="redtess")][0]

                shotk2      = dline[(amptit=="shotk2")][0]
                shottess    = dline[(amptit=="shottess")][0]

                p_k2        = dline[(amptit=="pbest")][0]
                p_tess      = dline[(amptit=="pbest_tqlp")][0]

                array_mrg   = np.array((epic, ampk2, erk2, rnk2, wnk2, amptess, ertess, rntess, wntess,\
                                            shotk2, shottess, p_k2, p_tess,\
                                                teff, teff_er, mass, mass_er, ruwe, hmag))

                #res.append(np.hstack((dline, teff, teff_er, mass, mass_er, ruwe, hmag)))
                res.append(array_mrg)


        res     = np.array(res,dtype='f8')
        #mscon   = (res[:,4]>res[:,6])
        #res     = res[mscon]
        print(res)

        #--------------
        EPICID  = res[:,0]     
        AM_K    = res[:,1]
        AM_K_e  = res[:,2]
        RN_K    = res[:,3]
        WN_K    = res[:,4]
        AM_T    = res[:,5]
        AM_T_e  = res[:,6]
        RN_T    = res[:,7]
        WN_T    = res[:,8]
        #AM_T    = res[:,7]
        #AM_T_e  = res[:,8]
        #WN_T    = res[:,9]
        SHOT_K  = res[:,9]
        SHOT_T  = res[:,10]
        Prot_K  = res[:,11]
        Prot_T  = res[:,12]
        print(Prot_K, Prot_T)
        Prot_VA = np.abs(Prot_K - Prot_T)/Prot_K
        Prot_VAab   = np.abs(Prot_K - Prot_T)
        #Prot_VA = (Prot_T - Prot_K)/Prot_K
        Teff    = res[:,13]
        Teff_e  = res[:,14]
        HMAG    = res[:,-1]
        #--------------

        fig     = plt.figure(figsize=(12,6.5))
        plt.rcParams["font.family"] = "Arial"
        # plot K2 noise
        ax1     = fig.add_subplot(2,3,1)
        #mp1     = ax1.scatter(Teff, NREL_K, s= 16, c=Prot_K, ec='orangered', norm=colors.LogNorm(vmin=0.1,vmax=35))
        mp1     = ax1.scatter(Teff, WN_K, s= 16, c=Prot_K, ec='orangered', norm=colors.LogNorm(vmin=0.1,vmax=35), label='obs noise')
        #mp1     = ax1.scatter(HMAG, WN_K, s= 16, c=Prot_K, ec='orangered', norm=colors.LogNorm(vmin=0.1,vmax=35), label='obs noise')
        ax1.scatter(Teff, SHOT_K, s= 16, c=Prot_K, ec='dimgrey' ,norm=colors.LogNorm(vmin=0.1,vmax=35), label='shot noise')
        #ax1.scatter(HMAG, SHOT_K, s= 16, c=Prot_K, ec='dimgrey' ,norm=colors.LogNorm(vmin=0.1,vmax=35), label='shot noise')
        cb1     = fig.colorbar(mp1, ax=ax1)
        cb1.set_label("Rotation Period [d]")
        ax1.set_yscale("log")
        ax1.set_ylabel("SD of noise")
        ax1.set_xlabel("Effective Temperature [K]")
        ax1.legend()
        print("EPICID" ,np.array(EPICID[((4500<Teff)&(5e-4<WN_K)&(Prot_K>2))],dtype='i8'))

        # plot TESS noise
        ax2     = fig.add_subplot(2,3,2)
        #mp2     = ax2.scatter(Teff, WN_T, s= 16, c=Prot_K, ec='dimgrey', marker='^', norm=colors.LogNorm(vmin=0.1,vmax=35))
        #mp2     = ax2.scatter(Teff, NREL_T, s= 16, c=Prot_K, ec='orangered', norm=colors.LogNorm(vmin=0.1,vmax=35))
        mp2     = ax2.scatter(Teff, WN_T, s= 16, c=Prot_K, ec='orangered', norm=colors.LogNorm(vmin=0.1,vmax=35), label='obs noise')
        #mp2     = ax2.scatter(HMAG, WN_T, s= 16, c=Prot_K, ec='orangered', norm=colors.LogNorm(vmin=0.1,vmax=35), label='obs noise')
        ax2.scatter(Teff, SHOT_T, s= 16, c=Prot_K, ec='dimgrey' ,norm=colors.LogNorm(vmin=0.1,vmax=35), label='shot noise')
        #ax2.scatter(HMAG, SHOT_T, s= 16, c=Prot_K, ec='dimgrey' ,norm=colors.LogNorm(vmin=0.1,vmax=35), label='shot noise')
        cb2     = fig.colorbar(mp2, ax=ax2)
        cb2.set_label("Rotation Period [d]")
        ax2.set_yscale("log")
        ax2.set_ylabel("SD of noise")
        ax2.set_xlabel("Effective Temperature [K]")
        ax2.legend()


        c_K     = np.median(WN_K/SHOT_K)
        c_T     = np.median(WN_T/SHOT_T)
        print(c_K, c_T)

        SQRED_K = WN_K**2 - SHOT_K**2
        COM_K   = np.min(SQRED_K)
        SQRED_T = WN_T**2 - SHOT_T**2
        COM_T   = np.min(SQRED_T)

        NREL_K  = np.sqrt(WN_K**2-SHOT_K**2)# - COM_K)
        NREL_T  = np.sqrt(WN_T**2-SHOT_T**2)# - COM_T)
        #NREL_K  = WN_K**2-c_K*SHOT_K**2# - COM_K)
        #NREL_T  = WN_T**2-c_T*SHOT_T**2# - COM_T)

        #ax3     = fig.add_subplot(2,3,3)
        ##mp2     = ax2.scatter(Teff, WN_T, s= 16, c=Prot_K, ec='dimgrey', marker='^', norm=colors.LogNorm(vmin=0.1,vmax=35))
        #mp3     = ax3.scatter(Teff, NREL_T, s= 16, c=Prot_K, ec='orangered', norm=colors.LogNorm(vmin=0.1,vmax=35))
        ##mp3     = ax3.scatter(Prot_K, NREL_T, s= 16, c=Teff, ec='orangered', vmin=3000,vmax=5000)
        ##mp3     = ax3.scatter(HMAG, NREL_T, s= 16, c=Teff, ec='orangered', vmin=3000,vmax=5000)
        ##mp3     = ax3.scatter(Teff, NREL_T/NREL_K, s= 16, c=Prot_K, ec='orangered', norm=colors.LogNorm(vmin=0.1,vmax=35))
        #ax3.scatter(Teff, NREL_K, s= 16, c=Prot_K, ec='seagreen', norm=colors.LogNorm(vmin=0.1,vmax=35))
        ##ax3.scatter(Prot_K, NREL_K, s= 16, c=Teff, ec='seagreen', vmin=3000,vmax=5000)
        ##ax3.scatter(Teff, SHOT_T, s= 16, c=Prot_K, ec='dimgrey' ,norm=colors.LogNorm(vmin=0.1,vmax=35), label='shot noise')
        ##ax3.scatter(HMAG, NREL_K, s= 16, c=Prot_K, ec='dimgrey' ,norm=colors.LogNorm(vmin=0.1,vmax=35), label='shot noise')
        #cb3     = fig.colorbar(mp3, ax=ax3)
        #cb3.set_label("Rotation Period [d]")
        #ax3.set_yscale("log")
        ##ax3.set_xscale("log")
        #ax3.set_ylabel("additional noise")
        ##ax3.set_ylim((5e-5,3e-1))
        #ax3.set_xlabel("Effective Temperature [K]")
        ##ax3.legend()
        
        
        ax3     = fig.add_subplot(2,3,3)
        #mp3     = ax3.scatter(Teff, RN_K, s= 16, c=Prot_K, ec='orangered', norm=colors.LogNorm(vmin=0.1,vmax=35), label='K2 red noise')
        #mp3     = ax3.scatter(Teff, AM_K, s= 16, c=Prot_K, ec='orangered', norm=colors.LogNorm(vmin=0.1,vmax=35), label='K2 red noise')
        mp3     = ax3.scatter(AM_K, AM_T/AM_K, s= 16, c=Prot_K, ec='orangered', norm=colors.LogNorm(vmin=0.1,vmax=35), label='K2 red noise')
        #mp2     = ax2.scatter(HMAG, WN_T, s= 16, c=Prot_K, ec='orangered', norm=colors.LogNorm(vmin=0.1,vmax=35), label='obs noise')
        #ax3.scatter(Teff, RN_T, s= 16, c=Prot_K, ec='dimgrey' ,norm=colors.LogNorm(vmin=0.1,vmax=35), label='TESS red noise')
        #ax3.scatter(Teff, AM_T, s= 16, c=Prot_K, ec='dimgrey' ,norm=colors.LogNorm(vmin=0.1,vmax=35), label='TESS red noise')
        #ax2.scatter(HMAG, SHOT_T, s= 16, c=Prot_K, ec='dimgrey' ,norm=colors.LogNorm(vmin=0.1,vmax=35), label='shot noise')
        cb3     = fig.colorbar(mp3, ax=ax3)
        cb3.set_label("Rotation Period [d]")
        ax3.set_yscale("log")
        ax3.set_xscale("log")
        #ax3.set_ylim((3e-5, 4e-2))
        ax3.set_ylabel("SD of noise")
        ax3.set_xlabel("Effective Temperature [K]")
        ax3.legend()

        print(len(EPICID))

        #con1   = ((Teff > 0)&((AM_K > WN_K) &(AM_T > WN_T)))
        #con1   = ((Teff > 0)&((AM_K > RN_K) &(AM_T > RN_T)))
        con1   = ((Teff > 0)&(Prot_VA <=0.2))
        #con1   = ((Teff > 0)&(Prot_VA <=0.2)&(Prot_VAab <= 1))
        #con1   = (Teff > 0)
        #print(len(EPICID[con1]))

        ax4     = fig.add_subplot(2,3,4)
        ax4.errorbar(AM_K[con1], AM_T[con1], xerr=AM_K_e[con1], yerr=AM_T_e[con1],\
            zorder=1, markersize=0.1, fmt='.', capsize=0.0, c='black', elinewidth=0.2)
        ax4.plot(np.linspace(1e-4,1e-1), np.linspace(1e-4,1e-1), c='black', ls=':', lw=1)
        mp4     = ax4.scatter(AM_K[con1], AM_T[con1],  c=Teff[con1], s=10, zorder=3, linewidth=0.5 ,ec='gray',cmap=plt.cm.plasma)
        ax4.set_xlabel("$h_{Kp}$ : Semi-Amplitude in K2")
        ax4.set_ylabel("$h_T$ : Semi-Amplitude in TESS")
        ax4.set_xlim((2e-4, 1e-1));ax4.set_ylim((2e-4, 1e-1))
        ax4.set_xscale("log");ax4.set_yscale("log")
        cb4 = fig.colorbar(mp4, ax=ax4)
        cb4.set_label("Effective Temperature [K]")

        ax4     = fig.add_subplot(2,3,5) 
        AM_REL_e= cal_err_wari(AM_T, AM_K, AM_T_e, AM_K_e)
        #print(yer)
        #exit()
        AM_REL  = AM_T/AM_K
        ax4.errorbar(Teff[con1], AM_REL[con1], xerr=Teff_e[con1], yerr=AM_REL_e[con1],  zorder=1, \
            fmt='.', markersize=0.5, capsize=0.0, c='black', elinewidth=0.2)
        #mp4     = ax4.scatter(Teff[con1], AM_REL[con1], c=Prot_K[con1], s=15, ec='gray' ,zorder=3,\
        con_10  = (con1&(10.<Prot_K))
        con_1   = (con1&((1.<Prot_K)&(Prot_K<=10.)))
        con_01  = (con1&(Prot_K<=1.))
        #mp4     = ax4.scatter(Teff[con1], AM_REL[con1], c=Prot_K, s=10, ec='gray' ,zorder=3, lw=0.5) 
        ax4.scatter(Teff[con_10], AM_REL[con_10], c='gold', s=10,  ec='black' ,zorder=3, lw=0.5) 
        ax4.scatter(Teff[con_1 ], AM_REL[con_1 ], c='limegreen', s=10, ec='black' ,zorder=3, lw=0.5) 
        ax4.scatter(Teff[con_01], AM_REL[con_01], c='black', s=10, ec='black' ,zorder=3, lw=0.5) 
        Tecon1  = Teff[con1]
        ARcon1  = AM_REL[con1]
        isort   = np.argsort(Tecon1)
        ax4.plot(Tecon1[isort], medfilt(ARcon1[isort],kernel_size=11), lw=1, c='blue', zorder=10) 
        #mp4     = ax4.scatter(Teff[con1&con2], AM_REL[con1&con2], c=AM_T[con1&con2], s=15, ec='gray' ,zorder=3,\
        #mp4     = ax4.scatter(Teff[con1], AM_REL[con1], c=Prot_VA[con1], s=15, ec='gray' ,zorder=3,\
                #norm=colors.LogNorm(vmin=0.1,vmax=35), \
                #linewidth=0.5, vmin=0.5, vmax=1.5)
        #mp      = ax4.scatter(res[:,7], res[:,5]/res[:,1], c=res[:,-1], s=30, ec='gray' ,zorder=3)
        ax4.set_yscale("log")
        ax4.set_ylabel("$h_{T}/h_{Kp}$")
        ax4.set_xlabel("Effective Temperature [K]")
        ax4.axhline(1,lw=1,ls=':',c='black')
        #cb4     = fig.colorbar(mp4, ax=ax4)
        #cb4.set_label("Rotation Period [d]")
        #print(np.array(EPICID[con1&(AM_REL > 2)], dtype='i8'))
        print(np.array(EPICID[con1&(AM_REL > 4)], dtype='i8'))

        ax6     = fig.add_subplot(2,3,6)
        #ax6.scatter(HMAG[con1], AM_REL[con1], s= 16, c=Prot_K[con1], ec='seagreen', norm=colors.LogNorm(vmin=0.1,vmax=35))
        #ax6.scatter(Teff[con1], Prot_VA[con1], s = 16, c=Prot_K[con1], ec='seagreen', norm=colors.LogNorm(vmin=0.1,vmax=35))
        ax6.scatter(Teff[con1], Prot_VAab[con1], s = 16, c=Prot_K[con1], ec='seagreen', norm=colors.LogNorm(vmin=0.1,vmax=35))
        #ax6.set_yscale("log")
        #ax6.set_ylabel("$h_{T}/h_{Kp}$")
        ax6.set_xlabel("Effective Temperature [K]")

        #ax6     = fig.add_subplot(2,3,6) 
        #ax6.scatter(WN_K[con1], AM_K[con1], c="dimgrey", s=15, ec='gray' ,zorder=3)
        #ax6.scatter(WN_T[con1], AM_T[con1], c="orangered", s=15, ec='orangered' ,zorder=3)
        #ax6.set_yscale("log")
        #ax6.set_xscale("log")
        #ax6.set_ylim((3e-5,1e-1))
        #ax6.set_xlim((3e-5,1e-1))

        vresult     = np.array((Teff[con1], AM_REL[con1], AM_REL_e[con1],\
            AM_K[con1], AM_K_e[con1], AM_T[con1], AM_T_e[con1],Prot_K[con1]))
        filekey     = ampfile.split("/")[-1]
        filekey     = filekey.split("_")[0]
        np.savetxt("./vresult/"+filekey+"_relamp.dat", vresult.T)

        plt.tight_layout()
        plt.show();exit()
        outname     = ampfile.split(".")[0] + "_plot.png"
        plt.savefig(outname, dpi=300)
        #plt.show()