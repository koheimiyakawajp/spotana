#---------------------------
# program名: plot_matome
#
#

import matplotlib.pyplot as plt
import numpy as np
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


def dimplot(ndim, sampler):
    plt.figure(figsize=[8,8])
    #plt.figure()
    for i in range(ndim):
        for j in range(i+1):
            plt.subplot(ndim,ndim,1+i*ndim+j)
            if(i==j):
                plt.tick_params(labelleft=False)
                plt.hist(sampler.flatchain[:,i],20,color='sandybrown')#,\
            else:
                x   = sampler.flatchain[:,j]
                y   = sampler.flatchain[:,i]
                xl,xu,xs    = bound(x)
                yl,yu,ys    = bound(y)

                #xx,yy = np.mgrid[2000:4000:100,0:0.5:0.025]
                xx,yy = np.mgrid[xl:xu:xs,yl:yu:ys]

                positions = np.vstack([xx.ravel(),yy.ravel()])
                value = np.vstack([x,y])
                kernel = gaussian_kde(value)

                f = np.reshape(kernel(positions).T, xx.shape)
                plt.contourf(xx,yy,f,cmap=cm.Oranges)
    plt.show()
    #plt.savefig("corner_plot.png")

