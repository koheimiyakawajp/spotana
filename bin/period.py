import numpy as np
from scipy import signal as sg
from matplotlib import pyplot as plt
from transitleastsquares import transitleastsquares as tls
from transitleastsquares import transit_mask
import copy
import random
from multiprocessing import Pool,cpu_count

def fap(pgram, prob, ndata):
    n   = ndata
    med = np.median(pgram)
    std = 1.48*np.median(np.abs(pgram-med))
    m   = len(pgram[(med+std*3 < pgram)])
    val = 1-np.power(1-np.power(1-prob,1./m),2./(n-3.))

    return val

def find_peak(freq, pgram, thres=0.01):
    peaks       = pgram[(thres<pgram)]
    freq_p      = freq[(thres<pgram)]

    return freq_p, peaks

def lomb_scargle(data, N=1000, pmin=0.1, pmax=10., prob=0.01):
    time    = copy.copy(data[0])
    flux    = copy.copy(data[1])

    timein  = time - time[0]
    freq    = np.linspace(2.*np.pi/pmax, 2.*np.pi/pmin, N)

    pgram   = sg.lombscargle(timein, flux, freq)

    thres   = fap(pgram, prob, len(time))

    f,p     = find_peak(freq, pgram, thres)

    return 2*np.pi/f,p,2*np.pi/freq,pgram

def lombscargle_wrap(input_list):
    timein,fapp,freq    = input_list
    pgram   = sg.lombscargle(timein, fapp, freq)
    return pgram


def sigmin_bootstrap(data, N=1000, pmin=0.1, pmax=10, nboot=100, seed=0):
    time    = copy.copy(data[0])
    fluxs   = copy.copy(data[1])

    timein  = time - time[0]
    freq    = np.linspace(2.*np.pi/pmax, 2.*np.pi/pmin, N)

    np.random.seed(seed)
    res     = []
    data_list   = []
    for i in range(nboot):
        np.random.shuffle(fluxs)
        fapp    = copy.copy(fluxs)
        data_list.append([timein, fapp, freq])

    p   = Pool(cpu_count())
    res     = p.map(lombscargle_wrap, data_list)
    p.close()
    res     = np.hstack(res)
    res     = np.sort(res)
    sigmin  = res[int(len(res)*0.999)]
    return sigmin


def tls_periodogram(data, rad_star=1., mas_star=1.):
    data[1]     = data[1] - np.median(data[1]) + 1.
    model       = tls(data[0], data[1])
    pgram       = model.power(R_star=rad_star, M_star=mas_star,\
                              duration_grid_step=2., oversampling_factor=1)

    return pgram

def mask_transit(data, period, duration, T0):
    med     = np.median(data[1])
    intransit   = transit_mask(data[0], period, duration, T0)
    data_ot     = copy.copy(data)
    data_ot[1,intransit]    = med
    return data_ot
