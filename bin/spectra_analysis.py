import numpy as np

def ext_CaIIHK(spectra):
    w1  = 3900
    w2  = 4000

    return spectra[((w1<spectra[:,0])&(spectra[:,0]<w2))]

def ext_Halpha(spectra):
    w1  = 6500
    w2  = 6600
    
    return spectra[((w1<spectra[:,0])&(spectra[:,0]<w2))]

def ext_FeII(spectra):
    w1  = 4890
    w2  = 5200

    return spectra[((w1<spectra[:,0])&(spectra[:,0]<w2))]