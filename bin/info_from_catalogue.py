#!/usr/bin/env python3

import astropy.units as u
from astropy.coordinates import SkyCoord
import numpy as np
from astroquery.gaia import Gaia
import sys

from astroquery.vizier import Vizier
from astroquery.ipac.irsa import Irsa
from astropy.table import Table
import math


def k2id_to_cood(k2id):
    filelist    = "./k2ticxmatch_20210831.csv"
    tarlist     = np.loadtxt(filelist, comments='tid', delimiter=",", usecols=[0,1,2,3], dtype='unicode')
    hittar      = tarlist[(tarlist[:,1]==str(int(k2id)))]
    if len(hittar) == 0:
        return np.nan, np.nan
    else:
        hittar      = hittar[0]
        ra          = hittar[2]
        dec         = hittar[3]

        return float(ra),float(dec)

def get_gaiadr2(k2id, rad=0.1):
    
    ra,dec  = k2id_to_cood(k2id)
    if math.isnan(ra):
        return np.nan,np.nan,np.nan,np.nan,np.nan

    Gaia.MAIN_GAIA_TABLE="gaiadr2.gaia_source"
    Gaia.ROW_LIMIT  = 1
    coord   = SkyCoord(ra=ra,dec=dec,unit=(u.degree,u.degree),\
        frame='icrs')
    radius  = u.Quantity(rad, u.deg)

    j   = Gaia.cone_search_async(coord, radius)
    r   = j.get_results()
    #print(r)
    #for i in r.columns:
    #    print(i)
    #exit()
    
    try:
        plx     = r['parallax'][0]
        plx_er  = r['parallax_error'][0]
        bprp    = r['bp_rp'][0]
        gof_al  = r['astrometric_gof_al'][0]
        d       = r['astrometric_excess_noise_sig'][0]
        return plx,plx_er,bprp,gof_al,d
    except:
        return np.nan,np.nan,np.nan,np.nan,np.nan


def get_gaia_temperature(k2id, rad=0.1):
    
    ra,dec  = k2id_to_cood(k2id)
    if math.isnan(ra):
        return np.nan,np.nan

    Gaia.MAIN_GAIA_TABLE="gaiadr3.gaia_source"
    #Gaia.MAIN_GAIA_TABLE="gaiaedr3.gaia_source"
    #Gaia.MAIN_GAIA_TABLE="gaiadr2.gaia_source"
    Gaia.ROW_LIMIT  = 1
    coord   = SkyCoord(ra=ra,dec=dec,unit=(u.degree,u.degree),\
        frame='icrs')
    radius  = u.Quantity(rad, u.deg)

    j   = Gaia.cone_search_async(coord, radius)
    r   = j.get_results()
    #for i in r.columns:
    #    print(i)

    if (r['teff_gspphot'][0] is np.ma.masked) :
        Gaia.MAIN_GAIA_TABLE="gaiadr2.gaia_source"
        Gaia.ROW_LIMIT  = 1
        coord   = SkyCoord(ra=ra,dec=dec,unit=(u.degree,u.degree),\
            frame='icrs')
        radius  = u.Quantity(rad, u.deg)

        j   = Gaia.cone_search_async(coord, radius)
        r   = j.get_results()
        #for i in r.columns:
        #    print(i)
        #print(r['teff_val'][0])
        #print(r['teff_percentile_lower'][0])
        #print(r['teff_percentile_upper'][0])
        teff    = r['teff_val'][0]
        er1     = r['teff_percentile_lower'][0]
        er2     = r['teff_percentile_upper'][0]

    else:
        teff    = r['teff_gspphot'][0]
        er1     = r['teff_gspphot_lower'][0]
        er2     = r['teff_gspphot_upper'][0]
    
    return teff, (er2 - er1)/2.

def get_gaia(k2id, rad=0.01):
    
    ra,dec  = k2id_to_cood(k2id)
    if math.isnan(ra):
        return np.nan,np.nan,np.nan,np.nan,np.nan,np.nan

    Gaia.MAIN_GAIA_TABLE="gaiadr3.gaia_source"
    #Gaia.MAIN_GAIA_TABLE="gaiaedr3.gaia_source"
    #Gaia.MAIN_GAIA_TABLE="gaiadr2.gaia_source"
    Gaia.ROW_LIMIT  = 1
    coord   = SkyCoord(ra=ra,dec=dec,unit=(u.degree,u.degree),\
        frame='icrs')
    radius  = u.Quantity(rad, u.deg)

    j   = Gaia.cone_search_async(coord, radius)
    r   = j.get_results()
    #print(r)
    #for i in r.columns:
    #    print(i)
    ##print(r['teff_gspphot'][0])
    #print(r['teff_val'][0])
    #exit()
    
    plx     = r['parallax'][0]
    if plx > 0:
        plx_er  = r['parallax_error'][0]
        bprp    = r['bp_rp'][0]
        ruwe    = r['ruwe'][0]
        gof_al  = r['astrometric_gof_al'][0]
        d       = r['astrometric_excess_noise_sig'][0]
    else:
        plx,plx_er,bprp,gof_al,d    = get_gaiadr2(k2id, rad=rad)
        ruwe    = np.nan

    return plx,plx_er,bprp,ruwe,gof_al,d

def get_2mass(k2id, rad=0.1):
    ra,dec  = k2id_to_cood(k2id)
    if math.isnan(ra):
        return np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
    coord   = SkyCoord(ra=ra,dec=dec,unit=(u.degree,u.degree),\
        frame='icrs')
    table = Irsa.query_region(coord, catalog="fp_psc", spatial="Cone",radius=rad*u.deg)
    #for i in table.columns:
    #    print(i)

    jmag    = table["j_m"][0]
    jmag_er = table["j_cmsig"][0]

    hmag    = table["h_m"][0]
    hmag_er = table["h_cmsig"][0]

    kmag    = table["k_m"][0]
    kmag_er = table["k_cmsig"][0]

    return jmag,jmag_er,hmag,hmag_er,kmag,kmag_er

def get_tic(k2id, rad=0.1):
    ra,dec  = k2id_to_cood(k2id)
    if math.isnan(ra):
        return np.nan
    
    rad     = 0.1
    print(ra,dec)
    coord   = SkyCoord(ra=ra,dec=dec,unit=(u.degree,u.degree),\
        frame='icrs')
    result = Vizier.query_region(coord, radius=rad*u.deg,\
                                 catalog='J/AJ/156/102/table9') #TIC
                                 #catalog='IV/38/tic') #TIC
                                 #catalog='IV/34/epic') #TIC
                                 #catalog='IV/39/tic82') #TIC

                                 #catalog='I/322A/out')#UCAC4
    #print(result)
    #exit()
    #print(result[0])
    #for  i in result[0].columns:
    #    print(i)
    #exit()
    if len(result) != 0:
        vmag    = result[0]['Vmag'][0]
        return vmag
    else:
        return np.nan

def get_tycho(k2id, rad=0.1):
    ra,dec  = k2id_to_cood(k2id)
    coord   = SkyCoord(ra=ra,dec=dec,unit=(u.degree,u.degree),\
        frame='icrs')
    result = Vizier.query_region(coord, radius=rad*u.deg,
                                 catalog='I/239/tyc_main')
                                 #catalog='I/239/hip_main')
                                 #catalog='IV/34/epic')
                                 #catalog='II/285/photo')
                                 #catalog='I/196/main')
                                 #catalog='IV/39/tic82') #TIC
                                 #catalog='J/AJ/156/102/table9') #TIC
                                 #catalog='I/322A/out')#UCAC4
    #print(result[0])
    #for  i in result[0].columns:
    #    print(i)
    #exit()
    if len(result) != 0:
        vmag    = result[0]['Vmag'][0]
        return vmag
    else:
        return np.nan

def get_lamost(k2id, rad=0.1):
    ra,dec  = k2id_to_cood(k2id)
    if math.isnan(ra):
        return np.nan
    coord   = SkyCoord(ra=ra,dec=dec,unit=(u.degree,u.degree),\
        frame='icrs')
    result = Vizier.query_region(coord, radius=rad*u.deg,
                                 catalog='I/239/tyc_main')
    vmag    = result[0]['Vmag'][0]

    return vmag

if __name__=='__main__':
    k2id    = sys.argv[1]
    #Irsa.print_catalogs()
    #exit()
    #tables = Gaia.load_tables(only_names=True)
    #for table in (tables):
    #    print(table.get_qualified_name())
    #exit()
    #gaiadr3_table = Gaia.load_table('gaiadr3.gaia_source')
    #for column in gaiadr3_table.columns:
    #    print(column.name)
    #exit()

    #get_gaia(k2id, 0.01)
    #get_2mass(k2id, 0.01)
    get_tycho(k2id, 0.01)

