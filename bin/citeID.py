import numpy as np
import sys
import pandas as pd

def radec_byepic(epicid_array):
    df      = pd.read_csv("./k2ticxmatch_20210831.csv",dtype='unicode')
    id_ra_dec   = []
    
    for epicid in epicid_array:
        hit_n   = len(df[df['epic']==str(epicid)].index)
        if hit_n != 0:
            target  = df[df['epic']==str(epicid)]
            for i in range(hit_n):
                ra  = float(target.iat[i,2])
                dec = float(target.iat[i,3])
                id_ra_dec.append([epicid, ra, dec])
    id_ra_dec   = np.array(id_ra_dec, dtype='f8')
    return id_ra_dec
    
def lamostid_byepic(epicid_array):
    """
        epicid -> lamost obj id and mjd.
    """
    listdata    = np.loadtxt("./lamost_table.csv", delimiter='|',\
                            dtype='unicode')

    in_ra   = listdata[:,(listdata[0]=="inputobjs_input_ra")].flatten()
    in_dec  = listdata[:,(listdata[0]=="inputobjs_input_dec")].flatten()
    in_ra   = np.array(in_ra[1:], dtype='f8')
    in_dec  = np.array(in_dec[1:], dtype='f8')

    ids     = listdata[:,(listdata[0]=="combined_obsid")].flatten()
    mjd     = listdata[:,(listdata[0]=="combined_mjd")].flatten()
    ids     = ids[1:]
    mjd     = mjd[1:]

    id_ra_dec   = radec_byepic(epicid_array)
    joint_arr   = []
    for ird in id_ra_dec:
        index   = ((in_ra==ird[1])&(in_dec==ird[2]))
        id_out  = ids[index]
        mjd_out = mjd[index]

        for obs_id, obs_mjd in zip(id_out, mjd_out):
            joint_arr.append([ird[0], ird[1], ird[2],\
                            obs_id, obs_mjd])
            #epicid, ra, dec, objid, mjd
    
    joint_arr   = np.array(joint_arr, dtype='f8')
    return joint_arr

