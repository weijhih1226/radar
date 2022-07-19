########################################
########### cross_section.py ###########
######## Author: Wei-Jhih Chen #########
######### Update: 2022/07/19 ###########
########################################

import numpy as np

def find_nearest_azimuth(refLon , refLat , selLon , selLat):
    lon_deg2km = 102.8282                               # Degree to km (Units: km/deg.)
    lat_deg2km = 111.1361                               # Degree to km (Units: km/deg.)
    x = (selLon - refLon) * lon_deg2km
    y = (selLat - refLat) * lat_deg2km
    dis = np.sqrt(x ** 2 + y ** 2)

    if x >= 0 and y >= 0:
        azi = np.arctan(y / x)
    elif x < 0:
        azi = np.arctan(y / x) + np.pi
    elif x >= 0 and y < 0:
        azi = np.arctan(y / x) + np.pi * 2
    azi = (-azi * 180 / np.pi + 90) % 360
    return azi , dis

def cross_section(varDZ , varZD , varPH , varKD , varRH , varVR , varSW , azi , selAzi):
    selAzi_idx = np.argmin(np.abs(azi - selAzi))
    selAzi_val = azi[selAzi_idx]
    varDZ_cs = varDZ[selAzi_idx , :]
    varZD_cs = varZD[selAzi_idx , :]
    varPH_cs = varPH[selAzi_idx , :]
    varKD_cs = varKD[selAzi_idx , :]
    varRH_cs = varRH[selAzi_idx , :]
    varVR_cs = varVR[selAzi_idx , :]
    varSW_cs = varSW[selAzi_idx , :]
    return varDZ_cs , varZD_cs , varPH_cs , varKD_cs , varRH_cs , varVR_cs , varSW_cs , selAzi_val