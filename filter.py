########################################
############## filter.py ###############
######## Author: Wei-Jhih Chen #########
######### Update: 2022/07/19 ###########
########################################

import numpy as np
from numpy.ma import masked_array as mama

def var_filter(filVar , inVar , flMin , flMax):
    if flMin == None and flMax == None:
        outVar = inVar
    elif flMin == None:
        outVar = mama(inVar , filVar > flMax)
    elif flMax == None:
        outVar = mama(inVar , filVar < flMin)
    else:
        outVar = mama(inVar , (filVar < flMin) | (filVar > flMax))
    return outVar

def ZD_filter(varZD , var , flNum):
    num_azi = np.size(varZD , 0)
    num_rng = np.size(varZD , 1)
    varZD = varZD.filled(fill_value = np.nan)
    varSmZD = np.empty([num_azi , num_rng])
    varSmZD.fill(np.nan)
    for cnt_azi in np.arange(0 , num_azi):
        for cnt_rng in np.arange(0 , num_rng - (flNum - 1)):
            varSmZD[cnt_azi , cnt_rng + int((flNum - 1) / 2)] = np.mean(varZD[cnt_azi][cnt_rng : cnt_rng + (flNum - 1)])
    varZD_DV = varZD - varSmZD
    varZD_SdDV = np.nanstd(varZD_DV)
    varZD[np.isnan(varSmZD)] = np.nan
    var[np.isnan(varSmZD)] = np.nan
    varZD[np.abs(varZD_DV) > varZD_SdDV] = np.nan
    var[np.abs(varZD_DV) > varZD_SdDV] = np.nan
    return varZD , var

def KD_filter(varPH , range , smNum):
    num_azi = np.size(varPH , 0)
    num_rng = np.size(varPH , 1)
    varPH = varPH.filled(fill_value = np.nan)
    varSmPH = np.empty([num_azi , num_rng])
    varSmPH.fill(np.nan)
    varKD = np.empty([num_azi , num_rng])
    varKD.fill(np.nan)
    for cnt_azi in np.arange(0 , num_azi):
        for cnt_rng in np.arange(0 , num_rng - 1):
            if varPH[cnt_azi , cnt_rng + 1] - varPH[cnt_azi , cnt_rng] < -180:
                varPH[cnt_azi , cnt_rng + 1 : ] = varPH[cnt_azi , cnt_rng + 1 : ] + 360
        for cnt_rng in np.arange(0 , num_rng - (smNum - 1)):
            varSmPH[cnt_azi , cnt_rng + int((smNum - 1) / 2)] = np.nanmean(varPH[cnt_azi , cnt_rng : cnt_rng + (smNum - 1)])
    for cnt_rng in np.arange(0 , num_rng - 1):
        varKD[: , cnt_rng] = (varSmPH[: , cnt_rng + 1] - varSmPH[: , cnt_rng]) / (range[cnt_rng + 1] - range[cnt_rng]) / 2
    return varKD