########################################
###### attenuation_correction.py #######
######## Author: Wei-Jhih Chen #########
######### Update: 2022/07/19 ###########
########################################

import numpy as np
from numpy.ma import masked_array as mama

def attenuation_correction(varDZ , varZD , varKD , delta_r , b1 , b2 , d1 , d2):
    num_azi = varDZ.shape[0]
    num_rng = varDZ.shape[1]
    Ah = np.zeros([num_azi , num_rng])
    Adp = np.zeros([num_azi , num_rng])
    varDZac = mama(np.zeros([num_azi , num_rng]) , np.zeros([num_azi , num_rng]))
    varZDac = mama(np.zeros([num_azi , num_rng]) , np.zeros([num_azi , num_rng]))
    varKD = mama(varKD , varKD < 0)
    for cnt_rng in range(num_rng):
        Ah[: , cnt_rng] = b1 * varKD[: , cnt_rng] ** b2
        varDZac[: , cnt_rng] = varDZ[: , cnt_rng] + 2 * np.sum(Ah[: , : cnt_rng] , axis = 1) * delta_r / 1000
        Adp[: , cnt_rng] = d1 * varKD[: , cnt_rng] ** d2
        varZDac[: , cnt_rng] = varZD[: , cnt_rng] + 2 * np.sum(Adp[: , : cnt_rng] , axis = 1) * delta_r / 1000
    return varDZac , varZDac

def attenuation_correction_C(varDZ , varZD , varKD , delta_r):
    # By Bringi et al. 1990 (B90)
    b1 = 0.08
    b2 = 1
    d1 = b1 * 0.1125
    d2 = 1
    varDZac , varZDac = attenuation_correction(varDZ , varZD , varKD , delta_r , b1 , b2 , d1 , d2)

def attenuation_correction_X(varDZ , varZD , varKD , delta_r = 50):
    # By FURUNO WR2100
    b1 = 0.233
    b2 = 1.02
    d1 = 0.0298
    d2 = 1.293
    varDZac , varZDac = attenuation_correction(varDZ , varZD , varKD , delta_r , b1 , b2 , d1 , d2)
    return varDZac , varZDac