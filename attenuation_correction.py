########################################
###### attenuation_correction.py #######
######## Author: Wei-Jhih Chen #########
######### Update: 2022/11/08 ###########
########################################

''' Calculate corrected values by the attenuation effect
--------------------
Correction method:
--------------------
Ah(i) = b1 * Kdp(i) ** b2
Zh_corr(i) = Zh(i) + 2 * sum(Ah(0) + Ah(1) + ... + Ah(i)) * delta_r
Adp(i) = d1 * Kdp(i) ** d2
Zdr_corr(i) = Zdr(i) + 2 * sum(Adp(0) + Adp(1) + ... + Adp(i)) * delta_r

--------------------
Functions:
--------------------
- attenuation_correction(Zh , Zdr , Kdp , dr , b1 , b2 , d1 , d2)
- attenuation_correction_C(Zh , Zdr , Kdp , dr) (for C band)
- attenuation_correction_X(Zh , Zdr , Kdp , dr) (for X band)
'''

import numpy as np
from numpy.ma import masked_array as mama

def attenuation_correction(Zh , Zdr , Kdp , delta_r , b1 , b2 , d1 , d2):
    ''' Attenuation correction by customized parameters: b1 , b2 , d1 , d2

    You can input parameters by yourself, or use attenuation_correction_C or
    attenuation_correction_X function to apply default parameters in each band, 
    and then you would get new values of Zh and Zdr after attenuation correction.

    Please refer to the correction method in the docstring of the module. 

    --------------------
    Args:
        Zh: numpy.ndarray
            (Shape: azimuth * range)
        Zdr: numpy.ndarray
            (Shape: azimuth * range)
        Kdp: numpy.ndarray
            (Shape: azimuth * range)
        delta_r: float
            resolution of range direction (Units: km)
        b1 , b2 , d1 , d2: float
            parameters of correction method (diff from band to band)

    --------------------
    Returns:
        Zh: numpy.ndarray (After correction)
        Zdr: numpy.ndarray (After correction)
    '''
    num_azi , num_rng = Zh.shape
    Ah = np.zeros(num_azi)
    Adp = np.zeros(num_azi)
    Kdp = mama(Kdp , Kdp < 0)
    for cnt_rng in range(num_rng):
        Ah += b1 * Kdp[: , cnt_rng] ** b2
        Adp += d1 * Kdp[: , cnt_rng] ** d2
        Zh[: , cnt_rng] += 2 * Ah * delta_r
        Zdr[: , cnt_rng] += 2 * Adp * delta_r
    return Zh , Zdr

def attenuation_correction_C(Zh , Zdr , Kdp , delta_r):
    ''' Attenuation correction of C band refer to Bringi et al. 1990 (B90).

    b1: 0.08
    b2: 1
    d1: b1 * 0.1125
    d2: 1

    --------------------
    Args:
        Zh: numpy.ndarray
        Zdr: numpy.ndarray
        Kdp: numpy.ndarray
        delta_r: float

        Please refer to the function "attenuation_correction".

    --------------------
    Returns:
        Zh: numpy.ndarray (After correction)
        Zdr: numpy.ndarray (After correction)
    '''
    b1 = 0.08
    b2 = 1
    d1 = 0.009
    d2 = 1
    return attenuation_correction(Zh , Zdr , Kdp , delta_r , b1 , b2 , d1 , d2)


def attenuation_correction_X(Zh , Zdr , Kdp , delta_r):
    ''' Attenuation correction of X band refer to FURUNO WR2100.

    b1: 0.233
    b2: 1.02
    d1: 0.0298
    d2: 1.293

    --------------------
    Args:
        Zh: numpy.ndarray
        Zdr: numpy.ndarray
        Kdp: numpy.ndarray
        delta_r: float

        Please refer to the function "attenuation_correction".

    --------------------
    Returns:
        Zh: numpy.ndarray (After correction)
        Zdr: numpy.ndarray (After correction)
    '''
    b1 = 0.233
    b2 = 1.02
    d1 = 0.0298
    d2 = 1.293
    return attenuation_correction(Zh , Zdr , Kdp , delta_r , b1 , b2 , d1 , d2)