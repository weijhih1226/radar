########################################
########### convert_grid.py ############
######## Author: Wei-Jhih Chen #########
######### Update: 2022/10/30 ###########
########################################

import numpy as np

def equivalent_earth_model(ele , alt , rngs):
    # ele: Angle of Elevation (Units: degree)
    # alt: Altitude of Station (Units: km)
    # rngs: Range (Units: km)
    R = 6371.25                             # Radius of The Earth (Units: km)
    K_E = 4 / 3
    ele = ele / 180 * np.pi     # Units: degree to radius
    hgtEEM = (rngs ** 2 + (K_E * R) ** 2 + 2 * rngs * K_E * R * np.sin(ele)) ** 0.5 - K_E * R + alt
    disEEM = K_E * R * np.arcsin(rngs * np.cos(ele) / (K_E * R + hgtEEM))
    return disEEM , hgtEEM

def equivalent_earth_model_by_elevations(eles , alt , rngs):
    disEEM = np.empty([len(eles) , len(rngs)])
    hgtEEM = np.empty([len(eles) , len(rngs)])
    for cnt_ele in range(len(eles)):
        disEEM[cnt_ele] , hgtEEM[cnt_ele] = equivalent_earth_model(eles[cnt_ele] , alt , rngs)
    return disEEM , hgtEEM

def polar_to_lonlat(azi , disEEM , hgtEEM , longitude , latitude):
    lon_deg2km = 102.8282                               # Degree to km (Units: km/deg.)
    lat_deg2km = 111.1361                               # Degree to km (Units: km/deg.)
    aziEEM = -(azi - 90) * np.pi / 180
    DisEEM , AziEEM = np.meshgrid(disEEM , aziEEM)      # Distance Meshgrid
    HgtEEM = np.meshgrid(hgtEEM , aziEEM)[0]            # Height Meshgrid
    xEEM = DisEEM * np.cos(AziEEM)
    yEEM = DisEEM * np.sin(AziEEM)
    LonEEM = longitude + xEEM / lon_deg2km
    LatEEM = latitude + yEEM / lat_deg2km
    return LonEEM , LatEEM

def convert_grid_ppi(var):
    num_azi = np.size(var , 0)
    num_rng = np.size(var , 1)
    nan_array = np.empty([num_azi + 1 , 1])
    nan_array.fill(np.nan)
    var = np.hstack((np.vstack((var , var[0 , :])) , nan_array))
    return var

def convert_grid_cs(var):
    num_ele = np.size(var , 0)
    num_rng = np.size(var , 1)
    nan_array1 = np.empty([1 , num_rng])
    nan_array2 = np.empty([num_ele + 1 , 1])
    nan_array1.fill(np.nan)
    nan_array2.fill(np.nan)
    var = np.hstack((np.vstack((var , nan_array1)) , nan_array2))
    return var