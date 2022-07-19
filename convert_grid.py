########################################
########### convert_grid.py ############
######## Author: Wei-Jhih Chen #########
######### Update: 2022/07/19 ###########
########################################

import numpy as np

def equivalent_earth_model(elevation , altitude , range):
    # elevation: Angle of Elevation (Units: degree)
    # altitude: Altitude of Station (Units: km)
    # range: Range (Units: km)
    a = 6371.25                             # Units: km
    k_e = 4 / 3
    theta_e_degree = elevation              # Units: degree
    theta_e = theta_e_degree / 180 * np.pi  # Units: radius
    hgtEEM = (range ** 2 + (k_e * a) ** 2 + 2 * range * k_e * a * np.sin(theta_e)) ** 0.5 - k_e * a + altitude
    disEEM = k_e * a * np.arcsin(range * np.cos(theta_e) / (k_e * a + hgtEEM))
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