########################################
########### convert_grid.py ############
######## Author: Wei-Jhih Chen #########
######### Update: 2022/11/15 ###########
########################################

import numpy as np
from scipy.interpolate import griddata as gd

def equivalent_earth_model(ele , Rng , alt):
    ''' Calculate the distances and heights of bins in each radar beam with specified elevation

    --------------------
    Args:
        ele: float
            the elevation of the radar beam (Units: deg.)
        Rng: numpy.ndarray
            the ranges of the radar beam (Units: km)
        alt: float
            the altitude of the radar station (Units: km)
    
    --------------------
    Returns:
        Dis: numpy.ndarray
            the distance of the radar beam (Units: km; Shape: range)
        Hgt: numpy.ndarray
            the height of the radar beam (Units: km; Shape: range)
    '''
    R = 6371.25                 # Radius of The Earth (Units: km)
    K_E = 4 / 3
    ele = ele / 180 * np.pi     # Units: degree to radius
    Hgt = (Rng ** 2 + (K_E * R) ** 2 + 2 * Rng * K_E * R * np.sin(ele)) ** 0.5 - K_E * R + alt
    Dis = K_E * R * np.arcsin(Rng * np.cos(ele) / (K_E * R + Hgt))
    return Dis , Hgt

def equivalent_earth_model_by_elevations(Ele , Rng , alt):
    ''' Calculate the distances and heights of bins in multiple radar beams with different elevations

    --------------------
    Args:
        Ele: numpy.ndarray
            the elevations of multiple radar beams (Units: deg.)
        Rng: numpy.ndarray
            the ranges of one radar beam (Units: km)
        alt: float
            the altitude of the radar station (Units: km)
    
    --------------------
    Returns:
        DisG: numpy.ndarray
            the distance of multiple radar beams (Units: km; Shape: elevation * range)
        HgtG: numpy.ndarray
            the height of multiple radar beams (Units: km; Shape: elevation * range)
    '''
    DisG = np.empty((len(Ele) , len(Rng)))
    HgtG = np.empty((len(Ele) , len(Rng)))
    for cnt_ele in range(len(Ele)):
        DisG[cnt_ele] , HgtG[cnt_ele] = equivalent_earth_model(Ele[cnt_ele] , Rng , alt)
    return DisG , HgtG

def polar_to_lonlat(Azi , Dis , Hgt , slon , slat):
    ''' Convert the polar coordinates to the longitude-latitude coordinates

    --------------------
    Args:
        Azi: numpy.ndarray
            the azimuths (Units: deg.; Shape: azimuth)
        Dis: numpy.ndarray
            the distance of one-direction radar beam (Units: km; Shape: range)
        Hgt: numpy.ndarray
            the height of one-direction radar beam (Units: km; Shape: range)
        slon: float
            the longitude of the radar station
        slat: float
            the latitude of the radar station
    
    --------------------
    Returns:
        LonG: numpy.ndarray
            the longitude (Shape: azimuth * range)
        LatG: numpy.ndarray
            the latitude (Shape: azimuth * range)
        HgtG: numpy.ndarray
            the height (Shape: azimuth * range)
    '''
    LON_DEG2KM = 102.8282                   # Degree to km (Units: km/deg.)
    LAT_DEG2KM = 111.1361                   # Degree to km (Units: km/deg.)
    Azi = -(Azi - 90) * np.pi / 180
    DisG , AziG = np.meshgrid(Dis , Azi)    # Distance Meshgrid
    HgtG = np.meshgrid(Hgt , Azi)[0] if Hgt is not None else None   # Height Meshgrid
    LonG = slon + DisG * np.cos(AziG) / LON_DEG2KM
    LatG = slat + DisG * np.sin(AziG) / LAT_DEG2KM
    return LonG , LatG , HgtG

def lonlat_map(ele , Azi , Rng , stainfo):
    ''' Convert the polar coordinates of the specific elevation to the longitude-latitude coordinates
    --------------------
    Args:
        ele: float
            the elevation of the radar beam (Units: deg.)
        Azi: numpy.ndarray
            the azimuths (Units: deg.; Shape: azimuth)
        Rng: numpy.ndarray
            the ranges of the radar beam (Units: km)
        stainfo: dict
            the information of the radar station
            'lon': float
                the longitude
            'lat': float
                the loatitude
            'alt': float
                the altitude

    --------------------
    Returns:
        LonG: numpy.ndarray
            the longitude (Shape: azimuth * range)
        LatG: numpy.ndarray
            the latitude (Shape: azimuth * range)
        HgtG: numpy.ndarray
            the height (Shape: azimuth * range)
    '''
    Dis , Hgt = equivalent_earth_model(ele , Rng , stainfo['alt'])
    LonG , LatG , HgtG = polar_to_lonlat(Azi , Dis , Hgt , stainfo['lon'] , stainfo['lat'])
    return LonG , LatG , HgtG

def reorder_var_map(xMin , xMax , xInt , yMin , yMax , yInt):
    x = np.arange(xMin + xInt / 2 , xMax + xInt / 2 , xInt)
    y = np.arange(yMin + yInt / 2 , yMax + yInt / 2 , yInt)
    return np.meshgrid(x , y)

def reorder_grid_map(xMin , xMax , xInt , yMin , yMax , yInt):
    x = np.arange(xMin , xMax + xInt , xInt)
    y = np.arange(yMin , yMax + yInt , yInt)
    return np.meshgrid(x , y)

def radial_reorder_3D(axis , alt , Azi , StartIdx , EndIdx , Rng , Ele , Var):
    num_azi = len(Azi)
    X_reorder , Z_reorder = reorder_var_map(axis['xMin'] , axis['xMax'] , axis['xInt'] , axis['zMin'] , axis['zMax'] , axis['zInt'])
    VarG = np.empty((num_azi , ) + X_reorder.shape)
    for cnt_azi in range(num_azi):
        Dis , Hgt = equivalent_earth_model_by_elevations(Ele[StartIdx[cnt_azi] : EndIdx[cnt_azi] + 1] , Rng , alt)
        XY_pt = np.hstack((Dis.reshape((Dis.size , 1)) , Hgt.reshape((Hgt.size , 1))))
        VarInAzi = Var[StartIdx[cnt_azi] : EndIdx[cnt_azi] + 1]
        Var_pt = VarInAzi.reshape((VarInAzi.size , 1)).filled(np.nan)
        VarG[cnt_azi] = gd(XY_pt , Var_pt , (X_reorder , Z_reorder) , method = 'linear' , fill_value = np.nan)[: , : , 0]
    return VarG

def radial_reorder_CV(axis , alt , StartIdx , EndIdx , Rng , Ele , Var , ismax = True):
    X_reorder , Z_reorder = reorder_var_map(axis['xMin'] , axis['xMax'] , axis['xInt'] , axis['zMin'] , axis['zMax'] , axis['zInt'])
    VarG = np.empty(X_reorder.shape)
    Dis , Hgt = equivalent_earth_model_by_elevations(Ele[StartIdx : EndIdx + 1] , Rng , alt)
    XY_pt = np.hstack((Dis.reshape((Dis.size , 1)) , Hgt.reshape((Hgt.size , 1))))
    VarInAzi = Var[StartIdx : EndIdx + 1]
    Var_pt = VarInAzi.reshape((VarInAzi.size , 1)).filled(np.nan)
    VarG = gd(XY_pt , Var_pt , (X_reorder , Z_reorder) , method = 'linear' , fill_value = np.nan)[: , : , 0]
    return np.nanmax(VarG , axis = 0) if ismax is True else np.nanpercentile(VarG , 25 , axis = 0)

def radial_CV_grid(axis , Azi , slon , slat):
    Dis = np.arange(axis['xMin'] , axis['xMax'] + axis['xInt'] , axis['xInt'])
    Azi = np.append(np.append(Azi[0] - (Azi[1] - Azi[0]) / 2 , Azi[:-1] + (Azi[1:] - Azi[:-1]) / 2) , Azi[-1] - (Azi[-1] - Azi[-2]) / 2)
    return polar_to_lonlat(Azi , Dis , None , slon , slat)

def radial_CV(axis , salt , Azi , StartIdx , EndIdx , Rng , Ele , Var , ismax = True):
    # Var = radial_reorder_3D(axis , salt , Azi , StartIdx , EndIdx , Rng , Ele , Var)
    # return np.nanmax(Var , axis = 1) if ismax is True else np.nanpercentile(Var , 25 , axis = 1)
    Rad = np.arange(axis['xMin'] + axis['xInt'] / 2 , axis['xMax'] + axis['xInt'] / 2 , axis['xInt'])
    num_rad = len(Rad)
    num_azi = len(Azi)
    VarG = np.empty((num_azi , num_rad))
    for cnt_azi in range(num_azi):
        VarG[cnt_azi] = radial_reorder_CV(axis , salt , StartIdx[cnt_azi] , EndIdx[cnt_azi] , Rng , Ele , Var , ismax = ismax)
    return VarG

def convert_grid_ppi(var):
    nan_array = np.empty((var.shape[0] + 1 , 1))
    nan_array.fill(np.nan)
    var = np.hstack((np.vstack((var , var[0 , :])) , nan_array))
    return var

def convert_grid_cs(var):
    nan_array1 = np.empty((1 , var.shape[1]))
    nan_array2 = np.empty((var.shape[0] + 1 , 1))
    nan_array1.fill(np.nan)
    nan_array2.fill(np.nan)
    var = np.hstack((np.vstack((var , nan_array1)) , nan_array2))
    return var