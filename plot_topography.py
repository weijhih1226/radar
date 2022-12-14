#!/home/C.cwj/anaconda3/envs/pyart_env/bin/python3
# -*- coding:utf-8 -*-

########################################
########## plot_topography.py ##########
######## Author: Wei-Jhih Chen #########
######### Update: 2022/12/14 ###########
########################################

import time
import cv2 as cv
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from pathlib import Path
from tools import timer
from convert_grid import polar_to_lonlat

INVALID_VALUE = -32767
# BOUND_WEST = 119.88190897
# BOUND_EAST = 122.244813473
# BOUND_SOUTH = 21.859099236
# BOUND_NORTH = 25.332818825
BOUND_WEST = 119.989962096
BOUND_EAST = 122.011229659
BOUND_SOUTH = 21.870002156
BOUND_NORTH = 25.324578876
MAP = {'west': BOUND_WEST , 'east': BOUND_EAST , 'south': BOUND_SOUTH , 'north': BOUND_NORTH}

COLOR_TOPO = '#777'

STATION_LONLAT = [121.75 , 24.75]
AZIMUTH = 225
DISTANCE = np.arange(0 , 50 , 0.5)

HOMEDIR = Path(r'C:\Users\wjchen\Documents\Research\Radar')
DEM_PATH = HOMEDIR/'Tools'/'dem'/'dem_500m.tif'

def read_tif(path):
    dem = cv.imread(str(path) , cv.IMREAD_UNCHANGED)
    dem = ma.masked_values(dem , INVALID_VALUE)
    dem = np.flip(dem , 0)
    dem = dem / 1000        # Units: km
    return dem

def gridmap_extent(dem , west , east , south , north):
    num_y , num_x = dem.shape
    lon = np.linspace(west , east , num_x + 1)
    lat = np.linspace(south , north , num_y + 1)
    Lon , Lat = np.meshgrid(lon , lat)
    return Lon , Lat

def map_extent(dem , west , east , south , north):
    num_y , num_x = dem.shape
    int_x = (east - west) / num_x
    int_y = (north - south) / num_y
    lon = np.arange(west + int_x / 2 , east , int_x)
    lat = np.arange(south + int_y / 2 , north , int_y)
    Lon , Lat = np.meshgrid(lon , lat)
    return Lon , Lat

def read_tif_gridmap(path , west , east , south , north):
    Dem = read_tif(path)
    Lon , Lat = gridmap_extent(Dem , west , east , south , north)
    return Lon , Lat , Dem

def read_tif_map(path , west , east , south , north):
    Dem = read_tif(path)
    Lon , Lat = map_extent(Dem , west , east , south , north)
    return Lon , Lat , Dem

def find_nearest_point(map_data , map_x , map_y , point_x , point_y):
    num_x = map_data.shape[1]
    idx = np.argmin(abs(map_x - point_x) + abs(map_y - point_y))
    idx_x = idx % num_x
    idx_y = idx // num_x
    return map_data[idx_y , idx_x]

def find_nearest_points(map_data , map_x , map_y , points_x , points_y):
    num_pnt = len(points_x)
    points_data = np.empty((num_pnt))
    for cnt_pnt in range(num_pnt):
        points_data[cnt_pnt] = find_nearest_point(map_data , map_x , map_y , points_x[cnt_pnt] , points_y[cnt_pnt])
    return points_data

def find_nearest_topo(path , west , east , south , north , slonlat , azi , Dis):
    LonG , LatG , DemG = read_tif_map(path , west , east , south , north)
    Lon , Lat , NULL = polar_to_lonlat(azi , Dis , None , slonlat[0] , slonlat[1])
    return find_nearest_points(DemG , LonG , LatG , Lon[0] , Lat[0])

def plot_nearest_topo(ax , path , west , east , south , north , slonlat , azi , Dis):
    Dem = find_nearest_topo(path , west , east , south , north , slonlat , azi , Dis)
    ax.fill_between(Dis , Dem , color = COLOR_TOPO)

def plot_tif_gridmap(ax , path , west , east , south , north):
    Lon , Lat , Dem = read_tif_gridmap(path , west , east , south , north)
    ax.pcolormesh(Lon , Lat , Dem , shading = 'flat')

@timer
def main():
    plt.close()
    fig , ax = plt.subplots(figsize = (12 , 10))
    # plot_tif_gridmap(ax , DEM_PATH , MAP['west'] , MAP['east'] , MAP['south'] , MAP['north'])
    plot_nearest_topo(ax , DEM_PATH , MAP['west'] , MAP['east'] , MAP['south'] , MAP['north'] , STATION_LONLAT , AZIMUTH , DISTANCE)
    plt.show()

if __name__ == '__main__':
    main()
