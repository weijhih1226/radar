########################################
######## plot_nexrad_levelII.py ########
######## Author: Wei-Jhih Chen #########
######### Update: 2022/06/09 ###########
########################################

import os
import pyart as pa
import numpy as np
import datetime as dt
import cartopy.crs as ccrs
import netCDF4 as nc
import matplotlib.pyplot as plt
import cfg.color as cfgc
from scipy import io
from datetime import datetime as dtdt
from numpy.ma import masked_array as mama
from cartopy.io.shapereader import Reader as shprd
from cartopy.feature import ShapelyFeature as shpft
from matplotlib.colors import ListedColormap , BoundaryNorm
from scipy.interpolate import griddata as gd

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

def equivalent_earth_model(elevation , altitude , range):
    # elevation: Angle of Elevation
    # altitude: Altitude of Station
    # range: Range
    a = 6371.25                                         # Units: km
    k_e = 4 / 3
    theta_e_degree = elevation                          # Units: degree
    theta_e = theta_e_degree / 180 * np.pi              # Units: radius
    hgtEEM = (range ** 2 + (k_e * a) ** 2 + 2 * range * k_e * a * np.sin(theta_e)) ** 0.5 - k_e * a + altitude
    disEEM = k_e * a * np.arcsin(range * np.cos(theta_e) / (k_e * a + hgtEEM))
    return disEEM , hgtEEM

def convert_coordinates(azi , disEEM , hgtEEM , longitude , latitude):
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

def RH_filter(varRH , var , flMin , flMax):
    var = mama(var , (varRH < flMin) | (varRH > flMax))
    return var

def SW_filter(varSW , var , flMax):
    var = mama(var , varSW > flMax)
    return var

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

def plot_ppi(axis , LonEEM , LatEEM , var , varInfo , staInfo , eleFix , datetimeStrLST , shpPath , matPath , outPath):
    ########## Grid ##########
    X = np.arange(axis['xMin'] * 10 , axis['xMax'] * 10 + axis['xInt'] * 10 , axis['xInt'] * 10) / 10
    Y = np.arange(axis['yMin'] * 10 , axis['yMax'] * 10 + axis['yInt'] * 10 , axis['yInt'] * 10) / 10
    XStr = []
    YStr = []
    for cnt_X in np.arange(0 , len(X)):
        XStr = np.append(XStr , f'{X[cnt_X]}$^o$E')
    for cnt_Y in np.arange(0 , len(Y)):
        YStr = np.append(YStr , f'{Y[cnt_Y]}$^o$E')
    ########## Color ##########
    colors , levels , ticks , tickLabels , cmin , cmax = cfgc.colors(varInfo['name'] , 'S')
    ########## Shape Files ##########
    shp = shpft(shprd(shpPath).geometries() , ccrs.PlateCarree() , 
                facecolor = (1 , 1 , 1 , 0) , edgecolor = (0 , 0 , 0 , 1) , linewidth = 1 , zorder = 10)
    ########## Terrain ##########
    terrain = io.loadmat(matPath)
    ########## Plot ##########
    plt.close()
    fig , ax = plt.subplots(figsize = [12 , 10] , subplot_kw = {'projection' : ccrs.PlateCarree()})
    # axis['xMin'] , axis['yMax'] + axis['yInt'] / 10 + axis['yInt'] / 4
    ax.text(0.125 , 0.905 , staInfo['name'] , fontsize = 18 , ha = 'left' , color = '#666666' , transform = fig.transFigure)
    ax.text(0.125 , 0.875 , varInfo['plotname'] , fontsize = 20 , ha = 'left' , transform = fig.transFigure)
    ax.text(0.744 , 0.905 , f'Elev. {eleFix:.2f}$^o$ ' + staInfo['scn'].upper() , fontsize = 18 , ha = 'right' , transform = fig.transFigure)
    ax.text(0.744 , 0.875 , datetimeStrLST , fontsize = 20 , ha = 'right' , transform = fig.transFigure)
    ax.set_extent([axis['xMin'] , axis['xMax'] , axis['yMin'] , axis['yMax']])
    ax.gridlines(xlocs = X , ylocs = Y , color = '#bbbbbb' , linewidth = 0.5 , alpha = 0.5 , draw_labels = False)
    ax.add_feature(shp)
    ax.contour(terrain['blon'] , terrain['blat'] , terrain['QPEterrain'] , levels = [500 , 1500 , 3000] , colors = '#C0C0C0' , linewidths = [0.5 , 1 , 1.5])
    ax.scatter(staInfo['lon'] , staInfo['lat'] , s = 50 , c = 'k' , marker = '^')
    plt.xticks(X , XStr , size = 10)
    plt.yticks(Y , YStr , size = 10)
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(levels , cmap.N)
    var[var > cmax] = np.nan
    var[var < cmin] = np.nan
    PC = ax.pcolormesh(LonEEM , LatEEM , var , shading = 'flat' , cmap = cmap , norm = norm , alpha = 1)
    PC.set_clim(cmin , cmax)
    cbar = plt.colorbar(PC , orientation = 'vertical' , ticks = ticks)
    cbar.ax.set_yticklabels(tickLabels)
    cbar.ax.tick_params(labelsize = 12)
    cbar.set_label(varInfo['units'] , size = 12)
    fig.savefig(outPath , dpi = 200)

def plot_cs(axis , XEEM , ZEEM , var , varInfo , staInfo , aziMean , datetimeStrLST , outPath):
    ########## Grid ##########
    xTick = np.arange(axis['xMin'] * 10 , axis['xMax'] * 10 + axis['xInt'] * 10 , axis['xInt'] * 10) / 10
    zTick = np.arange(axis['zMin'] * 10 , axis['zMax'] * 10 + axis['zInt'] * 10 , axis['zInt'] * 10) / 10
    ########## Color ##########
    colors , levels , ticks , tickLabels , cmin , cmax = cfgc.colors(varInfo['name'] , 'S')
    ########## Plot ##########
    plt.close()
    fig , ax = plt.subplots(figsize = [12 , 10])
    ax.text(0.125 , 0.920 , staInfo['name'] , fontsize = 18 , ha = 'left' , color = '#666666' , transform = fig.transFigure)
    ax.text(0.125 , 0.890 , varInfo['plotname'] , fontsize = 20 , ha = 'left' , transform = fig.transFigure)
    ax.text(0.744 , 0.920 , f'Azi. {aziMean:.2f}$^o$ CS' , fontsize = 18 , ha = 'right' , transform = fig.transFigure)
    ax.text(0.744 , 0.890 , datetimeStrLST , fontsize = 20 , ha = 'right' , transform = fig.transFigure)
    ax.axis([axis['xMin'] , axis['xMax'] , axis['zMin'] , axis['zMax']])
    ax.scatter(0 , staInfo['alt'] , s = 50 , c = 'k' , marker = '^')
    if aziMean >= 180:
        ax.invert_xaxis()
    plt.xticks(xTick , size = 10)
    plt.yticks(zTick , size = 10)
    plt.xlabel('Distance from Radar (km)')
    plt.ylabel('Altitude (km)')
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(levels , cmap.N)
    var[var > cmax] = np.nan
    var[var < cmin] = np.nan
    PC = ax.pcolormesh(XEEM , ZEEM , var , shading = 'flat' , cmap = cmap , norm = norm , alpha = 1)
    PC.set_clim(cmin , cmax)
    ax.grid(visible = True , color = '#bbbbbb' , linewidth = .5 , alpha = .5 , zorder = 0)
    cbar = plt.colorbar(PC , orientation = 'vertical' , ticks = ticks , extend = 'both')
    cbar.ax.set_yticklabels(tickLabels)
    cbar.ax.tick_params(labelsize = 12)
    cbar.set_label(varInfo['units'] , size = 12)
    fig.savefig(outPath , dpi = 200)
    print(f"{outPath} - Done!")

def plot_cs_reorder(axis , XG , ZG , varXZ , varInfo , staInfo , azi , datetimeStrLST , outPath):
    ########## Grid ##########
    xTick = np.arange(axis['xMin'] * 10 , axis['xMax'] * 10 + axis['xInt'] * 10 , axis['xInt'] * 10) / 10
    zTick = np.arange(axis['zMin'] * 10 , axis['zMax'] * 10 + axis['zInt'] * 10 , axis['zInt'] * 10) / 10
    ########## Color ##########
    colors , levels , ticks , tickLabels , cmin , cmax = cfgc.colors(varInfo['name'] , 'S')
    ########## Plot ##########
    plt.close()
    fig , ax = plt.subplots(figsize = [12 , 10])
    ax.text(0.125 , 0.920 , staInfo['name'] , fontsize = 18 , ha = 'left' , color = '#666666' , transform = fig.transFigure)
    ax.text(0.125 , 0.890 , varInfo['plotname'] , fontsize = 20 , ha = 'left' , transform = fig.transFigure)
    ax.text(0.744 , 0.920 , f'Azi. {azi:.2f}$^o$ CS' , fontsize = 18 , ha = 'right' , transform = fig.transFigure)
    ax.text(0.744 , 0.890 , datetimeStrLST , fontsize = 20 , ha = 'right' , transform = fig.transFigure)
    ax.axis([axis['xMin'] , axis['xMax'] , axis['zMin'] , axis['zMax']])
    ax.scatter(0 , staInfo['alt'] , s = 250 , c = 'k' , marker = '^')
    if azi >= 180:
        ax.invert_xaxis()
    plt.xticks(xTick , size = 10)
    plt.yticks(zTick , size = 10)
    plt.xlabel('Distance from Radar (km)')
    plt.ylabel('altitudeitude (km)')
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(levels , cmap.N)
    varXZ[varXZ > cmax] = np.nan
    varXZ[varXZ < cmin] = np.nan
    PC = ax.pcolormesh(XG , ZG , varXZ , shading = 'flat' , cmap = cmap , norm = norm , alpha = 1)
    PC.set_clim(cmin , cmax)
    ax.grid(visible = True , color = '#bbbbbb' , linewidth = .5 , alpha = .5 , zorder = 0)
    cbar = plt.colorbar(PC , orientation = 'vertical' , ticks = ticks , extend = 'both')
    cbar.ax.set_yticklabels(tickLabels)
    cbar.ax.tick_params(labelsize = 12)
    cbar.set_label(varInfo['units'] , size = 12)
    fig.savefig(outPath , dpi = 200)
    varXZ.tofile(f'{outPath[:-4]}.dat')
    print(f"{outPath} - Done!")

########## Case Setting ##########
case_date = '20200716'
# case_time = '043300'
# case_time = '043900'    # ZDR Best
# case_time = '044500'    # KDP Best
# case_time = '045100'
# case_time = '045700'
case_time = '050300'    # DBZ Best
# case_time = '050900'
# case_time = '051500'

plot_type = 'CS'
sel_var = []            # All: []
sel_aziCS = [200 , 201 , 202 , 203 , 204 , 205 , 206 , 207 , 208 , 209 , 210]          # NP: []

########## Parameters Setting ##########
station_name = 'RCWF'
scan_type = 'PPI'
flRH_min = 0.7
flRH_max = 1.05
flSW_max = 4
flZD_num = 5
flPH_num = 5
smPH_num = 27
_FillValue = -9999.

longitude_NTU = 121.81599
latitude_NTU = 24.76395
altitude_NTU = 0.01         # Units: km

########## Path Setting ##########
homeDir = '/home/C.cwj/Radar/'
inDir = f'{homeDir}cases/RAW-{station_name}/{case_date}/'
inPath = f'{inDir}/{case_date}{case_time}.raw'
shpPath = f'{homeDir}Tools/shp/taiwan_county/COUNTY_MOI_1090820.shp'     # TWNcountyTWD97
matPath = f'{homeDir}Tools/mat/QPESUMS_terrain.mat'                      # TWNterrainTWD97

########## Create Directory ##########
outDir = f'{homeDir}pic/{station_name}/{case_date}/'
outDir2 = f'{homeDir}pic/{station_name}/{case_date}/Reorder/'
if not(os.path.isdir(f'{outDir}{plot_type}/')):
    os.makedirs(f'{outDir}{plot_type}/')
    print(f'Create Directory: {outDir}{plot_type}/')
if not(os.path.isdir(f'{outDir2}{plot_type}/')):
    os.makedirs(f'{outDir2}{plot_type}/')
    print(f'Create Directory: {outDir2}{plot_type}/')

########## Read Variables ##########
instrument_name = 'NEXRAD WSR-88D'  # Instrument Name
data_type = 'Level-II'
radar = pa.io.read_nexrad_archive(inPath)

# Time
datetime = dtdt.strptime(radar.time['units'][14:34] , '%Y-%m-%dT%H:%M:%SZ')
dateStrLST = dtdt.strftime(datetime + dt.timedelta(hours = 8) , '%Y%m%d')
timeStrLST = dtdt.strftime(datetime + dt.timedelta(hours = 8) , '%H%M%S')
datetimeStrLST = dtdt.strftime(datetime + dt.timedelta(hours = 8) , '%Y/%m/%d %H:%M:%S LST')

# Station
longitude = radar.longitude['data'][0]
latitude = radar.latitude['data'][0]
altitude = radar.altitude['data'][0] / 1000     # Units: km
# Find Azimuth and Distance across the Other Station
azi_NTU , dis_NTU = find_nearest_azimuth(longitude , latitude , longitude_NTU , latitude_NTU)   # 172.67680892202117 , 34.635276253600125
sel_aziCS = np.append(sel_aziCS , azi_NTU)
# dis_NTU = 34.75

# Instrument Parameters
range = radar.range['data'] / 1000              # Units: km
Azimuth = radar.azimuth['data']
Elevation = radar.elevation['data']
rangeG = np.append(range - np.append(range[1] - range[0] , range[1:] - range[:-1]) / 2 , range[-1] + (range[-1] - range[-2]) / 2)

swp_number = radar.sweep_number['data']
swp_angle = radar.fixed_angle['data']
bm_width = 1                                    # Units: degree

idx_swpStart = radar.sweep_start_ray_index['data']
idx_swpEnd = radar.sweep_end_ray_index['data']

# Numbers
num_rng = range.shape[0]
num_rngG = num_rng + 1
num_eleA = swp_number[-1] + 1            # Number: 18
num_Azi = idx_swpEnd - idx_swpStart + 1
num_sel_aziCS = np.size(sel_aziCS , 0)

eleFix = np.empty([num_eleA])
for cnt_eleA in np.arange(0 , num_eleA):
    eleFix[cnt_eleA] = np.mean(Elevation[idx_swpStart[cnt_eleA] : idx_swpEnd[cnt_eleA] + 1])

scan_type = radar.scan_type
vcp_pattern = radar.metadata['vcp_pattern']
unambiguous_range = radar.instrument_parameters['unambiguous_range']['data'] / 1000     # Units: km
nyquist_velocity = radar.instrument_parameters['nyquist_velocity']['data']              # Units: m/s

########## Read Fields ##########
var_inName = np.array(['reflectivity' , 'differential_reflectivity' , 'differential_phase' , 'cross_correlation_ratio' , 'velocity' , 'spectrum_width'])
var_name = np.array(['DZ' , 'ZD' , 'PH' , 'RH' , 'VR' , 'SW' , 'KD'])
var_plot = np.array(['Z$_{HH}$' , 'Z$_{DR}$' , '$\phi$$_{DP}$' , r'$\rho$$_{HV}$' , 'V$_R$' , 'SW' , 'K$_{DP}$'])
var_units = np.array(['dBZ' , 'dB' , 'Deg.' , '' , 'm s$^{-1}$' , 'm s$^{-1}$' , 'Deg. km$^{-1}$'])
num_inVar = len(var_inName)
num_var = len(var_name)

for cnt_inVar in np.arange(0 , num_inVar):
    exec(f"var{var_name[cnt_inVar]}_all = radar.fields['{var_inName[cnt_inVar]}']['data']")

########## CS Initiation ##########
varDZ_cs_all = np.empty([num_sel_aziCS , num_eleA , num_rng])
varZD_cs_all = np.empty([num_sel_aziCS , num_eleA , num_rng])
varPH_cs_all = np.empty([num_sel_aziCS , num_eleA , num_rng])
varKD_cs_all = np.empty([num_sel_aziCS , num_eleA , num_rng])
varRH_cs_all = np.empty([num_sel_aziCS , num_eleA , num_rng])
varVR_cs_all = np.empty([num_sel_aziCS , num_eleA , num_rng])
varSW_cs_all = np.empty([num_sel_aziCS , num_eleA , num_rng])
sel_aziCS_val_all = np.empty([num_sel_aziCS , num_eleA])
varDZ_cs_all.fill(np.nan)
varZD_cs_all.fill(np.nan)
varPH_cs_all.fill(np.nan)
varKD_cs_all.fill(np.nan)
varRH_cs_all.fill(np.nan)
varVR_cs_all.fill(np.nan)
varSW_cs_all.fill(np.nan)
sel_aziCS_val_all.fill(np.nan)

########## Plot PPI ##########
# for cnt_ele in np.arange(num_ele - 1 , num_ele):
for cnt_eleA in np.arange(0 , num_eleA):
    ########## Select Variables (VCP215) ##########
    cnt_swp = radar.sweep_number['data'][cnt_eleA]
    if cnt_swp == 0 or cnt_swp == 2 or cnt_swp == 4:
        nameVars_in = ['DZ' , 'ZD' , 'PH' , 'RH']
        nameVars_out = ['DZ' , 'ZD' , 'PH' , 'KD' , 'RH']
        plotVars_out = ['Z$_{HH}$' , 'Z$_{DR}$' , '$\phi$$_{DP}$' , 'K$_{DP}$' , r'$\rho$$_{HV}$']
        unitVars_out = ['dBZ' , 'dB' , 'Deg.' , 'Deg. km$^{-1}$' , '']
        nameVars_create = ['RH']
        nameVars_flRH = ['DZ' , 'ZD' , 'PH']
        nameVars_flSW = []
        nameVars_flZD = ['DZ' , 'PH']
        nameVars_flKD = ['PH']
    elif cnt_swp == 1 or cnt_swp == 3 or cnt_swp == 5:
        nameVars_in = ['VR' , 'SW']
        nameVars_out = ['VR' , 'SW']
        plotVars_out = ['V$_R$' , 'SW']
        unitVars_out = ['m s$^{-1}$' , 'm s$^{-1}$']
        nameVars_create = ['VR' , 'SW']
        nameVars_flRH = []
        nameVars_flSW = ['VR']
        nameVars_flZD = []
        nameVars_flKD = []
    else:
        nameVars_in = ['DZ' , 'ZD' , 'PH' , 'RH' , 'VR' , 'SW']
        nameVars_out = ['DZ' , 'ZD' , 'PH' , 'KD' , 'RH' , 'VR' , 'SW']
        plotVars_out = ['Z$_{HH}$' , 'Z$_{DR}$' , '$\phi$$_{DP}$' , 'K$_{DP}$' , r'$\rho$$_{HV}$' , 'V$_R$' , 'SW']
        unitVars_out = ['dBZ' , 'dB' , 'Deg.' , 'Deg. km$^{-1}$' , '' , 'm s$^{-1}$' , 'm s$^{-1}$']
        nameVars_create = ['RH' , 'VR' , 'SW']
        nameVars_flRH = ['DZ' , 'ZD' , 'PH']
        nameVars_flSW = ['DZ' , 'ZD' , 'PH' , 'VR']
        nameVars_flZD = ['DZ' , 'PH']
        nameVars_flKD = ['PH']
    num_varOut = len(nameVars_out)

    if not(sel_var):
        cnt_varOuts = np.arange(0 , num_varOut)
    else:
        cnt_varOuts = np.array([] , dtype = 'i')
        for cnt_selVar in np.arange(0 , len(sel_var)):
            try:
                cnt_varOuts = np.append(cnt_varOuts , nameVars_out.index(sel_var[cnt_selVar]))
            except ValueError:
                cnt_varOuts = cnt_varOuts

    ########## Convert Coordinates ##########
    num_azi = num_Azi[cnt_eleA]
    azimuth = Azimuth[idx_swpStart[cnt_eleA] : idx_swpEnd[cnt_eleA] + 1]
    azimuthG = np.append(azimuth , azimuth[0]) - (360 / num_azi / 2) % 360
    disEEM_G , hgtEEM_G = equivalent_earth_model(eleFix[cnt_eleA] , altitude , rangeG)
    LonEEM_G , LatEEM_G = convert_coordinates(azimuthG , disEEM_G , hgtEEM_G , longitude , latitude)

    ########## Filters ##########
    for nameVar_in in nameVars_in:
        exec(f'var{nameVar_in}_raw = var{nameVar_in}_all[idx_swpStart[cnt_eleA] : idx_swpEnd[cnt_eleA] + 1 , :]')
        exec(f'var{nameVar_in}_raw = mama(var{nameVar_in}_raw , var{nameVar_in}_raw == _FillValue)')
    if nameVars_create:
        for nameVar_create in nameVars_create:
            exec(f'var{nameVar_create} = var{nameVar_create}_raw')
    if nameVars_flRH:
        for nameVar_flRH in nameVars_flRH:
            exec(f'var{nameVar_flRH} = RH_filter(varRH_raw , var{nameVar_flRH}_raw , flRH_min , flRH_max)')
    if nameVars_flSW:
        for nameVar_flSW in nameVars_flSW:
            exec(f'var{nameVar_flSW} = SW_filter(varSW_raw , var{nameVar_flSW} , flSW_max)')
    if nameVars_flZD:
        for nameVar_flZD in nameVars_flZD:
            exec(f'varZD , var{nameVar_flZD} = ZD_filter(varZD_raw , var{nameVar_flZD} , flZD_num)')
    if nameVars_flKD:
        for nameVar_flKD in nameVars_flKD:
            exec(f'varKD = KD_filter(var{nameVar_flKD} , range , sm{nameVar_flKD}_num)')

    ########## Fill Empty ##########
    if cnt_swp == 0 or cnt_swp == 2 or cnt_swp == 4:
        varVR = np.empty([num_azi , num_rng])
        varSW = np.empty([num_azi , num_rng])
        varVR.fill(np.nan)
        varSW.fill(np.nan)
    elif cnt_swp == 1 or cnt_swp == 3 or cnt_swp == 5:
        varDZ = np.empty([num_azi , num_rng])
        varZD = np.empty([num_azi , num_rng])
        varPH = np.empty([num_azi , num_rng])
        varKD = np.empty([num_azi , num_rng])
        varRH = np.empty([num_azi , num_rng])
        varDZ.fill(np.nan)
        varZD.fill(np.nan)
        varPH.fill(np.nan)
        varKD.fill(np.nan)
        varRH.fill(np.nan)

    ########## Cross Section ##########
    for cnt_sel_aziCS in np.arange(0 , num_sel_aziCS):
        (varDZ_cs_all[cnt_sel_aziCS , cnt_eleA , :] , varZD_cs_all[cnt_sel_aziCS , cnt_eleA , :] , 
        varPH_cs_all[cnt_sel_aziCS , cnt_eleA , :] , varKD_cs_all[cnt_sel_aziCS , cnt_eleA , :] , 
        varRH_cs_all[cnt_sel_aziCS , cnt_eleA , :] , varVR_cs_all[cnt_sel_aziCS , cnt_eleA , :] , 
        varSW_cs_all[cnt_sel_aziCS , cnt_eleA , :] , 
        sel_aziCS_val_all[cnt_sel_aziCS , cnt_eleA]) = cross_section(varDZ , varZD , varPH , varKD , varRH , varVR , varSW , azimuth , sel_aziCS[cnt_sel_aziCS])

    # varDZ , varZD , varPH , varVR = RH_filter(varRH_raw , varDZ_raw , varZD_raw , varPH_raw , varVR_raw , flRH_min , flRH_max)
    # varDZ , varZD , varPH , varVR = SW_filter(varSW_raw , varDZ , varZD , varPH , varVR , flSW_max)
    # varZD , varDZ , varPH , varVR = ZD_filter(varZD_raw , varDZ , varPH , varVR , flZD_num)
    # varKD = KD_filter(varPH , range , smPH_num)
    # varRH = varRH_raw
    # varSW = varSW_raw
    # # varDaPH , varCrPH , varPH0 = PH_dealiasing(varPH , 0)
    # # varCrPH , varSmCrPH = PhiDP_filter(varCrPH , flPH_num , smPH_num)
    # # varPhKD = PhiDPtoKDP(varSmCrPH , range , np.nan)

    # for cnt_varOut in cnt_varOuts:
    #     ########## Convert PPI Grid ##########
    #     exec(f'var{nameVars_out[cnt_varOut]} = convert_grid_ppi(var{nameVars_out[cnt_varOut]})')

    #     ########## Path & Info Setting ##########
    #     outPath = f'{outDir}{nameVars_out[cnt_varOut]}_{dateStrLST}_{timeStrLST}_{cnt_swp:02.0f}.png'
    #     staInfo = {'name' : station_name , 'lon' : longitude , 'lat' : latitude , 'alt' : altitude , 'scn' : scan_type}
    #     varInfo = {'name' : nameVars_out[cnt_varOut] , 'plotname' : plotVars_out[cnt_varOut] , 'units' : unitVars_out[cnt_varOut]}

    #     ########## Plot PPI ##########
    #     axis_ppi = {'xMin' : 121.5 , 'xMax' : 122.1 , 'xInt' : 0.1 , 'yMin' : 24.5 , 'yMax' : 25.1 , 'yInt' : 0.1}
    #     plot_ppi(axis_ppi , LonEEM_G , LatEEM_G , varInfo , staInfo , eleFix[cnt_ele] , datetimeStrLST , shpPath , matPath , outPath)

    print(f'{datetimeStrLST} - {cnt_swp:02.0f} Elev. {eleFix[cnt_eleA]:.2f}^o - Finish!')

########## Cross Section ##########


for cnt_sel_aziCS in np.arange(0 , num_sel_aziCS):
    # Cartesian Grid
    if cnt_sel_aziCS == num_sel_aziCS - 1:
        xMin = dis_NTU ;    xMax = dis_NTU + 40 ;   xInt = 0.25
    else:
        xMin = 20 ;         xMax = 70 ;             xInt = 0.25
    zMin = 0 ;          zMax = 20 ;             zInt = 0.25
    xG = np.arange(xMin , xMax + xInt , xInt)
    zG = np.arange(zMin , zMax + zInt , zInt)
    x = np.arange(xMin + xInt / 2 , xMax + xInt / 2 , xInt)
    z = np.arange(zMin + zInt / 2 , zMax + zInt / 2 , zInt)
    X , Z = np.meshgrid(x , z)
    XG , ZG = np.meshgrid(xG , zG)

    for cnt_var in np.arange(0 , num_var):
        ########## Convert CS Grid ##########
        if var_name[cnt_var] == 'DZ' or var_name[cnt_var] == 'ZD' or var_name[cnt_var] == 'PH' or var_name[cnt_var] == 'KD' or var_name[cnt_var] == 'RH':
            eleFixCS = np.append(eleFix[0:6:2] , eleFix[6:18])
            aziMeanCS = np.mean(np.append(sel_aziCS_val_all[cnt_sel_aziCS][0:6:2] , sel_aziCS_val_all[cnt_sel_aziCS][6:18]))
            var = eval(f'np.vstack((var{var_name[cnt_var]}_cs_all[cnt_sel_aziCS][0:6:2][:] , var{var_name[cnt_var]}_cs_all[cnt_sel_aziCS][6:18][:]))')
        elif var_name[cnt_var] == 'VR' or var_name[cnt_var] == 'SW':
            eleFixCS = np.append(eleFix[1:7:2] , eleFix[6:18])
            aziMeanCS = np.mean(np.append(sel_aziCS_val_all[cnt_sel_aziCS][1:7:2] , sel_aziCS_val_all[cnt_sel_aziCS][6:18]))
            var = eval(f'np.vstack((var{var_name[cnt_var]}_cs_all[cnt_sel_aziCS][1:7:2][:] , var{var_name[cnt_var]}_cs_all[cnt_sel_aziCS][6:18][:]))')
        eleFixCS_G = np.append(np.append(eleFixCS[0] - bm_width / 2 , eleFixCS[:-1] + (eleFixCS[1:] - eleFixCS[:-1]) / 2) , eleFixCS[-1] + bm_width / 2)
        num_ele = len(eleFixCS)
        num_eleG = len(eleFixCS_G)
        DisEEM = np.empty([num_ele , num_rng])
        HgtEEM = np.empty([num_ele , num_rng])
        DisEEM_G = np.empty([num_eleG , num_rngG])
        HgtEEM_G = np.empty([num_eleG , num_rngG])
        for cnt_ele in np.arange(0 , num_ele):
            DisEEM[cnt_ele , :] , HgtEEM[cnt_ele , :] = equivalent_earth_model(eleFixCS[cnt_ele] , altitude , range)
        for cnt_eleG in np.arange(0 , num_eleG):
            DisEEM_G[cnt_eleG , :] , HgtEEM_G[cnt_eleG , :] = equivalent_earth_model(eleFixCS_G[cnt_eleG] , altitude , rangeG)
        points = np.hstack([DisEEM.reshape([DisEEM.size , 1]) , HgtEEM.reshape([HgtEEM.size , 1])])
        
        cmin = cfgc.colors(var_name[cnt_var] , 'S')[4]
        cmax = cfgc.colors(var_name[cnt_var] , 'S')[5]
        var[var > cmax] = np.nan
        var[var < cmin] = np.nan

        varP = var.reshape([var.size , 1])
        varXZ = gd(points , varP , (X , Z) , method = 'linear' , fill_value = np.nan)[: , : , 0]

        ########## Path & Info Setting ##########
        outPath = f'{outDir}CS/{var_name[cnt_var]}_{dateStrLST}_{timeStrLST}_{sel_aziCS[cnt_sel_aziCS] * 10:04.0f}.png'
        outPath2 = f'{outDir2}CS/{var_name[cnt_var]}_{dateStrLST}_{timeStrLST}_{sel_aziCS[cnt_sel_aziCS] * 10:04.0f}.png'
        staInfo = {'name' : station_name , 'lon' : longitude , 'lat' : latitude , 'alt' : altitude , 'scn' : scan_type}
        varInfo = {'name' : var_name[cnt_var] , 'plotname' : var_plot[cnt_var] , 'units' : var_units[cnt_var]}

        ########## Plot CS ##########
        axis_cs = {'xMin' : xMin , 'xMax' : xMax , 'xInt' : 5 , 'zMin' : 0 , 'zMax' : 20 , 'zInt' : 1}
        plot_cs(axis_cs , DisEEM_G , HgtEEM_G , var , varInfo , staInfo , aziMeanCS , datetimeStrLST , outPath)
        plot_cs_reorder(axis_cs , XG , ZG , varXZ , varInfo , staInfo , aziMeanCS , datetimeStrLST , outPath2)

# if __name__ == '__main__':
#     main()