#!/home/C.cwj/anaconda3/envs/pyart_env/bin/python3
# -*- coding:utf-8 -*-

########################################
######## plot_nexrad_levelII.py ########
######## Author: Wei-Jhih Chen #########
######### Update: 2022/12/12 ###########
########################################

import os
import pyart as pa
import numpy as np
import datetime as dt
import cfg.color as cfgc
from filter import *
from convert_grid import *
from cross_section import *
from plot_radar import *
from pathlib import Path
from datetime import datetime as dtdt
from scipy.interpolate import griddata as gd

########## Case Setting ##########
CASE_DATE = '20200716'
# CASE_TIME = '043300'
# CASE_TIME = '043900'    # ZDR Best
# CASE_TIME = '044500'    # KDP Best
# CASE_TIME = '045100'
# CASE_TIME = '045700'
CASE_TIME = '050300'    # DBZ Best
# CASE_TIME = '050900'
# CASE_TIME = '051500'

STATION_NAME = 'RCWF'
BAND = 'S'
PLOT_TYPE = 'CS'
SCAN_TYPE = 'PPI'

sel_var = []            # All: []
sel_aziCS = [200 , 201 , 202 , 203 , 204 , 205 , 206 , 207 , 208 , 209 , 210]          # NP: []

########## Parameters Setting ##########
flRH_min = 0.7
flRH_max = 1.05
flSW_max = 4
flZD_num = 5
flPH_num = 5
smPH_num = 27
_FillValue = -9999.

LONGITUDE_NTU = 121.81599
LATITUDE_NTU = 24.76395
ALTITUDE_NTU = 0.01         # Units: km

########## Path Setting ##########
INEXT = '.raw'
HOMEDIR = Path(r'/home/C.cwj/Radar')
INDIR = HOMEDIR/'cases'/f'RAW-{STATION_NAME}'/CASE_DATE
INPATH = INDIR/f'{CASE_DATE}{CASE_TIME}{INEXT}'
SHP_PATH = HOMEDIR/'Tools'/'shp'/'taiwan_county'/'COUNTY_MOI_1090820.shp'    # TWNcountyTWD97
MAT_PATH = HOMEDIR/'Tools'/'mat'/'QPESUMS_terrain.mat'                       # TWNterrainTWD97
OUTDIR = HOMEDIR/'pic'/STATION_NAME/CASE_DATE
OUTDIR_CS = HOMEDIR/'pic'/STATION_NAME/CASE_DATE/PLOT_TYPE
OUTDIR_REORDER = HOMEDIR/'pic'/STATION_NAME/CASE_DATE/PLOT_TYPE/'Reorder'

########## Read Variables ##########
INSTRUMENT_NAME = 'NEXRAD WSR-88D'  # Instrument Name
DATA_TYPE = 'Level-II'
radar = pa.io.read_nexrad_archive(INPATH)

# Time
datetime = dtdt.strptime(radar.time['units'][14:34] , '%Y-%m-%dT%H:%M:%SZ')
datetimeLST = datetime + dt.timedelta(hours = 8)
datetimeStrLST = dtdt.strftime(datetime + dt.timedelta(hours = 8) , '%Y/%m/%d %H:%M:%S LST')

# Station
longitude = radar.longitude['data'][0]
latitude = radar.latitude['data'][0]
altitude = radar.altitude['data'][0] / 1000     # Units: km
# Find Azimuth and Distance across the Other Station
azi_NTU , dis_NTU = find_nearest_azimuth(longitude , latitude , LONGITUDE_NTU , LATITUDE_NTU)   # 172.67680892202117 , 34.635276253600125
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

SCAN_TYPE = radar.scan_type
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
for cnt_eleA in np.arange(num_eleA):
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
    LonEEM_G , LatEEM_G = polar_to_lonlat(azimuthG , disEEM_G , hgtEEM_G , longitude , latitude)

    ########## Filters ##########
    for nameVar_in in nameVars_in:
        exec(f'var{nameVar_in}_raw = var{nameVar_in}_all[idx_swpStart[cnt_eleA] : idx_swpEnd[cnt_eleA] + 1 , :]')
        exec(f'var{nameVar_in}_raw = mama(var{nameVar_in}_raw , var{nameVar_in}_raw == _FillValue)')
    if nameVars_create:
        for nameVar_create in nameVars_create:
            exec(f'var{nameVar_create} = var{nameVar_create}_raw')
    if nameVars_flRH:
        for nameVar_flRH in nameVars_flRH:
            exec(f'var{nameVar_flRH} = var_filter(varRH_raw , var{nameVar_flRH}_raw , flRH_min , flRH_max)')
    if nameVars_flSW:
        for nameVar_flSW in nameVars_flSW:
            exec(f'var{nameVar_flSW} = var_filter(varSW_raw , var{nameVar_flSW} , None , flSW_max)')
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


for cnt_sel_aziCS in np.arange(num_sel_aziCS):
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

    for cnt_var in np.arange(num_var):
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
        for cnt_ele in np.arange(num_ele):
            DisEEM[cnt_ele , :] , HgtEEM[cnt_ele , :] = equivalent_earth_model(eleFixCS[cnt_ele] , altitude , range)
        for cnt_eleG in np.arange(num_eleG):
            DisEEM_G[cnt_eleG , :] , HgtEEM_G[cnt_eleG , :] = equivalent_earth_model(eleFixCS_G[cnt_eleG] , altitude , rangeG)
        points = np.hstack([DisEEM.reshape([DisEEM.size , 1]) , HgtEEM.reshape([HgtEEM.size , 1])])
        
        cmin = cfgc.colors(var_name[cnt_var] , 'S')[4]
        cmax = cfgc.colors(var_name[cnt_var] , 'S')[5]
        var[var > cmax] = np.nan
        var[var < cmin] = np.nan

        varP = var.reshape([var.size , 1])
        varXZ = gd(points , varP , (X , Z) , method = 'linear' , fill_value = np.nan)[: , : , 0]

        ########## Path & Info Setting ##########
        staInfo = {'name' : STATION_NAME , 'lon' : longitude , 'lat' : latitude , 'alt' : altitude , 'scn' : SCAN_TYPE}
        varInfo = {'name' : var_name[cnt_var] , 'plotname' : var_plot[cnt_var] , 'units' : var_units[cnt_var]}

        ########## Plot CS ##########
        axis_cs = {'xMin' : xMin , 'xMax' : xMax , 'xInt' : 5 , 'zMin' : 0 , 'zMax' : 20 , 'zInt' : 1}
        plot_cs(axis_cs , DisEEM_G , HgtEEM_G , var , varInfo , staInfo , aziMeanCS , datetimeLST , OUTDIR_CS , BAND)
        plot_cs_reorder(axis_cs , XG , ZG , varXZ , varInfo , staInfo , aziMeanCS , datetimeLST , OUTDIR_REORDER , BAND)

# if __name__ == '__main__':
#     main()