#!/home/C.cwj/anaconda3/envs/pyart_env/bin/python3

########################################
#### plot_furuno_wr2100_archive.py #####
######## Author: Wei-Jhih Chen #########
######### Update: 2022/10/30 ###########
########################################

import timeit
import numpy as np
import datetime as dt
from filter import *
from convert_grid import *
from attenuation_correction import *
from read_furuno_wr2100_archive import *
from plot_radar import *
from plot_consistency import *
from pathlib import Path
from datetime import datetime as dtdt
from scipy.interpolate import griddata as gd

########## INPUT Setting ##########
CASE_DATE = '20220826'
CASE_TIME = '050002'    # DBZ Best
# CASE_DATE = '20200716'
# CASE_TIME = '043159'
# CASE_TIME = '043753'    # ZDR Best
# CASE_TIME = '044347'    # KDP Best
# CASE_TIME = '044941'
# CASE_TIME = '045535'
# CASE_TIME = '050129'    # DBZ Best
# CASE_TIME = '050723'
# CASE_TIME = '051317'
SCAN_TYPE = 'RHI'
STATION_NAME = 'NTU'    # Station Name
PRODUCT_ID = '0092'     # Product number

INEXT = 'rhi.gz'
HOMEDIR = Path(r'/home/C.cwj/Radar')
INDIR = HOMEDIR/'cases'/f'RAW-{STATION_NAME}'/CASE_DATE
INPATHS = sorted(list(INDIR.glob(f'{PRODUCT_ID}_{CASE_DATE}_{CASE_TIME}_*.{INEXT}')))
OUTDIR = HOMEDIR/'pic'/STATION_NAME/CASE_DATE/SCAN_TYPE
OUTDIR_REORDER = HOMEDIR/'pic'/STATION_NAME/CASE_DATE/'Reorder'/SCAN_TYPE
OUTDIR_SC = HOMEDIR/'pic'/STATION_NAME/CASE_DATE/'SC'

########## Plot Setting ##########
# Plot RHI
# X_MIN_RHI = 0;          X_MAX_RHI = 40;         X_INT_RHI = 5
X_MIN_RHI = -40;        X_MAX_RHI = 0;          X_INT_RHI = 5
Z_MIN_RHI = 0;          Z_MAX_RHI = 20;         Z_INT_RHI = 1
AXIS_RHI = {'xMin': X_MIN_RHI , 'xMax': X_MAX_RHI , 'xInt': X_INT_RHI , 
            'zMin': Z_MIN_RHI , 'zMax': Z_MAX_RHI , 'zInt': Z_INT_RHI}

# Plot Self-Consistency
X_MIN_ZDRH = 0;         X_MAX_ZDRH = 6;         X_INT_ZDRH = 1;         X_BIN_ZDRH = 12
Y_MIN_ZDRH = 0.8;       Y_MAX_ZDRH = 1;         Y_INT_ZDRH = 0.02;      Y_BIN_ZDRH = 20
AXIS_ZDRH = {'xMin': X_MIN_ZDRH , 'xMax': X_MAX_ZDRH , 'xInt': X_INT_ZDRH , 'xBin': X_BIN_ZDRH , 
             'yMin': Y_MIN_ZDRH , 'yMax': Y_MAX_ZDRH , 'yInt': Y_INT_ZDRH , 'yBin': Y_BIN_ZDRH}

X_MIN_DZRH = 0;         X_MAX_DZRH = 70;        X_INT_DZRH = 10;        X_BIN_DZRH = 14
Y_MIN_DZRH = 0.8;       Y_MAX_DZRH = 1;         Y_INT_DZRH = 0.02;      Y_BIN_DZRH = 20
AXIS_DZRH = {'xMin': X_MIN_DZRH , 'xMax': X_MAX_DZRH , 'xInt': X_INT_DZRH , 'xBin': X_BIN_DZRH , 
             'yMin': Y_MIN_DZRH , 'yMax': Y_MAX_DZRH , 'yInt': Y_INT_DZRH , 'yBin': Y_BIN_DZRH}

########## Reorder Setting ##########
X_MIN_REORDER = -40 ;   X_MAX_REORDER = 0 ;     X_INT_REORDER = 0.25
Z_MIN_REORDER = 0 ;     Z_MAX_REORDER = 20 ;    Z_INT_REORDER = 0.25

########## Parameters Setting ##########
SEL_AZI = []
# SEL_AZI = [171 , 75 , 102]
SEL_AZI_NUM = []
# SEL_AZI_NUM = [57]
# SEL_AZI_NUM = [57 , 25 , 34]
# SEL_AZI_NUM = [2 , 34 , 25]
VAR_IN = ['DBZ' , 'ZDR' , 'PHIDP' , 'KDP' , 'RHOHV' , 'VEL' , 'WIDTH']
VAR_SELECT = ['DBZ' , 'ZDR' , 'PHIDP' , 'KDP' , 'RHOHV' , 'VEL' , 'WIDTH' , 'DBZ_AC' , 'ZDR_AC']
VAR = {'DBZ': {'name': 'DZ' , 'plotname': 'Z$_{HH}$' , 'units': 'dBZ'} , 
       'ZDR': {'name': 'ZD' , 'plotname': 'Z$_{DR}$' , 'units': 'dB'} , 
       'PHIDP': {'name': 'PH' , 'plotname': '$\phi$$_{DP}$' , 'units': 'Deg.'} , 
       'KDP': {'name': 'KD' , 'plotname': 'K$_{DP}$' , 'units': 'Deg. km$^{-1}$'} , 
       'RHOHV': {'name': 'RH' , 'plotname': r'$\rho$$_{HV}$' , 'units': ''} , 
       'VEL': {'name': 'VR' , 'plotname': 'V$_R$' , 'units': 'm s$^{-1}$'} , 
       'WIDTH': {'name': 'SW' , 'plotname': 'SW' , 'units': 'm s$^{-1}$'} , 
       'RRR': {'name': 'RR' , 'plotname': 'RainRate' , 'units': 'mm hr$^{-1}$'} , 
       'QC_INFO': {'name': 'QC' , 'plotname': 'QC Info' , 'units': ''} , 
       'DBZ_AC': {'name': 'DZac' , 'plotname': 'Z$_{HH}$ (AC)' , 'units': 'dBZ'} , 
       'ZDR_AC': {'name': 'ZDac' , 'plotname': 'Z$_{DR}$ (AC)' , 'units': 'dB'}}
DZ_MIN = 0
RH_MIN = 0.7
RH_MAX = 1.1

########## RADAR CONSTANT CALIBRATION ##########
# Origin: 1.7027899999999999e-16(H) , 1.9105599999999997e-16(V)
# After: 6.680812773836028e-15(H,V))
RADAR_CONSTANT = lambda wavelength , loss , power_transmission , antenna_gain , beamwidth_H , beamwidth_V , pulse_width , K_2: \
                 (np.pi ** 5 * 10 ** -17 * power_transmission * antenna_gain ** 2 * beamwidth_H * beamwidth_V * pulse_width * K_2) / \
                 (6.75 * 2 ** 14 * np.log(2) * wavelength ** 2 * loss) / 1000
WAVELENGTH = 3.19           # cm
LOSS_H = 10 ** (4 / 10)     # log10 to ratio
LOSS_V = 10 ** (4 / 10)     # log10 to ratio
POWER_TRANSMISSION_H = 100  # W
POWER_TRANSMISSION_V = 100  # W
GAIN_H = 10 ** (34.0 / 10)  # log10 to ratio
GAIN_V = 10 ** (34.0 / 10)  # log10 to ratio
BEAMWIDTH_H = 2.7           # degree
BEAMWIDTH_V = 2.7           # degree
K_2 = 0.93

def makeDirs(dirPaths):
    for dirPath in dirPaths:
        if not(dirPath.is_dir()):
            dirPath.mkdir(parents = True)
            print(f'Create Directory: {dirPath}')

def reorder_map(xMin , xMax , xInt , yMin , yMax , yInt):
    x = np.arange(xMin + xInt / 2 , xMax + xInt / 2 , xInt)
    y = np.arange(yMin + yInt / 2 , yMax + yInt / 2 , yInt)
    return np.meshgrid(x , y)

def reorder_grid_map(xMin , xMax , xInt , yMin , yMax , yInt):
    x = np.arange(xMin , xMax + xInt , xInt)
    y = np.arange(yMin , yMax + yInt , yInt)
    return np.meshgrid(x , y)

def main():
    makeDirs([OUTDIR , OUTDIR_REORDER , OUTDIR_SC])

    ########## Plot Each Sweep ##########
    # cnt_ray_all = 0

    # REORDER Cartesian Grid
    X , Z = reorder_map(X_MIN_REORDER , X_MAX_REORDER , X_INT_REORDER , Z_MIN_REORDER , Z_MAX_REORDER , Z_INT_REORDER)
    X_G , Z_G = reorder_grid_map(X_MIN_REORDER , X_MAX_REORDER , X_INT_REORDER , Z_MIN_REORDER , Z_MAX_REORDER , Z_INT_REORDER)

    NUM_FILE_LIST = SEL_AZI_NUM if SEL_AZI_NUM else np.arange(len(INPATHS))
    for cnt_file in NUM_FILE_LIST:
        INPATH = INPATHS[cnt_file]

        ########## Read ##########
        (datetime , RANGE , fields , metadata , SCAN_TYPE , 
         LATITUDE , LONGITUDE , ALTITUDE , 
         sweep_number , sweep_mode , aziFix , 
         sweep_start_ray_index , sweep_end_ray_index , 
         azimuth , ELEVATION , 
         INSTRUMENT_PARAMETERS) = read_rhi(INPATH)

        # print(INSTRUMENT_PARAMETERS['radar_constant_H'] , INSTRUMENT_PARAMETERS['radar_constant_V'])

        if SEL_AZI:
            if not([True for azi in SEL_AZI if abs(aziFix['data'] - azi) <= 1]):
                print(f"Skip Azimuth: {aziFix['data']:.2f}^o")
                continue

        for var in VAR_IN:
            VAR[var]['data'] = fields[var]['data']

        PULSE_WIDTH = INSTRUMENT_PARAMETERS['pulse_width']
        RC_H = RADAR_CONSTANT(WAVELENGTH , LOSS_H , POWER_TRANSMISSION_H , GAIN_H , BEAMWIDTH_H , BEAMWIDTH_V , PULSE_WIDTH , K_2)
        RC_V = RADAR_CONSTANT(WAVELENGTH , LOSS_V , POWER_TRANSMISSION_V , GAIN_V , BEAMWIDTH_H , BEAMWIDTH_V , PULSE_WIDTH , K_2)
        varZV = 10 * np.log10(10 ** ((VAR['DBZ']['data'] - VAR['ZDR']['data']) / 10))
        varZV = 10 * np.log10((10 ** (varZV / 10) * INSTRUMENT_PARAMETERS['radar_constant_V'] / RC_V))
        VAR['DBZ']['data'] = 10 * np.log10((10 ** (VAR['DBZ']['data'] / 10) * INSTRUMENT_PARAMETERS['radar_constant_H'] / RC_H))
        VAR['ZDR']['data'] = 10 * np.log10(10 ** ((VAR['DBZ']['data'] - varZV) / 10))

        ########## Filters ##########
        # varDZ
        for var in ['DBZ' , 'ZDR' , 'PHIDP' , 'KDP' , 'RHOHV' , 'VEL']:
            VAR[var]['data'] = var_filter(VAR['DBZ']['data'] , VAR[var]['data'] , DZ_MIN , None)
        # varRH
        for var in ['DBZ' , 'ZDR' , 'PHIDP' , 'KDP' , 'VEL']:
            VAR[var]['data'] = var_filter(VAR['RHOHV']['data'] , VAR[var]['data'] , RH_MIN , RH_MAX)
        # Attenuation Correction
        VAR['DBZ_AC']['data'] , VAR['ZDR_AC']['data'] = attenuation_correction_X(VAR['DBZ']['data'] , VAR['ZDR']['data'] , VAR['KDP']['data'])

        ########## Plot ##########
        RANGE = RANGE['data']
        ELEVATION = ELEVATION['data']
        datetimeLST = datetime['data'] + dt.timedelta(hours = 8)
        STA_INFO = {'name' : STATION_NAME , 'lon' : LONGITUDE['data'] , 'lat' : LATITUDE['data'] , 'alt' : ALTITUDE['data'] , 'scn' : SCAN_TYPE}

        RANGE_G = np.append(RANGE , RANGE[-1] + (RANGE[-1] - RANGE[-2]))     # Units: km
        ELEVATION_G = np.append(ELEVATION - np.append(ELEVATION[1] - ELEVATION[0] , ELEVATION[1:] - ELEVATION[:-1]) / 2 , ELEVATION[-1] + (ELEVATION[-1] - ELEVATION[-2]) / 2)

        # DISEEM , HGTEEM = equivalent_earth_model_by_elevations(ELEVATION , ALTITUDE['data'] , RANGE)
        DISEEM_G , HGTEEM_G = equivalent_earth_model_by_elevations(ELEVATION_G , ALTITUDE['data'] , RANGE_G)
        # XY_POINTS = np.hstack([DISEEM.reshape([DISEEM.size , 1]) , HGTEEM.reshape([HGTEEM.size , 1])])
        
        for var in VAR_SELECT:
            # VAR_POINTS = VAR[var]['data'].reshape([VAR[var]['data'].size , 1]).filled(fill_value = np.nan)
            # VAR[var]['reorder'] = gd(XY_POINTS , VAR_POINTS , (X , Z) , method = 'linear' , fill_value = np.nan)[: , : , 0]
            plot_rhi(AXIS_RHI , DISEEM_G , HGTEEM_G , VAR[var] , STA_INFO , aziFix['data'] , datetimeLST , OUTDIR , 'X')
            # plot_rhi_reorder(AXIS_RHI , X_G , Z_G , VAR[var] , STA_INFO , aziFix['data'] , datetimeLST , OUTDIR_REORDER , 'X')
        # plot_selfconsistency(AXIS_ZDRH , VAR['ZDR'] , VAR['RHOHV'] , aziFix['data'] , datetimeLST , OUTDIR_SC)
        # plot_selfconsistency(AXIS_DZRH , VAR['DBZ'] , VAR['RHOHV'] , aziFix['data'] , datetimeLST , OUTDIR_SC)

    # sweep_start_ray_index = np.append(sweep_start_ray_index , cnt_ray_all)
    # cnt_ray_all += total_number_of_sweep
    # sweep_end_ray_index = np.append(sweep_end_ray_index , cnt_ray_all - 1)

if __name__ == '__main__':
    print('Processing Start!')
    RUNTIME = timeit.timeit(main , number = 1)
    print(f'Runtime: {RUNTIME} second(s) - Processing End!')