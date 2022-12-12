#!/home/C.cwj/anaconda3/envs/pyart_env/bin/python3
# -*- coding:utf-8 -*-

########################################
#### plot_furuno_wr2100_archive.py #####
######## Author: Wei-Jhih Chen #########
######### Update: 2022/12/12 ###########
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
# CASE_DATE = '20200716'
CASE_START = dtdt(2022 , 8 , 26 , 5 , 30 , 0)
CASE_END = dtdt(2022 , 8 , 26 , 6 , 0 , 0)
# CASE_DATE = '20200716'
# CASE_TIME = '043159'
# CASE_TIME = '043753'    # ZDR Best
# CASE_TIME = '044347'    # KDP Best
# CASE_TIME = '044941'
# CASE_TIME = '045535'
# CASE_TIME = '050129'    # DBZ Best
# CASE_TIME = '050723'
# CASE_TIME = '051317'
PLOT_TYPE = 'PPPI'
SCAN_TYPE = 'RHI'
STATION_NAME = 'NTU'    # Station Name
PRODUCT_ID = '0092'     # Product number
BAND = 'X'

SEL_ELE = [2.5 , 3.6 , 4.6 , 6.2 , 9.8 , 14.5 , 19.7 , 25.4 , 30.2]
SEL_AZI = []
# SEL_AZI = [171 , 75 , 102]
SEL_AZI_NUM = []
# SEL_AZI_NUM = [57]
# SEL_AZI_NUM = [57 , 25 , 34]
# SEL_AZI_NUM = [2 , 34 , 25]

INEXT = '[.gz|.rhi]'
HOMEDIR = Path(r'/home/C.cwj/Radar')
INDIR = HOMEDIR/'cases'/f'RAW-{STATION_NAME}'/CASE_DATE
INPATHS = INDIR.glob(f'{PRODUCT_ID}_{CASE_DATE}_*{INEXT}')
SHP_PATH = HOMEDIR/'Tools'/'shp'/'taiwan_county'/'COUNTY_MOI_1090820.shp'   # TWNcountyTWD97
MAT_PATH = HOMEDIR/'Tools'/'mat'/'QPESUMS_terrain.mat'                      # TWNterrainTWD97
OUTDIR = HOMEDIR/'pic'/STATION_NAME/CASE_DATE/SCAN_TYPE
OUTDIR_REORDER = HOMEDIR/'pic'/STATION_NAME/CASE_DATE/'Reorder'/SCAN_TYPE
OUTDIR_SC = HOMEDIR/'pic'/STATION_NAME/CASE_DATE/'SC'
OUTDIR_PPPI = HOMEDIR/'pic'/STATION_NAME/CASE_DATE/'PPPI'
OUTDIR_CV = HOMEDIR/'pic'/STATION_NAME/CASE_DATE/'CV'

########## Plot Setting ##########
# Plot RHI
# X_MIN_RHI = 0;          X_MAX_RHI = 40;         X_INT_RHI = 5
# X_MIN_RHI = -40;        X_MAX_RHI = 0;          X_INT_RHI = 5
X_MIN_RHI = 20;         X_MAX_RHI = 60;         X_INT_RHI = 5
Z_MIN_RHI = 0;          Z_MAX_RHI = 20;         Z_INT_RHI = 1
AXIS_RHI = {'xMin': X_MIN_RHI , 'xMax': X_MAX_RHI , 'xInt': X_INT_RHI , 
            'zMin': Z_MIN_RHI , 'zMax': Z_MAX_RHI , 'zInt': Z_INT_RHI}

# Plot PPI
X_MIN_PPI = 121.3;      X_MAX_PPI = 122.2;      X_INT_PPI = 0.1
Y_MIN_PPI = 24.3;       Y_MAX_PPI = 25.2;       Y_INT_PPI = 0.1
# X_MIN_PPI = 121.5;      X_MAX_PPI = 122.1;      X_INT_PPI = 0.1
# Y_MIN_PPI = 24.5;       Y_MAX_PPI = 25.1;       Y_INT_PPI = 0.1
AXIS_PPI = {'xMin': X_MIN_PPI , 'xMax': X_MAX_PPI , 'xInt': X_INT_PPI , 
            'yMin': Y_MIN_PPI , 'yMax': Y_MAX_PPI , 'yInt': Y_INT_PPI}

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
# X_MIN_REORDER = -40 ;   X_MAX_REORDER = 0 ;     X_INT_REORDER = 0.25
X_MIN_REORDER = 0 ;     X_MAX_REORDER = 60 ;    X_INT_REORDER = 0.25
Z_MIN_REORDER = 0 ;     Z_MAX_REORDER = 20 ;    Z_INT_REORDER = 0.25
AXIS_REORDER = {'xMin': X_MIN_REORDER , 'xMax': X_MAX_REORDER , 'xInt': X_INT_REORDER , 
                'zMin': Z_MIN_REORDER , 'zMax': Z_MAX_REORDER , 'zInt': Z_INT_REORDER}

########## Parameters Setting ##########
VAR_IN = ['DBZ' , 'ZDR' , 'PHIDP' , 'KDP' , 'RHOHV' , 'VEL' , 'WIDTH']
VAR_SELECT = ['DBZ' , 'ZDR' , 'PHIDP' , 'KDP' , 'RHOHV' , 'VEL' , 'WIDTH' , 'DBZ_AC' , 'ZDR_AC']
VAR = {'DBZ': {'name': 'DZ' , 'plotname': 'Z$_{HH}$' , 'units': 'dBZ' , 'data': None} , 
       'ZDR': {'name': 'ZD' , 'plotname': 'Z$_{DR}$' , 'units': 'dB' , 'data': None} , 
       'PHIDP': {'name': 'PH' , 'plotname': '$\phi$$_{DP}$' , 'units': 'Deg.' , 'data': None} , 
       'KDP': {'name': 'KD' , 'plotname': 'K$_{DP}$' , 'units': 'Deg. km$^{-1}$' , 'data': None} , 
       'RHOHV': {'name': 'RH' , 'plotname': r'$\rho$$_{HV}$' , 'units': '' , 'data': None} , 
       'VEL': {'name': 'VR' , 'plotname': 'V$_R$' , 'units': 'm s$^{-1}$' , 'data': None} , 
       'WIDTH': {'name': 'SW' , 'plotname': 'SW' , 'units': 'm s$^{-1}$' , 'data': None} , 
       'RRR': {'name': 'RR' , 'plotname': 'RainRate' , 'units': 'mm hr$^{-1}$' , 'data': None} , 
       'QC_INFO': {'name': 'QC' , 'plotname': 'QC Info' , 'units': '' , 'data': None} , 
       'DBZ_AC': {'name': 'DZac' , 'plotname': 'Z$_{HH}$ (AC)' , 'units': 'dBZ' , 'data': None} , 
       'ZDR_AC': {'name': 'ZDac' , 'plotname': 'Z$_{DR}$ (AC)' , 'units': 'dB' , 'data': None}}
DZ_MIN = 0
RH_MIN = 0.7
RH_MAX = 1.1
INVALID = -999

def plot_pseudo_ppi(files , selEles , inVars , shpPath , matPath , outDir):
    num_file = len(files)
    num_eleSel = len(selEles)
    Azi = np.zeros((num_file))
    EleSel = np.zeros((num_eleSel))
    var_all = {}
    for var in inVars:
        var_all[var] = np.empty((num_eleSel , num_file , 1004))

    for cnt_file in range(num_file):
        (datetime , NULL , NULL , NULL , 
         STA_LAT , STA_LON , STA_ALT , 
         NULL , NULL , aziFix , 
         NULL , NULL , 
         Rng , NULL , Ele , fields) = reader_corrected_by_radar_constant(files[cnt_file])

        Azi[cnt_file] = aziFix['data']
        Rng = Rng['data']
        Ele = Ele['data']
        if cnt_file == 0:
            datetimeLST = datetime['data'] + dt.timedelta(hours = 8)

        for cnt_ele in range(num_eleSel):
            idx_sel = np.argmin(np.abs(Ele - selEles[cnt_ele]))
            for var in inVars:
                var_all[var][cnt_ele , cnt_file] = fields[var]['data'][idx_sel]
            EleSel[cnt_ele] += Ele[idx_sel]
    EleSel /= num_file

    # Filters
    for var in ['DBZ' , 'ZDR' , 'PHIDP' , 'KDP' , 'RHOHV' , 'VEL']:
        var_all[var] = var_filter(var_all['DBZ'] , var_all[var] , DZ_MIN , None)
    for var in ['DBZ' , 'ZDR' , 'PHIDP' , 'KDP' , 'VEL']:
        var_all[var] = var_filter(var_all['RHOHV'] , var_all[var] , RH_MIN , RH_MAX)
    var_all['DBZ_AC'] , var_all['ZDR_AC'] = attenuation_correction_X(var_all['DBZ'] , var_all['ZDR'] , var_all['KDP'] , Rng[1] - Rng[0])
    
    STA_INFO = {'name' : STATION_NAME , 'lon' : STA_LON['data'] , 'lat' : STA_LAT['data'] , 'alt' : STA_ALT['data'] , 'scn' : SCAN_TYPE}
    Rng = np.append(Rng , Rng[-1] + (Rng[-1] - Rng[-2]))
    Azi = np.append(Azi - np.append(Azi[1] - Azi[0] , Azi[1:] - Azi[:-1]) / 2 , Azi[-1] + (Azi[-1] - Azi[-2]) / 2)
    for cnt_ele in range(num_eleSel):
        LonG , LatG , NULL = lonlat_map(EleSel[cnt_ele] , Azi , Rng , STA_INFO)
        for var in VAR_SELECT:
            VAR[var]['data'] = var_all[var][cnt_ele]
            plot_ppi(AXIS_PPI , LonG , LatG , VAR[var] , STA_INFO , EleSel[cnt_ele] , datetimeLST , shpPath , matPath , outDir)

def plot_pseudo_ppi2(inDir , inId , inExt , selTime , selEles , inVars , shpPath , matPath , outDir):
    num_eleSel = len(selEles)
    EleSel = np.zeros((num_eleSel))
    (datetime , NULL , NULL , NULL , 
     STA_LAT , STA_LON , STA_ALT , 
     sweep_number , sweep_mode , aziFix , 
     sweep_start_ray_index , sweep_end_ray_index , 
     range , azimuth , elevation , fields) = read_volume_scan(inDir , inId , selTime , inExt)

def plot_single_rhi(infile):
    (datetime , metadata , INSTRUMENT_PARAMETERS , scan_type , 
     latitude , longitude , altitude , 
     sweep_number , sweep_mode , aziFix , 
     sweep_start_ray_index , sweep_end_ray_index , 
     range , azimuth , elevation , fields) = reader_corrected_by_radar_constant(infile)

def main():
    make_dirs([OUTDIR , OUTDIR_REORDER , OUTDIR_SC , OUTDIR_PPPI , OUTDIR_CV])
    SEL_TIMES = find_volume_scan_times(INPATHS , CASE_START , CASE_END)

    # Plot Each Sweep
    # cnt_ray_all = 0

    # REORDER Cartesian Grid
    # X , Z = reorder_map(X_MIN_REORDER , X_MAX_REORDER , X_INT_REORDER , Z_MIN_REORDER , Z_MAX_REORDER , Z_INT_REORDER)
    # X_G , Z_G = reorder_grid_map(X_MIN_REORDER , X_MAX_REORDER , X_INT_REORDER , Z_MIN_REORDER , Z_MAX_REORDER , Z_INT_REORDER)

    for sel_time in SEL_TIMES:
        (datetimes , NULL , NULL , NULL , 
         LATITUDE , LONGITUDE , ALTITUDE , 
         NULL , NULL , Fixed_angle , 
         Sweep_start_ray_index , Sweep_end_ray_index , 
         Range , Azimuth , Elevation , Fields) = read_volume_scan(INDIR , PRODUCT_ID , sel_time , INEXT)

        datetimeLST = datetimes['data'][0] + dt.timedelta(hours = 8)
        Fixed_angle = Fixed_angle['data']
        Sweep_start_ray_index = Sweep_start_ray_index['data']
        Sweep_end_ray_index = Sweep_end_ray_index['data']
        Range = Range['data']
        Azimuth = Azimuth['data']
        Elevation = Elevation['data']

        for var in VAR_IN:
            VAR[var]['data'] = Fields[var]['data']

        # Invalid Value
        for var in VAR_IN:
            VAR[var]['data'] = ma.array(VAR[var]['data'] , mask = VAR[var]['data'] == INVALID , copy = False)
        # varDZ
        for var in ['DBZ' , 'ZDR' , 'PHIDP' , 'KDP' , 'RHOHV' , 'VEL']:
            VAR[var]['data'] = var_filter(VAR['DBZ']['data'] , VAR[var]['data'] , DZ_MIN , None)
        # varRH
        for var in ['DBZ' , 'ZDR' , 'PHIDP' , 'KDP' , 'VEL']:
            VAR[var]['data'] = var_filter(VAR['RHOHV']['data'] , VAR[var]['data'] , RH_MIN , RH_MAX)
        # Attenuation Correction
        VAR['DBZ_AC']['data'] , VAR['ZDR_AC']['data'] = attenuation_correction_X(VAR['DBZ']['data'] , VAR['ZDR']['data'] , VAR['KDP']['data'] , Range[1] - Range[0])

        ########## Plot CV ##########
        STA_INFO = {'name' : STATION_NAME , 'lon' : LONGITUDE['data'] , 'lat' : LATITUDE['data'] , 'alt' : ALTITUDE['data'] , 'scn' : SCAN_TYPE}

        LonCV , LatCV , NULL = radial_CV_grid(AXIS_REORDER , Fixed_angle , LONGITUDE['data'] , LATITUDE['data'])
        # VAR['DBZ_AC']['data'] = radial_CV(AXIS_REORDER , ALTITUDE['data'] , Fixed_angle , Sweep_start_ray_index , Sweep_end_ray_index , Range , Elevation , VAR['DBZ_AC']['data'])
        # plot_cv(AXIS_PPI , LonCV , LatCV , VAR['DBZ_AC'] , STA_INFO , datetimeLST , SHP_PATH , MAT_PATH , OUTDIR_CV , band = BAND)

        VAR['ZDR_AC']['data'] = radial_CV(AXIS_REORDER , ALTITUDE['data'] , Fixed_angle , Sweep_start_ray_index , Sweep_end_ray_index , Range , Elevation , VAR['ZDR_AC']['data'])
        plot_cv(AXIS_PPI , LonCV , LatCV , VAR['ZDR_AC'] , STA_INFO , datetimeLST , SHP_PATH , MAT_PATH , OUTDIR_CV , band = BAND)

        VAR['KDP']['data'] = radial_CV(AXIS_REORDER , ALTITUDE['data'] , Fixed_angle , Sweep_start_ray_index , Sweep_end_ray_index , Range , Elevation , VAR['KDP']['data'])
        plot_cv(AXIS_PPI , LonCV , LatCV , VAR['KDP'] , STA_INFO , datetimeLST , SHP_PATH , MAT_PATH , OUTDIR_CV , band = BAND)

        VAR['RHOHV']['data'][VAR['RHOHV']['data'] < 0.8] = np.nan
        VAR['RHOHV']['data'] = radial_CV(AXIS_REORDER , ALTITUDE['data'] , Fixed_angle , Sweep_start_ray_index , Sweep_end_ray_index , Range , Elevation , VAR['RHOHV']['data'] , ismax = False)
        plot_cv(AXIS_PPI , LonCV , LatCV , VAR['RHOHV'] , STA_INFO , datetimeLST , SHP_PATH , MAT_PATH , OUTDIR_CV , band = BAND)


        # INFILES = find_volume_scan_files(INDIR , PRODUCT_ID , sel_time , INEXT)
        
        # plot_pseudo_ppi(INFILES , SEL_ELE , VAR_IN , SHP_PATH , MAT_PATH , OUTDIR_PPPI)

        # NUM_FILE_LIST = SEL_AZI_NUM if SEL_AZI_NUM else range(len(INFILES))
        # for cnt_file in NUM_FILE_LIST:
        #     ########## Read ##########
        #     (datetime , metadata , INSTRUMENT_PARAMETERS , SCAN_TYPE , 
        #     LATITUDE , LONGITUDE , ALTITUDE , 
        #     sweep_number , sweep_mode , aziFix , 
        #     sweep_start_ray_index , sweep_end_ray_index , 
        #     RANGE , azimuth , ELEVATION , fields) = reader_select(INFILES[cnt_file] , SEL_READER)

        #     RANGE = RANGE['data']
        #     ELEVATION = ELEVATION['data']
        #     datetimeLST = datetime['data'] + dt.timedelta(hours = 8)
        #     # print(INSTRUMENT_PARAMETERS['radar_constant_H'] , INSTRUMENT_PARAMETERS['radar_constant_V'])

        #     if SEL_AZI:
        #         if not([True for azi in SEL_AZI if abs(aziFix['data'] - azi) <= 1]):
        #             print(f"Skip Azimuth: {aziFix['data']:.2f}^o")
        #             continue

        #     for var in VAR_IN:
        #         VAR[var]['data'] = fields[var]['data']

        #     ########## RADAR CONSTANT CALIBRATION ##########
        #     RC_H = RADAR_CONSTANT(
        #         WAVELENGTH , LOSS_H , POWER_TRANSMISSION_H , GAIN_H , 
        #         BEAMWIDTH_H , BEAMWIDTH_V , INSTRUMENT_PARAMETERS['pulse_width'] , K_2
        #     )
        #     RC_V = RADAR_CONSTANT(
        #         WAVELENGTH , LOSS_V , POWER_TRANSMISSION_V , GAIN_V , 
        #         BEAMWIDTH_H , BEAMWIDTH_V , INSTRUMENT_PARAMETERS['pulse_width'] , K_2
        #     )
        #     VAR['DBZ']['data'] , VAR['ZDR']['data'] = correct_Zh_Zdr_by_radar_constant(
        #         INSTRUMENT_PARAMETERS['radar_constant_H'] , INSTRUMENT_PARAMETERS['radar_constant_V'] , 
        #         RC_H , RC_V , VAR['DBZ']['data'] , VAR['ZDR']['data']
        #     )

        #     ########## Filters ##########
        #     # varDZ
        #     for var in ['DBZ' , 'ZDR' , 'PHIDP' , 'KDP' , 'RHOHV' , 'VEL']:
        #         VAR[var]['data'] = var_filter(VAR['DBZ']['data'] , VAR[var]['data'] , DZ_MIN , None)
        #     # varRH
        #     for var in ['DBZ' , 'ZDR' , 'PHIDP' , 'KDP' , 'VEL']:
        #         VAR[var]['data'] = var_filter(VAR['RHOHV']['data'] , VAR[var]['data'] , RH_MIN , RH_MAX)
        #     # Attenuation Correction
        #     VAR['DBZ_AC']['data'] , VAR['ZDR_AC']['data'] = attenuation_correction_X(VAR['DBZ']['data'] , VAR['ZDR']['data'] , VAR['KDP']['data'] , RANGE[1] - RANGE[0])

        #     ########## Plot ##########
        #     STA_INFO = {'name' : STATION_NAME , 'lon' : LONGITUDE['data'] , 'lat' : LATITUDE['data'] , 'alt' : ALTITUDE['data'] , 'scn' : SCAN_TYPE}

        #     RANGE_G = np.append(RANGE , RANGE[-1] + (RANGE[-1] - RANGE[-2]))     # Units: km
        #     ELEVATION_G = np.append(ELEVATION - np.append(ELEVATION[1] - ELEVATION[0] , ELEVATION[1:] - ELEVATION[:-1]) / 2 , ELEVATION[-1] + (ELEVATION[-1] - ELEVATION[-2]) / 2)

        #     # DIS , HGT = equivalent_earth_model_by_elevations(ELEVATION , RANGE , ALTITUDE['data'])
        #     DIS_G , HGT_G = equivalent_earth_model_by_elevations(ELEVATION_G , RANGE_G , ALTITUDE['data'])
        #     # XY_POINTS = np.hstack((DIS.reshape([DIS.size , 1]) , HGT.reshape([HGT.size , 1])))
            
        #     for var in VAR_SELECT:
        #         # VAR_POINTS = VAR[var]['data'].reshape([VAR[var]['data'].size , 1]).filled(fill_value = np.nan)
        #         # VAR[var]['reorder'] = gd(XY_POINTS , VAR_POINTS , (X , Z) , method = 'linear' , fill_value = np.nan)[: , : , 0]
        #         plot_rhi(AXIS_RHI , DIS_G , HGT_G , VAR[var] , STA_INFO , aziFix['data'] , datetimeLST , OUTDIR , 'X')
        #         # plot_rhi_reorder(AXIS_RHI , X_G , Z_G , VAR[var] , STA_INFO , aziFix['data'] , datetimeLST , OUTDIR_REORDER , 'X')
        #     # plot_selfconsistency(AXIS_ZDRH , VAR['ZDR'] , VAR['RHOHV'] , aziFix['data'] , datetimeLST , OUTDIR_SC)
        #     # plot_selfconsistency(AXIS_DZRH , VAR['DBZ'] , VAR['RHOHV'] , aziFix['data'] , datetimeLST , OUTDIR_SC)

        # # sweep_start_ray_index = np.append(sweep_start_ray_index , cnt_ray_all)
        # # cnt_ray_all += total_number_of_sweep
        # # sweep_end_ray_index = np.append(sweep_end_ray_index , cnt_ray_all - 1)

if __name__ == '__main__':
    print('Processing Start!')
    RUNTIME = timeit.timeit(main , number = 1)
    print(f'Runtime: {RUNTIME} second(s) - Processing End!')