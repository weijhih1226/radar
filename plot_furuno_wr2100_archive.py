#!/home/C.cwj/anaconda3/envs/pyart_env/bin/python3

########################################
#### plot_furuno_wr2100_archive.py #####
######## Author: Wei-Jhih Chen #########
######### Update: 2022/07/20 ###########
########################################

import os , glob , time
import numpy as np
import datetime as dt
from filter import *
from convert_grid import *
from attenuation_correction import *
from read_furuno_wr2100_archive import *
from plot_radar import *
from plot_consistency import *
from datetime import datetime as dtdt
from scipy.interpolate import griddata as gd

def main():
    ########## Case Setting ##########
    case_date = '20200716'
    # case_time = '043159'
    # case_time = '043753'    # ZDR Best
    # case_time = '044347'    # KDP Best
    # case_time = '044941'
    # case_time = '045535'
    case_time = '050129'    # DBZ Best
    # case_time = '050723'
    # case_time = '051317'

    num_selAzi = 57
    # num_selAzi = 2

    num_selAzi = 25
    # num_selAzi = 34

    num_selAzi = 34
    # num_selAzi = 25

    ########## Parameters Setting ##########
    station_name = 'NTU'        # Station Name
    product_number = '0092'     # Product number
    scan_type = 'RHI'
    flDZ_min = 0
    flRH_min = 0.7
    flRH_max = 1.1

    # sel_azi = 171
    # sel_azi = 75
    sel_azi = 102
    sel_var = [0 , 1 , 2 , 3 , 4 , 5 , 6 , 9 , 10]
    var_inName = np.array(['DBZ' , 'ZDR' , 'PHIDP' , 'KDP' , 'RHOHV' , 'VEL' , 'WIDTH' , 'RRR' , 'QC_INFO'])
    var_name = np.array(['DZ' , 'ZD' , 'PH' , 'KD' , 'RH' , 'VR' , 'SW' , 'RR' , 'QC' , 'DZac' , 'ZDac'])
    var_plot = np.array(['Z$_{HH}$' , 'Z$_{DR}$' , '$\phi$$_{DP}$' , 'K$_{DP}$' , r'$\rho$$_{HV}$' , 'V$_R$' , 'SW' , 'RainRate' , 'QC Info' , 'Z$_{HH}$ (AC)' , 'Z$_{DR}$ (AC)'])
    var_units = np.array(['dBZ' , 'dB' , 'Deg.' , 'Deg. km$^{-1}$' , '' , 'm s$^{-1}$' , 'm s$^{-1}$' , 'mm hr$^{-1}$' , '' , 'dBZ' , 'dB'])

    ########## Path Setting ##########
    homeDir = '/home/C.cwj/Radar/'
    inDir = f'{homeDir}cases/RAW-{station_name}/{case_date}/'
    inPaths = sorted(glob.glob(f'{inDir}/{product_number}_{case_date}_{case_time}_*.rhi'))
    outDir = f'{homeDir}pic/{station_name}/{case_date}/{scan_type}/'
    outDir2 = f'{homeDir}pic/{station_name}/{case_date}/Reorder/{scan_type}/'
    outDir3 = f'{homeDir}pic/{station_name}/{case_date}/SC/'
    if not(os.path.isdir(outDir)):
        os.makedirs(outDir)
        print(f'Create Directory: {outDir}')
    if not(os.path.isdir(outDir2)):
        os.makedirs(outDir2)
        print(f'Create Directory: {outDir2}')
    if not(os.path.isdir(outDir3)):
        os.makedirs(outDir3)
        print(f'Create Directory: {outDir3}')

    ########## Plot Each Sweep ##########
    print('Processing Start!')
    start_time = time.time()
    # cnt_ray_all = 0
    # sweep_start_ray_index = np.array([] , dtype = np.ushort)
    # sweep_end_ray_index = np.array([] , dtype = np.ushort)
    # datetime = np.array([] , dtype = dtdt)
    # fixed_angle = np.array([] , dtype = np.float32)
    # Rain = np.array([] , dtype = np.float32)
    # Zhh = np.array([] , dtype = np.float32)
    # V = np.array([] , dtype = np.float32)
    # Zdr = np.array([] , dtype = np.float32)
    # Kdp = np.array([] , dtype = np.float32)
    # Phidp = np.array([] , dtype = np.float32)
    # Rhohv = np.array([] , dtype = np.float32)
    # W = np.array([] , dtype = np.float32)
    # QC = np.array([] , dtype = np.ushort)
    # Quality_information = np.array([] , dtype = np.int8)
    # Signal_shading = np.array([] , dtype = np.int8)
    # Signal_extinction = np.array([] , dtype = np.int8)
    # Clutter_reference = np.array([] , dtype = np.int8)
    # Pulse_blind_area = np.array([] , dtype = np.int8)
    # Sector_blank = np.array([] , dtype = np.int8)
    # Fix_1 = np.array([] , dtype = np.int8)

    # Cartesian Grid
    xMin = -40 ; xMax = 0 ; xInt = 0.25
    zMin = 0 ; zMax = 20 ; zInt = 0.25
    xG = np.arange(xMin , xMax + xInt , xInt)
    zG = np.arange(zMin , zMax + zInt , zInt)
    x = np.arange(xMin + xInt / 2 , xMax + xInt / 2 , xInt)
    z = np.arange(zMin + zInt / 2 , zMax + zInt / 2 , zInt)
    X , Z = np.meshgrid(x , z)
    XG , ZG = np.meshgrid(xG , zG)

    num_file = len(inPaths)
    # for cnt_file in np.arange(num_selAzi , num_selAzi + 1):
    for cnt_file in np.arange(0 , num_file):
        inPath = inPaths[cnt_file]

        ########## Read ##########
        (datetime , range , fields , metadata , scan_type , 
         latitude , longitude , altitude , 
         sweep_number , sweep_mode , aziFix , 
         sweep_start_ray_index , sweep_end_ray_index , 
         azimuth , elevation , 
         instrument_parameters) = read_rhi(inPath)

        datetime = datetime['data']
        range = range['data']
        elevation = elevation['data']
        aziFix = aziFix['data']

        print(instrument_parameters['radar_constant_H'] , instrument_parameters['radar_constant_V'])

        if abs(aziFix - sel_azi) > 1:
            print(f'Skip Azimuth: {aziFix:.2f}^o')
            continue

        latitude = latitude['data']
        longitude = longitude['data']
        altitude = altitude['data']

        varDZ = fields['DBZ']['data']
        varZD = fields['ZDR']['data']
        varPH = fields['PHIDP']['data']
        varKD = fields['KDP']['data']
        varRH = fields['RHOHV']['data']
        varVR = fields['VEL']['data']
        varSW = fields['WIDTH']['data']

        # Radar constant calibration (Origin: 1.7027899999999999e-16(H) , 1.9105599999999997e-16(V) ; After: 6.680812773836028e-15(H,V))
        radar_constant = lambda wavelength , loss , power_transmission , antenna_gain , beamwidth_H , beamwidth_V , pulse_width , K_2: (np.pi ** 5 * 10 ** -17 * power_transmission * antenna_gain ** 2 * beamwidth_H * beamwidth_V * pulse_width * K_2) / (6.75 * 2 ** 14 * np.log(2) * wavelength ** 2 * loss) / 1000
        WAVELENGTH = 3.19           # cm
        LOSS_H = 10 ** (4 / 10)     # log10 to ratio
        LOSS_V = 10 ** (4 / 10)     # log10 to ratio
        POWER_TRANSMISSION_H = 100  # W
        POWER_TRANSMISSION_V = 100  # W
        GAIN_H = 10 ** (34.0 / 10)  # log10 to ratio
        GAIN_V = 10 ** (34.0 / 10)  # log10 to ratio
        BEAMWIDTH_H = 2.7           # degree
        BEAMWIDTH_V = 2.7           # degree
        if instrument_parameters['Tx_pulse_specification'] == 1 or instrument_parameters['Tx_pulse_specification'] == 2 or instrument_parameters['Tx_pulse_specification'] == 3:
            PULSE_WIDTH = 50        # us (Q0N , long)
        elif instrument_parameters['Tx_pulse_specification'] == 4 or instrument_parameters['Tx_pulse_specification'] == 7:
            PULSE_WIDTH = 0.5       # us (P0N , short)
        elif instrument_parameters['Tx_pulse_specification'] == 5 or instrument_parameters['Tx_pulse_specification'] == 8:
            PULSE_WIDTH = 0.66      # us (P0N , short)
        elif instrument_parameters['Tx_pulse_specification'] == 6 or instrument_parameters['Tx_pulse_specification'] == 9 or instrument_parameters['Tx_pulse_specification'] == 10:
            PULSE_WIDTH = 1         # us (P0N , short)
        K_2 = 0.93
        RC_H = radar_constant(WAVELENGTH , LOSS_H , POWER_TRANSMISSION_H , GAIN_H , BEAMWIDTH_H , BEAMWIDTH_V , PULSE_WIDTH , K_2)
        RC_V = radar_constant(WAVELENGTH , LOSS_V , POWER_TRANSMISSION_V , GAIN_V , BEAMWIDTH_H , BEAMWIDTH_V , PULSE_WIDTH , K_2)
        varZV = 10 * np.log10(10 ** ((varDZ - varZD) / 10))
        varDZ = 10 * np.log10((10 ** (varDZ / 10) * instrument_parameters['radar_constant_H'] / RC_H))
        varZV = 10 * np.log10((10 ** (varZV / 10) * instrument_parameters['radar_constant_V'] / RC_V))
        varZD = 10 * np.log10(10 ** ((varDZ - varZV) / 10))

        ########## Filters ##########
        # varDZ
        varDZ = var_filter(varDZ , varDZ , flDZ_min , None)
        varZD = var_filter(varDZ , varZD , flDZ_min , None)
        varPH = var_filter(varDZ , varPH , flDZ_min , None)
        varKD = var_filter(varDZ , varKD , flDZ_min , None)
        varRH = var_filter(varDZ , varRH , flDZ_min , None)
        varVR = var_filter(varDZ , varVR , flDZ_min , None)
        # varRH
        varDZ = var_filter(varRH , varDZ , flRH_min , flRH_max)
        varZD = var_filter(varRH , varZD , flRH_min , flRH_max)
        varPH = var_filter(varRH , varPH , flRH_min , flRH_max)
        varKD = var_filter(varRH , varKD , flRH_min , flRH_max)
        varVR = var_filter(varRH , varVR , flRH_min , flRH_max)
        # Attenuation Correction
        varDZac , varZDac = attenuation_correction_X(varDZ , varZD , varKD)

        ########## Plot ##########
        dateStrLST = dtdt.strftime(datetime + dt.timedelta(hours = 8) , '%Y%m%d')
        timeStrLST = dtdt.strftime(datetime + dt.timedelta(hours = 8) , '%H%M%S')
        datetimeStrLST = dtdt.strftime(datetime + dt.timedelta(hours = 8) , '%Y/%m/%d %H:%M:%S LST')

        rangeG = np.append(range , range[-1] + (range[-1] - range[-2]))   # Units: km
        elevationG = np.append(elevation - np.append(elevation[1] - elevation[0] , elevation[1:] - elevation[:-1]) / 2 , elevation[-1] + (elevation[-1] - elevation[-2]) / 2)

        num_rng = len(range)
        num_ray = len(elevation)
        num_rngG = len(rangeG)
        num_rayG = len(elevationG)
        DisEEM = np.empty([num_ray , num_rng])
        HgtEEM = np.empty([num_ray , num_rng])
        DisEEM_G = np.empty([num_rayG , num_rngG])
        HgtEEM_G = np.empty([num_rayG , num_rngG])
        for cnt_ray in np.arange(0 , num_ray):
            DisEEM[cnt_ray , :] , HgtEEM[cnt_ray , :] = equivalent_earth_model(elevation[cnt_ray] , altitude , range)
        for cnt_rayG in np.arange(0 , num_rayG):
            DisEEM_G[cnt_rayG , :] , HgtEEM_G[cnt_rayG , :] = equivalent_earth_model(elevationG[cnt_rayG] , altitude , rangeG)
        points = np.hstack([DisEEM.reshape([DisEEM.size , 1]) , HgtEEM.reshape([HgtEEM.size , 1])])
        
        # for cnt_var in [0]:
        for cnt_var in sel_var:
            var = eval(f'var{var_name[cnt_var]}')
            varP = var.reshape([var.size , 1]).filled(fill_value = np.nan)
            varXZ = gd(points , varP , (X , Z) , method = 'linear' , fill_value = np.nan)[: , : , 0]

            outPath = f'{outDir}{var_name[cnt_var]}_{dateStrLST}_{timeStrLST}_{aziFix * 10:04.0f}.png'
            outPath2 = f'{outDir2}{var_name[cnt_var]}_{dateStrLST}_{timeStrLST}_{aziFix * 10:04.0f}.png'
            staInfo = {'name' : station_name , 'lon' : longitude , 'lat' : latitude , 'alt' : altitude , 'scn' : scan_type}
            varInfo = {'name' : var_name[cnt_var] , 'plotname' : var_plot[cnt_var] , 'units' : var_units[cnt_var]}

            # axis_rhi = {'xMin' : 0 , 'xMax' : 40 , 'xInt' : 5 , 'zMin' : 0 , 'zMax' : 20 , 'zInt' : 1}
            axis_rhi = {'xMin' : xMin , 'xMax' : xMax , 'xInt' : 5 , 'zMin' : zMin , 'zMax' : zMax , 'zInt' : 1}
            plot_rhi(axis_rhi , DisEEM_G , HgtEEM_G , var , varInfo , staInfo , aziFix , datetimeStrLST , outPath , 'X')
            plot_rhi_reorder(axis_rhi , XG , ZG , varXZ , varInfo , staInfo , aziFix , datetimeStrLST , outPath2 , 'X')

        outPath3 = f'{outDir3}{var_name[1]}_{var_name[4]}_{dateStrLST}_{timeStrLST}_{aziFix * 10:04.0f}.png'
        axis_sc = {'xMin' : 0 , 'xMax' : 6 , 'xInt' : 1 , 'xBin' : 12 , 'yMin' : 0.8 , 'yMax' : 1 , 'yInt' : 0.02 , 'yBin' : 20}
        plot_selfconsistency(axis_sc , varZD , varRH , var_plot[1] , var_plot[4] , var_units[1] , var_units[4] , aziFix , datetimeStrLST , outPath3)

        outPath3 = f'{outDir3}{var_name[0]}_{var_name[4]}_{dateStrLST}_{timeStrLST}_{aziFix * 10:04.0f}.png'
        axis_sc = {'xMin' : 0 , 'xMax' : 70 , 'xInt' : 10 , 'xBin' : 14 , 'yMin' : 0.8 , 'yMax' : 1 , 'yInt' : 0.02 , 'yBin' : 20}
        plot_selfconsistency(axis_sc , varDZ , varRH , var_plot[0] , var_plot[4] , var_units[0] , var_units[4] , aziFix , datetimeStrLST , outPath3)

    # sweep_start_ray_index = np.append(sweep_start_ray_index , cnt_ray_all)
    # cnt_ray_all += total_number_of_sweep
    # sweep_end_ray_index = np.append(sweep_end_ray_index , cnt_ray_all - 1)
    
    end_time = time.time()
    run_time = end_time - start_time
    print(f'Runtime: {run_time} second(s) - Processing End!')

if __name__ == '__main__':
    main()