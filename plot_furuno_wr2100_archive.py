#!/home/C.cwj/Radar/anaconda3/envs/pyart_env/bin/python

########################################
#### plot_furuno_wr2100_archive.py #####
######## Author: Wei-Jhih Chen #########
######### Update: 2022/06/09 ###########
########################################

import os , glob , time
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import cfg.color as cfgc
from numpy.ma import masked_array as mama
from datetime import datetime as dtdt
from matplotlib.colors import ListedColormap , BoundaryNorm , LogNorm
from cartopy.feature import ShapelyFeature as shpft
from read_furuno_wr2100_archive import read_rhi
from scipy.interpolate import griddata as gd

def equivalent_earth_model(elevationvation , altitudeitude , range):
    # elevationvation: Angle of elevationvation (Units: degree)
    # altitudeitude: altitudeitude of Station (Units: km)
    # range: Range (Units: km)
    a = 6371.25                             # Units: km
    k_e = 4 / 3
    theta_e_degree = elevationvation              # Units: degree
    theta_e = theta_e_degree / 180 * np.pi  # Units: radius
    hgtEEM = (range ** 2 + (k_e * a) ** 2 + 2 * range * k_e * a * np.sin(theta_e)) ** 0.5 - k_e * a + altitudeitude
    disEEM = k_e * a * np.arcsin(range * np.cos(theta_e) / (k_e * a + hgtEEM))
    return disEEM , hgtEEM

def var_filter(filVar , inVar , flMin , flMax):
    if not(flMin):
        if flMin == 0:
            flMin_bool = True
        else:
            flMin_bool = False
    else:
        flMin_bool = True
    if not(flMax):
        if flMax == 0:
            flMax_bool = True
        else:
            flMax_bool = False
    else:
        flMax_bool = True
    if not(flMin_bool) and not(flMax_bool):
        outVar = inVar
    elif not(flMin_bool):
        outVar = mama(inVar , filVar > flMax)
    elif not(flMax_bool):
        outVar = mama(inVar , filVar < flMin)
    else:
        outVar = mama(inVar , (filVar < flMin) | (filVar > flMax))
    return outVar

def attenuation_correction(varDZ , varZD , varKD):
    num_azi_elevation = varDZ.shape[0]
    num_range = varDZ.shape[1]
    b1 = 0.233
    b2 = 1.02
    d1 = 0.0298
    d2 = 1.293
    delta_r = 50
    Ah = np.zeros([num_azi_elevation , num_range])
    Adp = np.zeros([num_azi_elevation , num_range])
    varDZac = mama(np.zeros([num_azi_elevation , num_range]) , np.zeros([num_azi_elevation , num_range]))
    varZDac = mama(np.zeros([num_azi_elevation , num_range]) , np.zeros([num_azi_elevation , num_range]))
    varKD = mama(varKD , varKD < 0)
    for cnt_range in np.arange(0 , num_range):
        Ah[: , cnt_range] = b1 * varKD[: , cnt_range] ** b2
        varDZac[: , cnt_range] = varDZ[: , cnt_range] + 2 * np.sum(Ah[: , : cnt_range] , axis = 1) * delta_r / 1000
        Adp[: , cnt_range] = d1 * varKD[: , cnt_range] ** d2
        varZDac[: , cnt_range] = varZD[: , cnt_range] + 2 * np.sum(Adp[: , : cnt_range] , axis = 1) * delta_r / 1000
    return varDZac , varZDac

def plot_rhi(axis , XEEM , ZEEM , var , varInfo , staInfo , azi , datetimeStrLST , outPath):
    ########## Grid ##########
    xTick = np.arange(axis['xMin'] * 10 , axis['xMax'] * 10 + axis['xInt'] * 10 , axis['xInt'] * 10) / 10
    zTick = np.arange(axis['zMin'] * 10 , axis['zMax'] * 10 + axis['zInt'] * 10 , axis['zInt'] * 10) / 10
    ########## Color ##########
    colors , levels , ticks , tickLabels , cmin , cmax = cfgc.colors(varInfo['name'] , 'X')
    ########## Plot ##########
    plt.close()
    fig , ax = plt.subplots(figsize = [12 , 10])
    ax.text(0.125 , 0.920 , staInfo['name'] , fontsize = 18 , ha = 'left' , color = '#666666' , transform = fig.transFigure)
    ax.text(0.125 , 0.890 , varInfo['plotname'] , fontsize = 20 , ha = 'left' , transform = fig.transFigure)
    ax.text(0.744 , 0.920 , f'Azi. {azi:.2f}$^o$ RHI' , fontsize = 18 , ha = 'right' , transform = fig.transFigure)
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
    PC = ax.pcolormesh(XEEM , ZEEM , var , shading = 'flat' , cmap = cmap , norm = norm , alpha = 1)
    PC.set_clim(cmin , cmax)
    ax.grid(visible = True , color = '#bbbbbb' , linewidth = .5 , alpha = .5 , zorder = 0)
    cbar = plt.colorbar(PC , orientation = 'vertical' , ticks = ticks , extend = 'both')
    cbar.ax.set_yticklabels(tickLabels)
    cbar.ax.tick_params(labelsize = 12)
    cbar.set_label(varInfo['units'] , size = 12)
    fig.savefig(outPath , dpi = 200)
    print(f"{outPath} - Done!")

def plot_rhi_reorder(axis , XG , ZG , varXZ , varInfo , staInfo , azi , datetimeStrLST , outPath):
    ########## Grid ##########
    xTick = np.arange(axis['xMin'] * 10 , axis['xMax'] * 10 + axis['xInt'] * 10 , axis['xInt'] * 10) / 10
    zTick = np.arange(axis['zMin'] * 10 , axis['zMax'] * 10 + axis['zInt'] * 10 , axis['zInt'] * 10) / 10
    ########## Color ##########
    colors , levels , ticks , tickLabels , cmin , cmax = cfgc.colors(varInfo['name'] , 'X')
    ########## Plot ##########
    plt.close()
    fig , ax = plt.subplots(figsize = [12 , 10])
    ax.text(0.125 , 0.920 , staInfo['name'] , fontsize = 18 , ha = 'left' , color = '#666666' , transform = fig.transFigure)
    ax.text(0.125 , 0.890 , varInfo['plotname'] , fontsize = 20 , ha = 'left' , transform = fig.transFigure)
    ax.text(0.744 , 0.920 , f'Azi. {azi:.2f}$^o$ RHI' , fontsize = 18 , ha = 'right' , transform = fig.transFigure)
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

def plot_selfconsistency(axis , var1 , var2 , varPlot1 , varPlot2 , varUnits1 , varUnits2 , aziFix , datetimeStrLST , outPath):
    ########## Grid ##########
    xTick = np.arange(axis['xMin'] * 100 , (axis['xMax'] + axis['xInt']) * 100 , axis['xInt'] * 100) / 100
    yTick = np.arange(axis['yMin'] * 100 , (axis['yMax'] + axis['yInt']) * 100 , axis['yInt'] * 100) / 100
    bpWidths = (axis['xMax'] - axis['xMin']) / axis['xBin'] / 5
    ########## Plot ##########
    var1 = var1.reshape([-1])
    var2 = var2.reshape([-1])
    var1_fil = np.delete(var1 , (var1 < axis['xMin']) | (var1 >= axis['xMax']) | (var2 < axis['yMin']) | (var2 >= axis['yMax']))
    var2_fil = np.delete(var2 , (var1 < axis['xMin']) | (var1 >= axis['xMax']) | (var2 < axis['yMin']) | (var2 >= axis['yMax']))
    plt.close()
    fig , ax = plt.subplots(figsize = [12 , 10])
    ax.text(0.125 , 0.920 , 'NTU' , fontsize = 20 , ha = 'left' , transform = fig.transFigure)
    ax.text(0.125 , 0.890 , 'X band' , fontsize = 20 , ha = 'left' , transform = fig.transFigure)
    ax.text(0.744 , 0.920 , f'Azi. {aziFix:03.1f}$^o$ RHI' , fontsize = 20 , ha = 'right' , transform = fig.transFigure)
    ax.text(0.744 , 0.890 , datetimeStrLST , fontsize = 20 , ha = 'right' , transform = fig.transFigure)
    # ax.plot([0 , 6] , [0.8 , 0.8] , c = 'k' , linewidth = 2 , alpha = 1 , zorder = 2)
    # ax.plot([6 , 6] , [0.8 , 1] , c = 'k' , linewidth = 2 , alpha = 1 , zorder = 2)
    # ax.plot([0 , 6] , [1 , 1] , c = 'k' , linewidth = 2 , alpha = 1 , zorder = 2)
    # ax.plot([0 , 0] , [0.8 , 1] , c = 'k' , linewidth = 2 , alpha = 1 , zorder = 2)
    H2D = plt.hist2d(var1 , var2 , bins = [axis['xBin'] , axis['yBin']] , range = [[axis['xMin'] , axis['xMax']] , [axis['yMin'] , axis['yMax']]] , cmap = 'hot_r' , norm = LogNorm(vmin = 1) , zorder = 0)[0]
    num_x , num_y = H2D.shape
    for cnt_x in np.arange(0 , num_x):
        # bp = []
        # for cnt_y in np.arange(0 , num_y):
        #     bp0 = np.ones([int(H2D[cnt_x , cnt_y])]) * (axis['yMin'] + (axis['yMax'] - axis['yMin']) / axis['yBin'] * (cnt_y + 0.5))
        #     bp = np.append(bp , bp0)
        bp = var2_fil[(var1_fil >= axis['xMin'] + (axis['xMax'] - axis['xMin']) / axis['xBin'] * cnt_x) & (var1_fil < axis['xMin'] + (axis['xMax'] - axis['xMin']) / axis['xBin'] * (cnt_x + 1))]
        BP = plt.boxplot(bp , positions = [axis['xMin'] + (axis['xMax'] - axis['xMin']) / axis['xBin'] * (cnt_x + 0.5)] , widths = bpWidths , manage_ticks = False , patch_artist = True , showfliers = False , zorder = 2)
        BP['boxes'][0].set(color = '#444444' , linewidth = 1)
        BP['boxes'][0].set(facecolor = '#444444')
        BP['medians'][0].set(color = 'k')
        BP['whiskers'][0].set(color = '#444444')
        BP['whiskers'][1].set(color = '#444444')
        BP['caps'][0].set(color = '#444444')
        BP['caps'][1].set(color = '#444444')
    ax.axis([axis['xMin'] , axis['xMax'] , axis['yMin'] , axis['yMax']])
    ax.grid(visible = True , c = '#bbbbbb' , linewidth = .5 , alpha = .5 , zorder = 1)
    cbar = plt.colorbar(orientation = 'vertical' , extend = 'max')
    cbar.ax.tick_params(labelsize = 12)
    plt.xticks(xTick , size = 12)
    plt.yticks(yTick , size = 12)
    plt.xlabel(f'{varPlot1} ({varUnits1})' , fontsize = 18)
    plt.ylabel(f'{varPlot2} ({varUnits2})' , fontsize = 18)
    fig.savefig(outPath , dpi = 200)
    print(f"{outPath} - Done!")

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
        varDZ = var_filter(varDZ , varDZ , flDZ_min , [])
        varZD = var_filter(varDZ , varZD , flDZ_min , [])
        varPH = var_filter(varDZ , varPH , flDZ_min , [])
        varKD = var_filter(varDZ , varKD , flDZ_min , [])
        varRH = var_filter(varDZ , varRH , flDZ_min , [])
        varVR = var_filter(varDZ , varVR , flDZ_min , [])
        # varRH
        varDZ = var_filter(varRH , varDZ , flRH_min , flRH_max)
        varZD = var_filter(varRH , varZD , flRH_min , flRH_max)
        varPH = var_filter(varRH , varPH , flRH_min , flRH_max)
        varKD = var_filter(varRH , varKD , flRH_min , flRH_max)
        varVR = var_filter(varRH , varVR , flRH_min , flRH_max)
        # Attenuation Correction
        varDZac , varZDac = attenuation_correction(varDZ , varZD , varKD)

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
            plot_rhi(axis_rhi , DisEEM_G , HgtEEM_G , var , varInfo , staInfo , aziFix , datetimeStrLST , outPath)
            plot_rhi_reorder(axis_rhi , XG , ZG , varXZ , varInfo , staInfo , aziFix , datetimeStrLST , outPath2)

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