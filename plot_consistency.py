########################################
######### plot_consistency.py ##########
######## Author: Wei-Jhih Chen #########
######### Update: 2022/11/18 ###########
########################################

import time
import matplotlib
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from read_furuno_wr2100_archive import *
from plot_radar import make_dirs
from datetime import datetime as dtdt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from matplotlib.colors import LogNorm

CASE_DATE = '20220826'
CASE_START = dtdt(2022 , 8 , 26 , 6 , 21 , 26)
# CASE_END = dtdt(2022 , 8 , 26 , 6 , 22 , 40)
# CASE_END = dtdt(2022 , 8 , 26 , 6 , 23 , 55)
# CASE_START = dtdt(2022 , 8 , 26 , 6 , 23 , 55)
CASE_END = dtdt(2022 , 8 , 26 , 6 , 27 , 38)
PLOT_TYPE = 'PPPI'
SCAN_TYPE = 'RHI'
STATION_NAME = 'NTU'    # Station Name
PRODUCT_ID = '0092'     # Product number

# SEL_AZI = [210]
# SEL_AZI = [201 , 204 , 207 , 210 , 213 , 216 , 219]
SEL_AZI = np.arange(180 , 237 , 3)

INEXT = '*.gz'
HOMEDIR = Path(r'/home/C.cwj/Radar')
INDIR = HOMEDIR/'cases'/f'RAW-{STATION_NAME}'/CASE_DATE
INPATHS = INDIR.glob(f'{PRODUCT_ID}_{CASE_DATE}_*.{INEXT}')
OUTDIR_TC = HOMEDIR/'pic'/STATION_NAME/CASE_DATE/'TC'

def plot_timeconsistency(inPaths , dtStart , dtEnd , sel_azi , outDir):
    make_dirs(outDir)
    sel_times = find_volume_scan_times(inPaths , dtStart , dtEnd)
    num_grp = len(sel_times) - 1
    num_azi = len(sel_azi)
    for cnt_grp in range(num_grp):
        (datetimes1 , NULL , NULL , NULL , 
         NULL , NULL , NULL , 
         NULL , NULL , Fixed_angle1 , 
         Sweep_start_ray_index1 , Sweep_end_ray_index1 , 
         Range1 , Azimuth1 , Elevation1 , Fields1) = read_volume_scan(INDIR , PRODUCT_ID , sel_times[cnt_grp] , INEXT)
        (datetimes2 , NULL , NULL , NULL , 
         NULL , NULL , NULL , 
         NULL , NULL , Fixed_angle2 , 
         Sweep_start_ray_index2 , Sweep_end_ray_index2 , 
         Range2 , Azimuth2 , Elevation2 , Fields2) = read_volume_scan(INDIR , PRODUCT_ID , sel_times[cnt_grp + 1] , INEXT)
        
        for cnt_azi in range(num_azi):
            idx_azi1 = np.argmin(abs(Fixed_angle1['data'] - sel_azi[cnt_azi]))
            idx_azi2 = np.argmin(abs(Fixed_angle2['data'] - sel_azi[cnt_azi]))

            datetimeLST1 = datetimes1['data'][idx_azi1] + dt.timedelta(hours = 8)
            datetimeLST2 = datetimes2['data'][idx_azi2] + dt.timedelta(hours = 8)

            aziFix1 = Fixed_angle1['data'][idx_azi1]
            aziFix2 = Fixed_angle2['data'][idx_azi2]

            OUTPATH = outDir/f"DZ_{dtdt.strftime(datetimeLST1 , '%Y%m%d_%H%M%S')}_{dtdt.strftime(datetimeLST2 , '%H%M%S')}_{aziFix1 * 10:04.0f}.png"

            idx_ele_start1 = Sweep_start_ray_index1['data'][idx_azi1]
            idx_ele_end1 = Sweep_end_ray_index1['data'][idx_azi1]
            num_ele1 = idx_ele_end1 - idx_ele_start1 + 1
            
            idx_ele_start2 = Sweep_start_ray_index2['data'][idx_azi2]
            idx_ele_end2 = Sweep_end_ray_index2['data'][idx_azi2]
            num_ele2 = idx_ele_end2 - idx_ele_start2 + 1

            Ele1 = Elevation1['data'][idx_ele_start1 : idx_ele_end1 + 1]
            Ele2 = Elevation2['data'][idx_ele_start2 : idx_ele_end2 + 1]
            Azi1 = Azimuth1['data'][idx_ele_start1 : idx_ele_end1 + 1]
            Azi2 = Azimuth2['data'][idx_ele_start2 : idx_ele_end2 + 1]
            Var1 = Fields1['DBZ']['data'][idx_ele_start1 : idx_ele_end1 + 1]
            Var2 = Fields2['DBZ']['data'][idx_ele_start2 : idx_ele_end2 + 1]

            isincrement1 = Elevation1['data'][idx_ele_end1] - Elevation1['data'][idx_ele_start1] > 0
            if isincrement1:
                Ele2 = Ele2[::-1]
                Azi2 = Azi2[::-1]
                Var2 = Var2[::-1]
            else:
                Ele1 = Ele1[::-1]
                Azi1 = Azi1[::-1]
                Var1 = Var1[::-1]

            R2 = np.empty((num_ele1 , num_ele2))
            for cnt_ele1 in range(num_ele1):
                for cnt_ele2 in range(num_ele2):
                    R2[cnt_ele1 , cnt_ele2] = r2_score(Var1[cnt_ele1] , Var2[cnt_ele2])

            if isincrement1:
                x_max , y_max = num_ele1 , num_ele2
                argmax = np.argmax(R2 , axis = 1)
                max = np.max(R2 , axis = 1)
            else:
                x_max , y_max = num_ele2 , num_ele1
                argmax = np.argmax(R2 , axis = 0)
                max = np.max(R2 , axis = 0)
            X = np.linspace(0 , x_max - 1 , x_max)

            plt.close()
            fig , ax1 = plt.subplots(figsize = [12 , 10])
            ax1.text(0.125 , 0.95 , STATION_NAME , fontsize = 18 , ha = 'left' , color = '#666666' , transform = fig.transFigure)
            ax1.text(0.125 , 0.92 , 'Z$_{HH}$' , fontsize = 20 , ha = 'left' , transform = fig.transFigure)
            ax1.text(0.125 , 0.89 , 'Correlation of Elevations' , fontsize = 20 , ha = 'left' , transform = fig.transFigure)
            ax1.text(0.900 , 0.92 , f'Azi. {aziFix1:.2f}$^o$' , fontsize = 18 , ha = 'right' , transform = fig.transFigure)
            ax1.text(0.900 , 0.89 , f"{dtdt.strftime(datetimeLST1 , '%Y/%m/%d %H:%M:%S')} vs. {dtdt.strftime(datetimeLST2 , '%H:%M:%S LST')}" , fontsize = 20 , ha = 'right' , transform = fig.transFigure)
            ax1.plot([0 , x_max - 1] , [0 , y_max - 1] , '#bbb' , linewidth = 1 , alpha = .5)
            ax1.plot(X , argmax , c = '#000' , alpha = 1 , zorder = 0)
            ax1.tick_params(axis = 'both' , labelcolor = '#000' , labelsize = 12)
            ax1.grid(color = '#bbb' , linewidth = .5 , alpha = .5)
            ax1.set_xlim(0 , x_max - 1)
            ax1.set_ylim(0 , y_max - 1)
            ax1.set_xlabel('Index of Incremental Elevations' , fontsize = 16)
            ax1.set_ylabel('Index of Decremental Elevations' , fontsize = 16)
            ax2 = ax1.twinx()
            ax2.plot(X , max , c = '#777' , alpha = .5 , zorder = 0)
            ax2.tick_params(axis = 'y' , labelcolor = '#777' , labelsize = 12)
            ax2.set_ylim(0 , 1)
            ax2.set_ylabel('r$^{2}$ Score' , fontsize = 16)
            ax2.yaxis.label.set_color('#777')
            fig.savefig(OUTPATH , dpi = 200)
            print(f"{OUTPATH} - Save!")

def plot_selfconsistency(axis , var1 , var2 , aziFix , datetimeLST , outDir):
    OUTPATH = outDir/f"{var1['name']}_{var2['name']}_{dtdt.strftime(datetimeLST , '%Y%m%d_%H%M%S')}_{aziFix * 10:04.0f}.png"
    ########## Grid ##########
    xTick = np.arange(axis['xMin'] * 100 , (axis['xMax'] + axis['xInt']) * 100 , axis['xInt'] * 100) / 100
    yTick = np.arange(axis['yMin'] * 100 , (axis['yMax'] + axis['yInt']) * 100 , axis['yInt'] * 100) / 100
    bpWidths = (axis['xMax'] - axis['xMin']) / axis['xBin'] / 5
    ########## Plot ##########
    var1['data'] = var1['data'].reshape([-1])
    var2['data'] = var2['data'].reshape([-1])
    var1_fil = np.delete(var1['data'] , (var1['data'] < axis['xMin']) | (var1['data'] >= axis['xMax']) | (var2['data'] < axis['yMin']) | (var2['data'] >= axis['yMax']))
    var2_fil = np.delete(var2['data'] , (var1['data'] < axis['xMin']) | (var1['data'] >= axis['xMax']) | (var2['data'] < axis['yMin']) | (var2['data'] >= axis['yMax']))
    plt.close()
    fig , ax = plt.subplots(figsize = [12 , 10])
    ax.text(0.125 , 0.920 , 'NTU' , fontsize = 20 , ha = 'left' , transform = fig.transFigure)
    ax.text(0.125 , 0.890 , 'X band' , fontsize = 20 , ha = 'left' , transform = fig.transFigure)
    ax.text(0.744 , 0.920 , f'Azi. {aziFix:03.1f}$^o$ RHI' , fontsize = 20 , ha = 'right' , transform = fig.transFigure)
    ax.text(0.744 , 0.890 , dtdt.strftime(datetimeLST , '%Y/%m/%d %H:%M:%S LST') , fontsize = 20 , ha = 'right' , transform = fig.transFigure)
    # ax.plot([0 , 6] , [0.8 , 0.8] , c = 'k' , linewidth = 2 , alpha = 1 , zorder = 2)
    # ax.plot([6 , 6] , [0.8 , 1] , c = 'k' , linewidth = 2 , alpha = 1 , zorder = 2)
    # ax.plot([0 , 6] , [1 , 1] , c = 'k' , linewidth = 2 , alpha = 1 , zorder = 2)
    # ax.plot([0 , 0] , [0.8 , 1] , c = 'k' , linewidth = 2 , alpha = 1 , zorder = 2)
    H2D = plt.hist2d(var1['data'] , var2['data'] , bins = [axis['xBin'] , axis['yBin']] , range = [[axis['xMin'] , axis['xMax']] , [axis['yMin'] , axis['yMax']]] , cmap = 'hot_r' , norm = LogNorm(vmin = 1) , zorder = 0)[0]
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
    plt.xlabel(f"{var1['plotname']} ({var1['units']})" , fontsize = 18)
    plt.ylabel(f"{var2['plotname']} ({var2['units']})" , fontsize = 18)
    fig.savefig(OUTPATH , dpi = 200)
    print(f'{OUTPATH} - Done!')

def main():
    case_date = '20200716'
    case_time1 = ['123932' , '124529' , '125126' , '125723' , '130321']
    case_time2 = ['124329' , '124359' , '125517' , '125547' , '130705']
    # case_time = ['123335' , '123211']
    # case_time = ['130918' , '130734']
    # case_time = ['131514' , '131852']
    azi = [172.7 , 171.0]
    # rng = 34.75
    station_name = ['RCWF' , 'NTU']
    plot_type = ['CS' , 'RHI']
    varName1 = ['DZ' , 'ZD' , 'PH' , 'RH' , 'VR' , 'SW' , 'KD']
    varName2 = ['DZac' , 'ZDac' , 'PH' , 'RH' , 'VR' , 'SW' , 'KD']
    varPlot1 = ['Z$_{HH}$' , 'Z$_{DR}$' , '$\phi$$_{DP}$' , r'$\rho$$_{HV}$' , 'V$_R$' , 'SW' , 'K$_{DP}$']
    varPlot2 = ['Z$_{HH}$ (AC)' , 'Z$_{DR}$ (AC)' , '$\phi$$_{DP}$' , r'$\rho$$_{HV}$' , 'V$_R$' , 'SW' , 'K$_{DP}$']
    varUnits = ['dBZ' , 'dB' , 'Deg.' , '' , 'm s$^{-1}$' , 'm s$^{-1}$' , 'Deg. km$^{-1}$']
    varMin = [0 , -2 , 0 , 0.6 , -18 , 0 , -1]
    varMax = [70 , 5 , 100 , 1.1 , 18 , 6 , 5]
    num_t = len(case_time1)
    num_var = len(varName1)

    homeDir = '/home/C.cwj/Radar/'
    inDir1 = f'{homeDir}pic/{station_name[0]}/{case_date}/Reorder/'
    inDir2 = f'{homeDir}pic/{station_name[1]}/{case_date}/Reorder/'

    for cnt_var in np.arange(0 , num_var):
        for cnt_t in np.arange(0 , num_t):
            inPath1 = f'{inDir1}{plot_type[0]}/{varName1[cnt_var]}_{case_date}_{case_time1[cnt_t]}_{azi[0] * 10:04.0f}.dat'
            inPath2 = f'{inDir2}{plot_type[1]}/{varName2[cnt_var]}_{case_date}_{case_time2[cnt_t]}_{azi[1] * 10:04.0f}.dat'
            outPath = f'{homeDir}pic/Consistency/{station_name[0]}_{station_name[1]}_{varName1[cnt_var]}_{varName2[cnt_var]}_{case_date}_{case_time1[cnt_t]}_{case_time2[cnt_t]}_{azi[0] * 10:04.0f}_{azi[1] * 10:04.0f}.png'

            datetime1 = dtdt.strptime(f'{case_date}_{case_time1[cnt_t]}' , '%Y%m%d_%H%M%S')
            datetime2 = dtdt.strptime(f'{case_date}_{case_time2[cnt_t]}' , '%Y%m%d_%H%M%S')
            datetimeStrLST1 = dtdt.strftime(datetime1 , '%Y/%m/%d %H:%M:%S LST')
            datetimeStrLST2 = dtdt.strftime(datetime2 , '%Y/%m/%d %H:%M:%S LST')

            var1 = np.fromfile(inPath1)
            var2 = np.fromfile(inPath2)
            
            idx_nan = np.where(np.isnan(var1))
            var1 = np.delete(var1 , idx_nan)
            var2 = np.delete(var2 , idx_nan)
            idx_nan = np.where(np.isnan(var2))
            var1 = np.delete(var1 , idx_nan).reshape((-1 , 1))
            var2 = np.delete(var2 , idx_nan)

            model = LinearRegression(fit_intercept = True)
            model.fit(var1 , var2)
            xfit = np.linspace(varMin[cnt_var] , varMax[cnt_var] , 100).reshape((-1 , 1))
            yfit = model.predict(xfit)
            r2 = model.score(var1 , var2)

            matplotlib.use('Agg')
            plt.close()
            fig , ax = plt.subplots(figsize = [12 , 10])
            ax.text(0.125 , 0.920 , f'{station_name[0]}' , fontsize = 20 , ha = 'left' , transform = fig.transFigure)
            ax.text(0.125 , 0.890 , f'{station_name[1]}' , fontsize = 20 , ha = 'left' , transform = fig.transFigure)
            ax.text(0.225 , 0.920 , varPlot1[cnt_var] , fontsize = 20 , ha = 'left' , transform = fig.transFigure)
            ax.text(0.225 , 0.890 , varPlot2[cnt_var] , fontsize = 20 , ha = 'left' , transform = fig.transFigure)
            ax.text(0.375 , 0.920 , f'Azi. {azi[0]:03.1f}$^o$ {plot_type[0]}' , fontsize = 20 , ha = 'left' , transform = fig.transFigure)
            ax.text(0.375 , 0.890 , f'Azi. {azi[1]:03.1f}$^o$ {plot_type[1]}' , fontsize = 20 , ha = 'left' , transform = fig.transFigure)
            ax.text(0.900 , 0.920 , datetimeStrLST1 , fontsize = 20 , ha = 'right' , transform = fig.transFigure)
            ax.text(0.900 , 0.890 , datetimeStrLST2 , fontsize = 20 , ha = 'right' , transform = fig.transFigure)
            ax.plot([varMin[cnt_var] , varMax[cnt_var]] , [0 , 0] , c = '#bbbbbb' , alpha = .5 , zorder = 0)
            ax.plot([0 , 0] , [varMin[cnt_var] , varMax[cnt_var]] , c = '#bbbbbb' , alpha = .5 , zorder = 0)
            ax.plot([varMin[cnt_var] , varMax[cnt_var]] , [varMin[cnt_var] , varMax[cnt_var]] , linestyle = '--' , c = 'k')
            ax.scatter(var1 , var2 , s = 25 , c = '#444444' , marker = '.')
            ax.plot(xfit , yfit , c = 'k')
            ax.text(0.890 , 0.150 , f'r$^2$ = {r2:.2f}' , fontsize = 20 , ha = 'right' , transform = fig.transFigure)
            if model.intercept_ > 0:
                ax.text(0.890 , 0.120 , f'{station_name[1]} = {model.coef_[0]:.2f} x {station_name[0]} + {model.intercept_:.2f}' , fontsize = 20 , ha = 'right' , transform = fig.transFigure)
            else:
                ax.text(0.890 , 0.120 , f'{station_name[1]} = {model.coef_[0]:.2f} x {station_name[0]} - {-model.intercept_:.2f}' , fontsize = 20 , ha = 'right' , transform = fig.transFigure)
            ax.axis([varMin[cnt_var] , varMax[cnt_var] , varMin[cnt_var] , varMax[cnt_var]])
            ax.grid(visible = True , c = '#bbbbbb' , linewidth = .5 , alpha = .5 , zorder = 0)
            plt.xticks(size = 12)
            plt.yticks(size = 12)
            plt.xlabel(f'{station_name[0]}-{varPlot1[cnt_var]} ({varUnits[cnt_var]})' , fontsize = 18)
            plt.ylabel(f'{station_name[1]}-{varPlot2[cnt_var]} ({varUnits[cnt_var]})' , fontsize = 18)
            fig.savefig(outPath , dpi = 200)

        # outPath = f'{homeDir}pic/Consistency/{station_name[0]}_{station_name[1]}_{varName1[cnt_var]}_{varName2[cnt_var]}_{case_date}_{case_time1[cnt_t]}_{case_time2[cnt_t]}_{azi[0] * 10:04.0f}_{azi[1] * 10:04.0f}.png'

if __name__ == '__main__':
    print('Processing Start!')
    RUNTIME = time.time()
    plot_timeconsistency(INPATHS , CASE_START , CASE_END , SEL_AZI , OUTDIR_TC)
    RUNTIME = time.time() - RUNTIME
    print(f'Runtime: {RUNTIME} second(s) - Processing End!')