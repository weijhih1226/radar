########################################
######### plot_consistency.py ##########
######## Author: Wei-Jhih Chen #########
######### Update: 2022/10/30 ###########
########################################

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dtdt
from sklearn.linear_model import LinearRegression
from matplotlib.colors import LogNorm

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

case_date = '20200716'
case_time1 = ['123932' , '124529' , '125126' , '125723' , '130321']
case_time2 = ['124329' , '124359' , '125517' , '125547' , '130705']
# case_time = ['123335' , '123211']
# case_time = ['123932' , '124329']
# case_time = ['124529' , '124359']
# case_time = ['125126' , '125517']
# case_time = ['125723' , '125547']
# case_time = ['130321' , '130705']
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