########################################
############ plot_radar.py #############
######## Author: Wei-Jhih Chen #########
######### Update: 2022/12/12 ###########
########################################

import cfg.color as cfgc
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime as dtdt
from matplotlib.colors import ListedColormap , BoundaryNorm

def make_dir(dirPath):
    if not isinstance(dirPath , Path):
        dirPath = Path(dirPath)
    if not dirPath.is_dir():
        dirPath.mkdir(parents = True)
        print(f'Create Directory: {dirPath}')

def make_dirs(*dirPaths):
    if isinstance(dirPaths[0] , list):
        dirPaths = dirPaths[0]
    for dirPath in dirPaths:
        make_dir(dirPath)

def plot_ppi(axis , Lon , Lat , var , staInfo , eleFix , datetimeLST , shpPath , matPath , outDir , band = 'S'):
    import cartopy.crs as ccrs
    from scipy import io
    from cartopy.io.shapereader import Reader as shprd
    from cartopy.feature import ShapelyFeature as shpft

    make_dir(outDir)
    OUTPATH = outDir/f"{var['name']}_{dtdt.strftime(datetimeLST , '%Y%m%d_%H%M%S')}_{eleFix * 10:04.0f}.png"
    ########## Grid ##########
    X = np.arange(axis['xMin'] * 10 , axis['xMax'] * 10 + axis['xInt'] * 10 , axis['xInt'] * 10) / 10
    Y = np.arange(axis['yMin'] * 10 , axis['yMax'] * 10 + axis['yInt'] * 10 , axis['yInt'] * 10) / 10
    XStr = []
    YStr = []
    for cnt_X in range(len(X)):
        XStr = np.append(XStr , f'{X[cnt_X]}$^o$E')
    for cnt_Y in range(len(Y)):
        YStr = np.append(YStr , f'{Y[cnt_Y]}$^o$E')
    ########## Color ##########
    colors , levels , ticks , tickLabels , cmin , cmax = cfgc.colors(var['name'] , band)
    ########## Terrain ##########
    terrain = io.loadmat(matPath)
    ########## Plot ##########
    plt.close()
    fig , ax = plt.subplots(figsize = (12 , 10) , subplot_kw = {'projection' : ccrs.PlateCarree()})
    # axis['xMin'] , axis['yMax'] + axis['yInt'] / 10 + axis['yInt'] / 4
    ax.text(0.125 , 0.905 , staInfo['name'] , fontsize = 18 , ha = 'left' , color = '#666666' , transform = fig.transFigure)
    ax.text(0.125 , 0.875 , var['plotname'] , fontsize = 20 , ha = 'left' , transform = fig.transFigure)
    ax.text(0.744 , 0.905 , f'Elev. {eleFix:.2f}$^o$ ' + staInfo['scn'].upper() , fontsize = 18 , ha = 'right' , transform = fig.transFigure)
    ax.text(0.744 , 0.875 , dtdt.strftime(datetimeLST , '%Y/%m/%d %H:%M:%S LST') , fontsize = 20 , ha = 'right' , transform = fig.transFigure)
    ax.set_extent([axis['xMin'] , axis['xMax'] , axis['yMin'] , axis['yMax']])
    ax.gridlines(xlocs = X , ylocs = Y , color = '#bbbbbb' , linewidth = 0.5 , alpha = 0.5 , draw_labels = False)
    ax.add_feature(shpft(shprd(shpPath).geometries() , ccrs.PlateCarree() , 
                   facecolor = (1 , 1 , 1 , 0) , edgecolor = (0 , 0 , 0 , 1) , linewidth = 1 , zorder = 11))
    ax.contour(terrain['blon'] , terrain['blat'] , terrain['QPEterrain'] , levels = [500 , 1500 , 3000] , colors = '#C0C0C0' , linewidths = [0.5 , 1 , 1.5] , zorder = 10)
    ax.scatter(staInfo['lon'] , staInfo['lat'] , s = 50 , c = 'k' , marker = '^')
    plt.xticks(X , XStr , size = 10)
    plt.yticks(Y , YStr , size = 10)
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(levels , cmap.N)
    # var['data'][var['data'] > cmax] = np.nan
    # var['data'][var['data'] < cmin] = np.nan
    PC = ax.pcolormesh(Lon , Lat , var['data'] , shading = 'flat' , cmap = cmap , norm = norm , alpha = 1)
    PC.set_clim(cmin , cmax)
    cbar = plt.colorbar(PC , orientation = 'vertical' , ticks = ticks , extend = 'both')
    cbar.ax.set_yticklabels(tickLabels)
    cbar.ax.tick_params(labelsize = 12)
    cbar.set_label(var['units'] , size = 12)
    fig.savefig(OUTPATH , dpi = 200)
    print(f"{OUTPATH} - Save!")

def plot_rhi_cs(axis , X , Z , var , staInfo , azi , datetimeLST , outDir , band , method):
    make_dir(outDir)
    OUTPATH = outDir/f"{var['name']}_{dtdt.strftime(datetimeLST , '%Y%m%d_%H%M%S')}_{azi * 10:04.0f}.png"
    ########## Grid ##########
    xTick = np.arange(axis['xMin'] * 10 , axis['xMax'] * 10 + axis['xInt'] * 10 , axis['xInt'] * 10) / 10
    zTick = np.arange(axis['zMin'] * 10 , axis['zMax'] * 10 + axis['zInt'] * 10 , axis['zInt'] * 10) / 10
    ########## Color ##########
    colors , levels , ticks , tickLabels , cmin , cmax = cfgc.colors(var['name'] , band)
    ########## Plot ##########
    plt.close()
    fig , ax = plt.subplots(figsize = (12 , 10))
    ax.text(0.125 , 0.920 , staInfo['name'] , fontsize = 18 , ha = 'left' , color = '#666666' , transform = fig.transFigure)
    ax.text(0.125 , 0.890 , var['plotname'] , fontsize = 20 , ha = 'left' , transform = fig.transFigure)
    ax.text(0.744 , 0.920 , f'Azi. {azi:.2f}$^o$ {method}' , fontsize = 18 , ha = 'right' , transform = fig.transFigure)
    ax.text(0.744 , 0.890 , dtdt.strftime(datetimeLST , '%Y/%m/%d %H:%M:%S LST') , fontsize = 20 , ha = 'right' , transform = fig.transFigure)
    ax.axis([axis['xMin'] , axis['xMax'] , axis['zMin'] , axis['zMax']])
    ax.scatter(0 , staInfo['alt'] , s = 250 , c = 'k' , marker = '^')
    if azi >= 180:
        ax.invert_xaxis()
    plt.xticks(xTick , size = 10)
    plt.yticks(zTick , size = 10)
    plt.xlabel('Distance from Radar (km)')
    plt.ylabel('Altitude (km)')
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(levels , cmap.N)
    # var['data'][var['data'] > cmax] = np.nan
    # var['data'][var['data'] < cmin] = np.nan
    PC = ax.pcolormesh(X , Z , var['data'] , shading = 'flat' , cmap = cmap , norm = norm , alpha = 1)
    PC.set_clim(cmin , cmax)
    ax.grid(visible = True , color = '#bbbbbb' , linewidth = .5 , alpha = .5 , zorder = 0)
    cbar = plt.colorbar(PC , orientation = 'vertical' , ticks = ticks , extend = 'both')
    cbar.ax.set_yticklabels(tickLabels)
    cbar.ax.tick_params(labelsize = 12)
    cbar.set_label(var['units'] , size = 12)
    fig.savefig(OUTPATH , dpi = 200)
    print(f"{OUTPATH} - Save!")

def plot_cv(axis , Lon , Lat , var , staInfo , datetimeLST , shpPath , matPath , outDir , band = 'S'):
    import cartopy.crs as ccrs
    from scipy import io
    from cartopy.io.shapereader import Reader as shprd
    from cartopy.feature import ShapelyFeature as shpft

    make_dir(outDir)
    OUTPATH = outDir/f"{var['name']}_{dtdt.strftime(datetimeLST , '%Y%m%d_%H%M%S')}.png"
    ########## Grid ##########
    X = np.arange(axis['xMin'] * 10 , axis['xMax'] * 10 + axis['xInt'] * 10 , axis['xInt'] * 10) / 10
    Y = np.arange(axis['yMin'] * 10 , axis['yMax'] * 10 + axis['yInt'] * 10 , axis['yInt'] * 10) / 10
    XStr = []
    YStr = []
    for cnt_X in range(len(X)):
        XStr = np.append(XStr , f'{X[cnt_X]}$^o$E')
    for cnt_Y in range(len(Y)):
        YStr = np.append(YStr , f'{Y[cnt_Y]}$^o$E')
    ########## Color ##########
    colors , levels , ticks , tickLabels , cmin , cmax = cfgc.colors(var['name'] , band)
    ########## Terrain ##########
    terrain = io.loadmat(matPath)
    ########## Plot ##########
    plt.close()
    fig , ax = plt.subplots(figsize = (12 , 10) , subplot_kw = {'projection' : ccrs.PlateCarree()})
    ax.text(0.125 , 0.905 , staInfo['name'] , fontsize = 18 , ha = 'left' , color = '#666666' , transform = fig.transFigure)
    ax.text(0.125 , 0.875 , var['plotname'] , fontsize = 20 , ha = 'left' , transform = fig.transFigure)
    ax.text(0.744 , 0.905 , 'Composite ' + staInfo['scn'].upper() , fontsize = 18 , ha = 'right' , transform = fig.transFigure)
    ax.text(0.744 , 0.875 , dtdt.strftime(datetimeLST , '%Y/%m/%d %H:%M:%S LST') , fontsize = 20 , ha = 'right' , transform = fig.transFigure)
    ax.set_extent([axis['xMin'] , axis['xMax'] , axis['yMin'] , axis['yMax']])
    ax.gridlines(xlocs = X , ylocs = Y , color = '#bbbbbb' , linewidth = 0.5 , alpha = 0.5 , draw_labels = False)
    ax.add_feature(shpft(shprd(shpPath).geometries() , ccrs.PlateCarree() , 
                   facecolor = (1 , 1 , 1 , 0) , edgecolor = (0 , 0 , 0 , 1) , linewidth = 1 , zorder = 11))
    ax.contour(terrain['blon'] , terrain['blat'] , terrain['QPEterrain'] , levels = [500 , 1500 , 3000] , colors = '#C0C0C0' , linewidths = [0.5 , 1 , 1.5] , zorder = 10)
    ax.scatter(staInfo['lon'] , staInfo['lat'] , s = 50 , c = 'k' , marker = '^')
    plt.xticks(X , XStr , size = 10)
    plt.yticks(Y , YStr , size = 10)
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(levels , cmap.N)
    # var['data'][var['data'] > cmax] = np.nan
    # var['data'][var['data'] < cmin] = np.nan
    PC = ax.pcolormesh(Lon , Lat , var['data'] , shading = 'flat' , cmap = cmap , norm = norm , alpha = 1)
    PC.set_clim(cmin , cmax)
    cbar = plt.colorbar(PC , orientation = 'vertical' , ticks = ticks , extend = 'both')
    cbar.ax.set_yticklabels(tickLabels)
    cbar.ax.tick_params(labelsize = 12)
    cbar.set_label(var['units'] , size = 12)
    fig.savefig(OUTPATH , dpi = 200)
    print(f"{OUTPATH} - Save!")

def plot_rhi(axis , XEEM , ZEEM , var , staInfo , azi , datetimeLST , outDir , band):
    method = 'RHI'
    plot_rhi_cs(axis , XEEM , ZEEM , var , staInfo , azi , datetimeLST , outDir , band , method)

def plot_cs(axis , XEEM , ZEEM , var , staInfo , aziMean , datetimeLST , outDir , band):
    method = 'CS'
    plot_rhi_cs(axis , XEEM , ZEEM , var , staInfo , aziMean , datetimeLST , outDir , band , method)

def plot_rhi_cs_reorder(axis , XG , ZG , var , staInfo , azi , datetimeLST , outDir , band , method):
    var['data'] = var['reorder']
    plot_rhi_cs(axis , XG , ZG , var , staInfo , azi , datetimeLST , outDir , band , method)
    OUTPATH = outDir/f"{var['name']}_{dtdt.strftime(datetimeLST , '%Y%m%d_%H%M%S')}_{azi * 10:04.0f}.dat"
    var['data'].tofile(f'{str(OUTPATH)}')

def plot_rhi_reorder(axis , XG , ZG , var , staInfo , azi , datetimeLST , outDir , band):
    method = 'RHI'
    plot_rhi_cs_reorder(axis , XG , ZG , var , staInfo , azi , datetimeLST , outDir , band , method)

def plot_cs_reorder(axis , XG , ZG , var , staInfo , aziMean , datetimeLST , outDir , band):
    method = 'CS'
    plot_rhi_cs_reorder(axis , XG , ZG , var , staInfo , aziMean , datetimeLST , outDir , band , method)