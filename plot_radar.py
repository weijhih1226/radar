########################################
############ plot_radar.py #############
######## Author: Wei-Jhih Chen #########
######### Update: 2022/07/20 ###########
########################################

import cfg.color as cfgc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap , BoundaryNorm

def plot_ppi(axis , LonEEM , LatEEM , var , varInfo , staInfo , eleFix , datetimeStrLST , shpPath , matPath , outPath):
    import cartopy.crs as ccrs
    from scipy import io
    from cartopy.io.shapereader import Reader as shprd
    from cartopy.feature import ShapelyFeature as shpft

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
    print(f"{outPath} - Save!")

def plot_rhi_cs(axis , XEEM , ZEEM , var , varInfo , staInfo , azi , datetimeStrLST , outPath , band , method):
    ########## Grid ##########
    xTick = np.arange(axis['xMin'] * 10 , axis['xMax'] * 10 + axis['xInt'] * 10 , axis['xInt'] * 10) / 10
    zTick = np.arange(axis['zMin'] * 10 , axis['zMax'] * 10 + axis['zInt'] * 10 , axis['zInt'] * 10) / 10
    ########## Color ##########
    colors , levels , ticks , tickLabels , cmin , cmax = cfgc.colors(varInfo['name'] , band)
    ########## Plot ##########
    plt.close()
    fig , ax = plt.subplots(figsize = [12 , 10])
    ax.text(0.125 , 0.920 , staInfo['name'] , fontsize = 18 , ha = 'left' , color = '#666666' , transform = fig.transFigure)
    ax.text(0.125 , 0.890 , varInfo['plotname'] , fontsize = 20 , ha = 'left' , transform = fig.transFigure)
    ax.text(0.744 , 0.920 , f'Azi. {azi:.2f}$^o$ {method}' , fontsize = 18 , ha = 'right' , transform = fig.transFigure)
    ax.text(0.744 , 0.890 , datetimeStrLST , fontsize = 20 , ha = 'right' , transform = fig.transFigure)
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
    print(f"{outPath} - Save!")

def plot_rhi(axis , XEEM , ZEEM , var , varInfo , staInfo , azi , datetimeStrLST , outPath , band):
    method = 'RHI'
    plot_rhi_cs(axis , XEEM , ZEEM , var , varInfo , staInfo , azi , datetimeStrLST , outPath , band , method)

def plot_cs(axis , XEEM , ZEEM , var , varInfo , staInfo , aziMean , datetimeStrLST , outPath , band):
    method = 'CS'
    plot_rhi_cs(axis , XEEM , ZEEM , var , varInfo , staInfo , aziMean , datetimeStrLST , outPath , band , method)

def plot_rhi_cs_reorder(axis , XG , ZG , varXZ , varInfo , staInfo , azi , datetimeStrLST , outPath , band , method):
    plot_rhi_cs(axis , XG , ZG , varXZ , varInfo , staInfo , azi , datetimeStrLST , outPath , band , method)
    varXZ.tofile(f'{outPath[:-4]}.dat')

def plot_rhi_reorder(axis , XG , ZG , varXZ , varInfo , staInfo , azi , datetimeStrLST , outPath , band):
    method = 'RHI'
    plot_rhi_cs_reorder(axis , XG , ZG , varXZ , varInfo , staInfo , azi , datetimeStrLST , outPath , band , method)

def plot_cs_reorder(axis , XG , ZG , varXZ , varInfo , staInfo , azi , datetimeStrLST , outPath , band):
    method = 'CS'
    plot_rhi_cs_reorder(axis , XG , ZG , varXZ , varInfo , staInfo , azi , datetimeStrLST , outPath , band , method)