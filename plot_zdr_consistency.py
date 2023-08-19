#!/usr/bin/env python3
# -*- coding:utf-8 -*-

########################################
####### plot_zdr_consistency.py ########
######## Author: Wei-Jhih Chen #########
######### Update: 2023/08/19 ###########
########################################

import numpy as np
import matplotlib.pyplot as plt

from read_furuno_wr2100_archive import *
from tools import timer
from tqdm import tqdm

CASE_DATE = '20211126'
# CASE_START = dtdt(2021 , 11 , 26 , 6 , 30 , 32)
# CASE_END = dtdt(2021 , 11 , 26 , 6 , 30 , 32)
CASE_START = dtdt(2021 , 11 , 26 , 6 , 6 , 57)
# CASE_START = dtdt(2021 , 11 , 26 , 6 , 18 , 44)
CASE_END = dtdt(2021 , 11 , 26 , 6 , 30 , 32)

STATION_NAME = 'NTU'    # Station Name
PRODUCT_ID = '0092'     # Product number
BAND = 'X'
SCAN_TYPE = 'RHI'
PLOT_TYPE = 'PPPI'

TICKSIZE = 10
FONTSIZE = 18
FIGSIZE = (12 , 10)
DPI = 200

INEXT = '[.gz|.rhi]'
HOMEDIR = Path(r'/home/C.cwj/Radar')
# HOMEDIR = Path(r'C:\Users\wjchen\Documents\Research\Radar')
INDIR = HOMEDIR/'cases'/f'RAW-{STATION_NAME}'/CASE_DATE
INPATHS = INDIR.glob(f'{PRODUCT_ID}_{CASE_DATE}_*{INEXT}')
OUTDIR = HOMEDIR/'pic'/STATION_NAME/CASE_DATE/'ZDRC'

def set_plot(figsize = FIGSIZE , dpi = DPI):
    plt.close()
    fig , ax = plt.subplots(figsize = figsize)
    fig.set_dpi(dpi)
    return fig , ax

def plot_histogram(x , tmin , tmax , outDir = None , ticksize = TICKSIZE , figsize = FIGSIZE , dpi = DPI):
    fig , ax = set_plot(figsize = figsize , dpi = dpi)
    x_avg = np.nanmean(x)
    x_med = np.nanmedian(x)
    print(x_avg , x_med)
    ax.hist(x , np.arange(-5 , 5.05 , 0.05))
    ylim = ax.get_ylim()
    ax.plot([x_avg , x_avg] , [*ylim] , 'k')
    ax.plot([x_med , x_med] , [*ylim] , 'k')
    ax.set_xlim(-5 , 5)
    ax.set_ylim(*ylim)
    ax.set_xticks(np.arange(-5 , 5.5 , 0.5))
    ax.set_xlabel('Elev. 90$^o$ Z$_{DR}$' , size = ticksize + 2)
    ax.set_ylabel('Counts' , size = ticksize + 2)
    ax.grid(axis = 'x' , color = '#bbbbbb' , linewidth = 0.5 , alpha = 0.5)
    outPath = outDir/f"HIST_ZD_{dtdt.strftime(tmin , '%Y%m%d_%H%M%S')}_{dtdt.strftime(tmax , '%H%M%S')}.png"
    fig.savefig(outPath)
    return fig , ax

def plot_timeseries(x , y , tmin , tmax , outDir = None , ticksize = TICKSIZE , figsize = FIGSIZE , dpi = DPI):
    fig , ax = set_plot(figsize = figsize , dpi = dpi)
    y_avg = np.mean(y)
    print(y_avg)
    ax.plot(x , y)
    ax.plot([x[0] , x[-1]] , [y_avg , y_avg] , 'k')
    ax.set_xlim(x[0] , x[-1])
    ax.set_ylim(-1 , 1)
    ax.set_yticks(np.arange(-1 , 1.2 , 0.2))
    ax.set_xlabel('Time' , size = ticksize + 2)
    ax.set_ylabel('Elev. 90$^o$ Z$_{DR}$' , size = ticksize + 2)
    ax.grid(color = '#bbbbbb' , linewidth = 0.5 , alpha = 0.5)
    outPath = outDir/f"TS_ZD_{dtdt.strftime(tmin , '%Y%m%d_%H%M%S')}_{dtdt.strftime(tmax , '%H%M%S')}.png"
    fig.savefig(outPath)
    return fig , ax

@timer
def main():
    SEL_TIMES = find_volume_scan_times(INPATHS , CASE_START , CASE_END)
    height = np.arange(0 , 3.55 , 0.05)
    datetimes = []
    var_zdrs_avg = []
    var_zdrs_all = []
    for sel_time in tqdm(SEL_TIMES):
        infiles = find_volume_scan_files(INDIR , PRODUCT_ID , sel_time , INEXT)
        for file in tqdm(infiles):
            reader = Reader_Selected_Elevation(file , 90)
            datetimes.append(reader.datetime['data'])
            var_zdrs = reader.get_selected_zdr()['data'][0,:71]
            var_zdrs_avg.append(np.mean(var_zdrs))
            var_zdrs_list = var_zdrs.tolist()
            while None in var_zdrs_list:
                var_zdrs_list.remove(None)
            var_zdrs_all += var_zdrs_list
    plot_timeseries(datetimes , var_zdrs_avg , datetimes[0] , datetimes[-1] , outDir = OUTDIR)
    plot_histogram(var_zdrs_all , datetimes[0] , datetimes[-1] , outDir = OUTDIR)

if __name__ == '__main__':
    main()