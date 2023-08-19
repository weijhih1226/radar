#!/usr/bin/env python3
# -*- coding:utf-8 -*-

########################################
############# read_color.py ############
######## Author: Wei-Jhih Chen #########
######### Update: 2022/08/19 ###########
########################################

import sys
import numpy as np
from pathlib import Path

COLOR_DIR = Path(__file__).parent.parent / 'cfg' / 'color'
COLOR_PATH_DZ = COLOR_DIR / 'dbz.color'
COLOR_PATH_ZD = COLOR_DIR / 'zdr.color'
COLOR_PATH_PH = COLOR_DIR / 'phidp.color'
COLOR_PATH_KD = COLOR_DIR / 'kdp.color'
COLOR_PATH_RH = COLOR_DIR / 'rhohv.color'
COLOR_PATH_VR = COLOR_DIR / 'vr.color'
COLOR_PATH_SW = COLOR_PATH_ZD

def read_hex_color_file(filepath):
    with open(filepath) as f:
        ls = [l.strip() for l in f]
    return ls

def colors(varName , band):
    if varName == 'DZ' or varName == 'DZac':
        colors = read_hex_color_file(COLOR_PATH_DZ)
        levels = np.arange(0 , 75 , 5)
        ticks = np.arange(0 , 70 , 5)
        tickLabels = ticks
        cmin = min(levels)
        # cmax = None
        cmax = max(levels)
    elif varName == 'ZD' or varName == 'ZDac':
        colors = read_hex_color_file(COLOR_PATH_ZD)
        levels = np.arange(-2 , 5.5 , .5)
        ticks = levels
        tickLabels = ticks
        cmin = min(levels)
        cmax = max(levels)
    elif varName == 'PH':
        colors = read_hex_color_file(COLOR_PATH_PH)
        levels = np.arange(0 , 195 , 15)
        ticks = levels
        tickLabels = ticks
        cmin = min(levels)
        cmax = max(levels)
    elif varName == 'KD':
        colors = read_hex_color_file(COLOR_PATH_KD)
        if band == 'S':
            levels = np.arange(-0.5 , 4 , .5)
            ticks = np.arange(0 , 3.5 , .5)
        elif band == 'X':
            levels = np.arange(-1 , 8 , 1)
            ticks = np.arange(0 , 7 , 1)
        tickLabels = ticks
        # cmin = None
        # cmax = None
        cmin = min(levels)
        cmax = max(levels)
    elif varName == 'RH':
        colors = read_hex_color_file(COLOR_PATH_RH)
        levels = np.arange(0.75 - 0.1 / 3 , 1.05 + 0.05 / 6 + 41 / 2520 , 41 / 2520)
        ticks = np.arange(75 , 110 , 5) / 100
        tickLabels = ticks
        # cmin = None
        # cmax = None
        cmin = min(levels)
        cmax = max(levels)
    elif varName == 'VR':
        colors = read_hex_color_file(COLOR_PATH_VR)
        levels = np.arange(-18 , 21 , 3)
        ticks = levels
        tickLabels = ticks
        cmin = min(levels)
        cmax = max(levels)
    elif varName == 'SW':
        colors = read_hex_color_file(COLOR_PATH_SW)
        levels = np.arange(0 , 7.5 , .5)
        ticks = levels
        tickLabels = ticks
        cmin = min(levels)
        cmax = max(levels)
    return colors , levels , ticks , tickLabels , cmin , cmax

def main(varName = sys.argv[1]):
    varName = varName.lower()
    if varName in ('db' , 'dz' , 'dbz' , 'zh' , 'zv' , 'zhh' , 'zvv' , 'zhv' , 'dzac'):
        varName = 'DZ'
    elif varName in ('zd' , 'zdr' , 'zdac'):
        varName = 'ZD'
    elif varName in ('ph' , 'phi' , 'phidp'):
        varName = 'PH'
    elif varName in ('kd' , 'kdp'):
        varName = 'KD'
    elif varName in ('rh' , 'rho' , 'rhohv'):
        varName = 'RH'
    elif varName in ('vr' , 'v'):
        varName = 'VR'
    elif varName in ('sw' , 'w'):
        varName = 'SW'
    color = colors(varName , 'S')[0]
    print(color)

if __name__ == '__main__':
    main()