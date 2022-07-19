########################################
############### color.py ###############
######## Author: Wei-Jhih Chen #########
######### Update: 2022/03/08 ###########
########################################

import numpy as np

def colors(varName , band):
    if varName == 'DZ' or varName == 'DZac':
        colors = ['#00FFFF','#01A0F6','#0000F6','#00FF00','#00C700','#009000','#E7C000','#FF9000','#FF0000','#D50000',
                 '#A60000','#FF00FF','#9954C8','#FFFFFF']
        levels = np.arange(0 , 75 , 5)
        ticks = np.arange(0 , 70 , 5)
        tickLabels = ticks
        cmin = min(levels)
        # cmax = None
        cmax = max(levels)
    elif varName == 'ZD' or varName == 'ZDac':
        colors = ['#0000BF','#0000FF','#003FFF','#007FFF','#00BFFF','#00FFFF','#3FFFBF','#7FFF7F','#BFFF3F','#FFFF00',
                 '#FFBF00','#FF7F00','#FF3F00','#FF0000']
        levels = np.arange(-2 , 5.5 , .5)
        ticks = levels
        tickLabels = ticks
        cmin = min(levels)
        cmax = max(levels)
    elif varName == 'PH':
        colors = ['#0000AA','#0000FF','#0055FF','#00AAFF','#00FFFF','#55FFAA','#AAFF55','#FFFF00','#FFAA00','#FF5500',
                 '#FF0000','#AA0000']
        levels = np.arange(0 , 195 , 15)
        ticks = levels
        tickLabels = ticks
        cmin = min(levels)
        cmax = max(levels)
    elif varName == 'KD':
        colors = ['#0000FF','#007FFF','#00FFFF','#7FFF7F','#FFFF00','#FF7F00','#FF0000','#7F0000']
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
        colors = ['#000000','#4C4C4C','#666666','#7F7F7F','#999999','#B2B2B2','#FF6666','#FF00FF','#9900FF','#4DDFFF',
                 '#6294EF','#5F54C0','#24FA25','#1AC607','#009200','#FFFF00','#FFCC00','#FF9900','#FF0000','#990000',
                 '#FFFFFF']
        levels = np.arange(0.75 - 0.1 / 3 , 1.05 + 0.05 / 6 + 41 / 2520 , 41 / 2520)
        ticks = np.arange(75 , 110 , 5) / 100
        tickLabels = ticks
        # cmin = None
        # cmax = None
        cmin = min(levels)
        cmax = max(levels)
    elif varName == 'VR':
        colors = ['#0000FF','#0033FF','#0066FF','#0099FF','#00CCFF','#00FFFF','#FFFF00','#FFCC00','#FF9900','#FF6600',
                 '#FF3300','#FF0000']
        levels = np.arange(-18 , 21 , 3)
        ticks = levels
        tickLabels = ticks
        cmin = min(levels)
        cmax = max(levels)
    elif varName == 'SW':
        colors = ['#0000BF','#0000FF','#003FFF','#007FFF','#00BFFF','#00FFFF','#3FFFBF','#7FFF7F','#BFFF3F','#FFFF00',
                 '#FFBF00','#FF7F00','#FF3F00','#FF0000']
        levels = np.arange(0 , 7.5 , .5)
        ticks = levels
        tickLabels = ticks
        cmin = min(levels)
        cmax = max(levels)
    return colors , levels , ticks , tickLabels , cmin , cmax