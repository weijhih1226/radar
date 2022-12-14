# -*- coding:utf-8 -*-

########################################
############### tools.py ###############
######## Author: Wei-Jhih Chen #########
######### Update: 2022/12/14 ###########
########################################

import time

def timer(func):
    def wrap():
        t_start = time.time()
        print('Processing...')
        func()
        t_end = time.time()
        t_count = t_end - t_start
        print(f'Runtime: {t_count} second(s) - Finish!')
    return wrap