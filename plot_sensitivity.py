#!/home/C.cwj/anaconda3/envs/pyart_env/bin/python

########################################
######### plot_sensitivity.py ##########
######## Author: Wei-Jhih Chen #########
######### Update: 2022/07/19 ###########
########################################

########## Units Descriptions ##########
# Pr: mW
# wl: cm
# R: km
# Pt: W
# bw_az , bw_el: degree
# tau: us

import numpy as np
import matplotlib.pyplot as plt

# Sensitivity Function
sensitivity = lambda Pr , wl , R , l , Pt , G , bw_az , bw_el , tau , K_2: (Pr * 6.75 * 2 ** 14 * np.log(2) * wl ** 2 * R ** 2 * l) / (np.pi ** 5 * 10 ** -17 * Pt * G ** 2 * bw_az * bw_el * tau * K_2)

Pr_RCWF = 10 ** (-114 / 10)     # mW
wl_RCWF = 10.71                 # cm
R_RCWF = np.linspace(0 , 500)   # km
l_RCWF = 10 ** (0.24 / 10)      # log10 to ratio
Pt_RCWF = 700 * 1000            # W
G_RCWF = 10 ** (45.5 / 10)      # log10 to ratio
bw_az_RCWF = 0.925              # degree
bw_el_RCWF = 0.925              # degree
tau_RCWF_s = 1.57               # us (P0N)
tau_RCWF_l = 4.71               # us (Q0N)

Pr_NTU = 10 ** (-110 / 10)      # mW
wl_NTU = 3.19                   # cm
R_NTU = np.linspace(0 , 70)     # km
l_NTU = 10 ** (0 / 10)          # log10 to ratio
Pt_NTU = 100                    # W
G_NTU = 10 ** (33.0 / 10)       # log10 to ratio
bw_az_NTU = 2.7                 # degree
bw_el_NTU = 2.7                 # degree
tau_NTU_s = 1                   # us (P0N)
tau_NTU_l = 50                  # us (Q0N)

K_2 = 0.93

Ze_RCWF_s = sensitivity(Pr_RCWF , wl_RCWF , R_RCWF , l_RCWF , Pt_RCWF , G_RCWF , bw_az_RCWF , bw_el_RCWF , tau_RCWF_s , K_2)
Ze_RCWF_l = sensitivity(Pr_RCWF , wl_RCWF , R_RCWF , l_RCWF , Pt_RCWF , G_RCWF , bw_az_RCWF , bw_el_RCWF , tau_RCWF_l , K_2)
Ze_NTU_s = sensitivity(Pr_NTU , wl_NTU , R_NTU , l_NTU , Pt_NTU , G_NTU , bw_az_NTU , bw_el_NTU , tau_NTU_s , K_2)
Ze_NTU_l = sensitivity(Pr_NTU , wl_NTU , R_NTU , l_NTU , Pt_NTU , G_NTU , bw_az_NTU , bw_el_NTU , tau_NTU_l , K_2)

plt.close()
fig , ax = plt.subplots()
ax.axis([0 , 70 , -25 , 50])
ax.grid(visible = True , color = '#bbbbbb' , linewidth = .5 , alpha = .5 , zorder = 0)
plt.plot(R_RCWF , 10 * np.log10(Ze_RCWF_s) , 'b--' , R_RCWF , 10 * np.log10(Ze_RCWF_l) , 'b-')
plt.plot(R_NTU , 10 * np.log10(Ze_NTU_s) , 'r--' , R_NTU , 10 * np.log10(Ze_NTU_l) , 'r-')
plt.legend(['WSR-88D (short pulse)' , 'WSR-88D (long pulse)' , 'WR-2100 (short pulse)' , 'WR-2100 (long pulse)'] , fontsize = 6)
plt.xticks(np.arange(0 , 75 , 5))
plt.yticks(np.arange(-25 , 55 , 5))
plt.xlabel('Range (km)')
plt.ylabel('dBZ')
plt.title('Sensitivity between WSR-88D & WR-2100')
fig.savefig('../pic/Sensitivity.png' , dpi = 200)
plt.show()