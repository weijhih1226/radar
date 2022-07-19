import netCDF4 as nc
import cartopy.crs as ccrs
import pyart as pa
import numpy as np
import datetime as dt
import matplotlib
import matplotlib.pyplot as plt
from cartopy.io.shapereader import Reader as shprd
from cartopy.feature import ShapelyFeature as shpft
from matplotlib.colors import ListedColormap , BoundaryNorm

def convert_coordinates(azi , disEEM , hgtEEM , longitude , latitude):
    lon_deg2km = 102.8282                               # Degree to km (Units: km/deg.)
    lat_deg2km = 111.1361                               # Degree to km (Units: km/deg.)

    aziEEM = -(azi - 90) * np.pi / 180
    DisEEM , AziEEM = np.meshgrid(disEEM , aziEEM)      # Distance Meshgrid
    HgtEEM = np.meshgrid(hgtEEM , aziEEM)[0]            # Height Meshgrid
    xEEM = DisEEM * np.cos(AziEEM)
    yEEM = DisEEM * np.sin(AziEEM)
    LonEEM = longitude + xEEM / lon_deg2km
    LatEEM = latitude + yEEM / lat_deg2km
    return LonEEM , LatEEM

########## Read NetCDF4 Files ##########
homeDir = '/home/C.cwj/Radar/'
inPath = f'{homeDir}cases/NC-NTU/20200716/0092_20200716_040230.nc'
datagrp = nc.Dataset(inPath)

########## Shape Files ##########
shpPath = f'{homeDir}Tools/Python/shp/taiwan_county/COUNTY_MOI_1090820.shp'     # TWNcountyTWD97
shp = shpft(shprd(shpPath).geometries() , ccrs.PlateCarree() , 
            facecolor = (1 , 1 , 1 , 0) , edgecolor = (0 , 0 , 0 , 1) , 
            linewidth = 1 , zorder = 10)

# Read Variables
range = datagrp.variables['range'][:] / 1000        # Size: 1004; Units: km
azimuth = datagrp.variables['azimuth'][:]           # Size: 20400; Units: deg
elevation = datagrp.variables['elevation'][:]       # Size: 20400; Units: deg

sweep_number = datagrp.variables['sweep_number'][:] # Azimuth numbers: 0 ~ 59
fixed_angle = datagrp.variables['fixed_angle'][:]   # Azimuth angles; Size: 60
sweep_start_ray_index = datagrp.variables['sweep_start_ray_index'][:]   # Start index of 60 azis
sweep_end_ray_index = datagrp.variables['sweep_end_ray_index'][:]       # End index of 60 azis

time_start = datagrp.variables['time_coverage_start'][:]
time_end = datagrp.variables['time_coverage_end'][:]
time_startStr = nc.chartostring(time_start)
time_endStr = nc.chartostring(time_end)

# Read Observables
varDBZ = datagrp.variables['DBZ'][:]                # Size: azi*ele , rng
# varVEL = datagrp.variables['VEL'][:]
# varZDR = datagrp.variables['ZDR'][:]
# varKDP = datagrp.variables['KDP'][:]
# varPHIDP = datagrp.variables['PHIDP'][:]
# varRHOHV = datagrp.variables['RHOHV'][:]
# varWIDTH = datagrp.variables['WIDTH'][:]
# varQCINFO = datagrp.variables['QC_INFO'][:]
# varRRR = datagrp.variables['RRR'][:]

# Station Location
latitude = datagrp.variables['latitude'][:]
longitude = datagrp.variables['longitude'][:]
altitude = datagrp.variables['altitude'][:] / 1000  # Units: km

# Range & Azimuth & Elevation Numbers
num_azi_ele = varDBZ.shape[0]                       # num_azi_ele: 20400
num_rng = varDBZ.shape[1]                           # num_rng: 1004
num_azi = sweep_number[-1] + 1                      # num_azi: 60
num_ele = int(num_azi_ele / num_azi)                # num_ele: 340

# Select One Azimuth (RHI from 60 Azimuths)
aziRHI = fixed_angle[0]
eleRHI = elevation[sweep_start_ray_index[0] : sweep_end_ray_index[0] + 1]
varDBZ_RHI = varDBZ[sweep_start_ray_index[0] : sweep_end_ray_index[0] + 1 , :]

print(fixed_angle)
print(eleRHI)

# Select One Elevation (Pseudo-PPI from 340 Elevations)
aziPPPI = azimuth[339 : len(azimuth) : num_ele * 2]
elePPPI = elevation[339]
varDBZ_PPPI = varDBZ[339 : len(azimuth) : num_ele * 2]

########## Convert Coordinates ##########
# RHI
Range , Ele = np.meshgrid(range , eleRHI * np.pi / 180.)
xRHI = Range * np.cos(Ele)
yRHI = Range * np.sin(Ele) + altitude

# Pseudo-PPI
Dist , Azi = np.meshgrid(range * np.cos(elePPPI * np.pi / 180.) , 
                         -(aziPPPI - 90) * np.pi / 180.)
xPPPI = Dist * np.cos(Azi)
yPPPI = Dist * np.sin(Azi)

lonPPPI , latPPPI = convert_coordinates(aziPPPI , range * np.cos(elePPPI * np.pi / 180.) , 
                    range * np.sin(elePPPI * np.pi / 180.) , longitude , latitude)

########## Plot ##########
colors = ['#00FFFF','#01A0F6','#0000F6','#00FF00','#00C700',
          '#009000','#E7C000','#FF9000','#FF0000','#D50000',
          '#A60000','#FF00FF','#9954C8','#FFFFFF']
levels = np.arange(0 , 75 , 5)
ticks = np.arange(0 , 70 , 5)
cmap = ListedColormap(colors)
norm = BoundaryNorm(levels , cmap.N)

# RHI
plt.close()
fig , ax = plt.subplots(figsize = [12 , 10])
CTF = ax.contourf(xRHI , yRHI , varDBZ_RHI , levels = levels , cmap = cmap , norm = norm , alpha = 1 , extend = 'both')
cbar = plt.colorbar(CTF , orientation = 'vertical' , ticks = ticks)
cbar.set_label('dBZ')
plt.xlim([-20 , 20])
plt.ylim([0 , 15])
plt.xlabel('Horizontal Distance (km)')
plt.ylabel('Altitude (km)')
plt.title(f'RHI   Equivalent Reflectivity Factor (dBZ)\nAzi: {aziRHI:.2f}' + '$^{o}$   ' + f'Time: {time_startStr}')
plt.show()

# Pseudo-PPI
plt.close()
fig , ax = plt.subplots(figsize = [12 , 10])
CTF = ax.contourf(xPPPI , yPPPI , varDBZ_PPPI , levels = levels , cmap = cmap , norm = norm , alpha = 1 , extend = 'both')
cbar = plt.colorbar(CTF , orientation = 'vertical' , ticks = ticks)
cbar.set_label('dBZ')
plt.xlim([-20 , 20])
plt.ylim([-20 , 20])
plt.xlabel('X Distance (km)')
plt.ylabel('Y Distance (km)')
plt.title(f'Pseudo-PPI   Equivalent Reflectivity Factor (dBZ)\nEle: {elePPPI}' + '$^{o}$   ' + f'Time: {time_startStr}')
plt.show()

# Pseudo-PPI (lon , lat)
plt.close()
fig , ax = plt.subplots(figsize = [12 , 10] , subplot_kw = {'projection' : ccrs.PlateCarree()})
CTF = ax.contourf(lonPPPI , latPPPI , varDBZ_PPPI , levels = levels , cmap = cmap , norm = norm , alpha = 1 , extend = 'both')
cbar = plt.colorbar(CTF , orientation = 'vertical' , ticks = ticks)
cbar.set_label('dBZ')
ax.add_feature(shp)
plt.xticks(np.arange(121.5 , 122.2 , 0.1))
plt.yticks(np.arange(24.5 , 25.2 , 0.1))
plt.xlim([121.5 , 122.1])
plt.ylim([24.5 , 25.1])
plt.title(f'Pseudo-PPI   Equivalent Reflectivity Factor (dBZ)\nEle: {elePPPI}' + '$^{o}$   ' + f'Time: {time_startStr}')
plt.show()


# matplotlib.use('Qt5Agg')
