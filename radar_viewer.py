#!/home/C.cwj/anaconda3/envs/arm_pyart/bin/python
# -*- coding:utf-8 -*-

########################################
########### radar_viewer.py ############
######## Author: Wei-Jhih Chen #########
######### Update: 2022/12/12 ###########
########################################

import sys , time , re , json
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime as dtdt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from threading import Thread
from PyQt5 import QtWidgets , QtCore , QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from gui.template_ui import Ui_MainWindow
from filter import *
from convert_grid import *
from attenuation_correction import *
from read_furuno_wr2100_archive import *

SCAN_TYPE = 'RHI'
STATION_NAME = 'NTU'    # Station Name
PRODUCT_ID = '0092'     # Product number
INEXT = '(*.gz *.rhi)'
INEXT2 = '\.gz|\.rhi'
INEXT3 = '[.gz|.rhi]'

HOMEDIR = Path(r'/home/C.cwj/Radar')
SHP_PATH = HOMEDIR/'Tools'/'shp'/'taiwan_county'/'COUNTY_MOI_1090820.shp'   # TWNcountyTWD97
MAT_PATH = HOMEDIR/'Tools'/'mat'/'QPESUMS_terrain.mat'                      # TWNterrainTWD97
AXIS_PATH = HOMEDIR/'Tools'/'gui'/'axis_rhi.json'

########## Plot Setting ##########
# X_MIN_RHI = 0;          X_MAX_RHI = 40;         X_INT_RHI = 5
# X_MIN_RHI = -40;        X_MAX_RHI = 0;          X_INT_RHI = 5
X_MIN_RHI = -60;        X_MAX_RHI = 60;         X_INT_RHI = 15
Z_MIN_RHI = 0;          Z_MAX_RHI = 20;         Z_INT_RHI = 1
AXIS_RHI = {'xMin': X_MIN_RHI , 'xMax': X_MAX_RHI , 'xInt': X_INT_RHI , 
            'zMin': Z_MIN_RHI , 'zMax': Z_MAX_RHI , 'zInt': Z_INT_RHI}

X_MIN_PPI = 121.3;      X_MAX_PPI = 122.2;      X_INT_PPI = 0.1
Y_MIN_PPI = 24.3;       Y_MAX_PPI = 25.2;       Y_INT_PPI = 0.1
# X_MIN_PPI = 121.5;      X_MAX_PPI = 122.1;      X_INT_PPI = 0.1
# Y_MIN_PPI = 24.5;       Y_MAX_PPI = 25.1;       Y_INT_PPI = 0.1
AXIS_PPI = {'xMin': X_MIN_PPI , 'xMax': X_MAX_PPI , 'xInt': X_INT_PPI , 
            'yMin': Y_MIN_PPI , 'yMax': Y_MAX_PPI , 'yInt': Y_INT_PPI}

########## Reorder Setting ##########
# X_MIN_REORDER = -40 ;   X_MAX_REORDER = 0 ;     X_INT_REORDER = 0.25
X_MIN_REORDER = 0 ;     X_MAX_REORDER = 60 ;    X_INT_REORDER = 0.25
Z_MIN_REORDER = 0 ;     Z_MAX_REORDER = 20 ;    Z_INT_REORDER = 0.25
AXIS_REORDER = {'xMin': X_MIN_REORDER , 'xMax': X_MAX_REORDER , 'xInt': X_INT_REORDER , 
                'zMin': Z_MIN_REORDER , 'zMax': Z_MAX_REORDER , 'zInt': Z_INT_REORDER}

########## Parameters Setting ##########
VAR_IN = ['DBZ' , 'ZDR' , 'PHIDP' , 'KDP' , 'RHOHV' , 'VEL' , 'WIDTH']
VAR_SELECT = ['DBZ' , 'ZDR' , 'PHIDP' , 'KDP' , 'RHOHV' , 'VEL' , 'WIDTH' , 'DBZ_AC' , 'ZDR_AC']
Fields = {'DBZ': {'name': 'DZ' , 'plotname': 'Z$_{HH}$' , 'units': 'dBZ' , 'data': None} , 
       'ZDR': {'name': 'ZD' , 'plotname': 'Z$_{DR}$' , 'units': 'dB' , 'data': None} , 
       'PHIDP': {'name': 'PH' , 'plotname': '$\phi$$_{DP}$' , 'units': 'Deg.' , 'data': None} , 
       'KDP': {'name': 'KD' , 'plotname': 'K$_{DP}$' , 'units': 'Deg. km$^{-1}$' , 'data': None} , 
       'RHOHV': {'name': 'RH' , 'plotname': r'$\rho$$_{HV}$' , 'units': '' , 'data': None} , 
       'VEL': {'name': 'VR' , 'plotname': 'V$_R$' , 'units': 'm s$^{-1}$' , 'data': None} , 
       'WIDTH': {'name': 'SW' , 'plotname': 'SW' , 'units': 'm s$^{-1}$' , 'data': None} , 
       'RRR': {'name': 'RR' , 'plotname': 'RainRate' , 'units': 'mm hr$^{-1}$' , 'data': None} , 
       'QC_INFO': {'name': 'QC' , 'plotname': 'QC Info' , 'units': '' , 'data': None} , 
       'DBZ_AC': {'name': 'DZac' , 'plotname': 'Z$_{HH}$ (AC)' , 'units': 'dBZ' , 'data': None} , 
       'ZDR_AC': {'name': 'ZDac' , 'plotname': 'Z$_{DR}$ (AC)' , 'units': 'dB' , 'data': None}}
DZ_MIN = 0
RH_MIN = 0.7
RH_MAX = 1.1
INVALID = -999

class Main(QtWidgets.QMainWindow , Ui_MainWindow):
    def __init__(self , parent = None):
        super(Main , self).__init__(parent)
        self.setupUi(self)
        self.subWindow = None
        self.inPath = None
        self.index = None
        self.mode = None
        self.reader = None
        self.plotVar = 'DBZ_AC'
        self.setWindowTitle('NTU Radar Viewer')
        self.toolButtonFile.clicked.connect(self.load_file)
        self.actionOpenFile.triggered.connect(self.load_file)
        self.toolButtonDir.clicked.connect(self.load_dir)
        self.actionOpenDir.triggered.connect(self.load_dir)
        self.listView.clicked.connect(self.clicked)

        self.scene = QtWidgets.QGraphicsScene()
        self.scene.setSceneRect(0, 0, 600, 500)
        self.scene.setObjectName("scene")
        self.graphicsView.setScene(self.scene)

        self.textBrowserFile.textChanged.connect(self.update_scene)

        self.axis_path = AXIS_PATH
        self.check_axis_file(self.axis_path)
        self.lineEditXMin.textChanged.connect(self.update_axis)
        self.lineEditXMax.textChanged.connect(self.update_axis)
        self.lineEditXInt.textChanged.connect(self.update_axis)
        self.lineEditZMin.textChanged.connect(self.update_axis)
        self.lineEditZMax.textChanged.connect(self.update_axis)
        self.lineEditZInt.textChanged.connect(self.update_axis)
        self.comboBox.currentIndexChanged.connect(self.update_listview)

        self.radioButtonZhh.clicked.connect(self.update_plotZhh)
        self.radioButtonZdr.clicked.connect(self.update_plotZdr)
        self.radioButtonKdp.clicked.connect(self.update_plotKdp)
        self.radioButtonPhidp.clicked.connect(self.update_plotPhidp)
        self.radioButtonRhohv.clicked.connect(self.update_plotRhohv)
        self.radioButtonVr.clicked.connect(self.update_plotVr)
        self.radioButtonSw.clicked.connect(self.update_plotSw)
        self.radioButtonZhhAC.clicked.connect(self.update_plotZhhAC)
        self.radioButtonZdrAC.clicked.connect(self.update_plotZdrAC)

        # sc = MplCanvas(self, width=5, height=4, dpi=100)
        # sc.axes.plot([0,1,2,3,4], [10,1,20,3,40])
        # self.setCentralWidget(sc)

        # Plot(self.scene , width = 6 , height = 5 , dpi = 100)

    def show_statusbar_current_dir(self , obj , path):
        if isinstance(obj , QtWidgets.QStatusBar) and str(path) != '':
            obj.showMessage(f'目前位於：{path.parent.absolute()}')

    def show_textbrowser_current_path(self , obj , path):
        if isinstance(obj , QtWidgets.QTextBrowser) and str(path) != '':
            obj.setText(str(path))

    def list_times_from_dir(self , dir , ext):
        paths = Path(dir).glob(f'*{ext}')
        return find_volume_scan_times(paths)

    def find_file_time(self , path):
        return dtdt.strptime(Path(path).name[5:20] , '%Y%m%d_%H%M%S')

    def list_files_from_loadfile(self , path , ext):
        path = re.sub(ext.strip('[]') , "" , str(path))
        path = Path(f'{path[:-3]}*')
        dir = path.parent.absolute()
        paths = dir.glob(f'{path.name}{ext}')
        return sorted([str(p.name) for p in paths])

    def list_files_from_datetime(self , dir , dt , ext):
        paths = dir.glob(f"*{dtdt.strftime(dt , '%Y%m%d_%H%M%S')}*{ext}")
        return sorted([str(p.name) for p in paths])

    def find_combobox_index(self , time , times):
        for i , t in enumerate(times):
            if time == t:
                return i

    def list_combobox(self , obj , strList , index):
        if isinstance(obj , QtWidgets.QComboBox):
            if obj.currentText() != '':
                obj.clear()
            obj.addItems(list(strList))
            obj.setCurrentIndex(index)

    def list_view(self , obj , strList):
        if isinstance(obj , QtWidgets.QListView):
            slm = QtCore.QStringListModel()
            slm.setStringList(list(strList))
            obj.setModel(slm)

    def load_file(self):
        self.mode = 'rhi'
        self.inPath , NULL = QtWidgets.QFileDialog.getOpenFileName(self , "開啟檔案" , "./" , INEXT)
        self.inPath = Path(self.inPath)
        self.inDir = self.inPath.parent.absolute()

        self.show_textbrowser_current_path(self.textBrowserFile , self.inPath)
        self.show_statusbar_current_dir(self.statusbar , self.inPath)
        self.selTime = self.find_file_time(self.inPath)
        self.selTimes = self.list_times_from_dir(self.inDir , INEXT3)
        self.inFilesStr = self.list_files_from_loadfile(self.inPath , INEXT3)
        self.comboBoxIdx = self.find_combobox_index(self.selTime , self.selTimes)
        self.selTimesStr = [dtdt.strftime(t , '%Y/%m/%d %H:%M:%S') for t in self.selTimes]
        self.list_combobox(self.comboBox , self.selTimesStr , self.comboBoxIdx)
        self.list_view(self.listView , self.inFilesStr)

    def load_dir(self):
        self.inDir = QtWidgets.QFileDialog.getExistingDirectory(self , "開啟資料夾" , "./")
        self.inDir = Path(self.inDir)
        if str(self.inDir) != '':
            self.textBrowserDir.setText(str(self.inDir))
            self.statusBar().showMessage(f'目前位於：{self.inDir}')

        CASE_DATE = self.inDir.name
        INPATHS = self.inDir.glob(f'{PRODUCT_ID}_{CASE_DATE}_*[{INEXT2}]')
        self.selTimes = find_volume_scan_times(INPATHS)
        self.selTimesStr = [dtdt.strftime(t , '%Y/%m/%d %H:%M:%S') for t in self.selTimes]
        self.list_view(self.listView , self.selTimesStr)
        self.mode = 'cv'

    def read_axis_file(self , fpath):
        with open(fpath) as f:
            self.axis = json.load(f)
            self.lineEditXMin.setText(str(self.axis['xMin']))
            self.lineEditXMax.setText(str(self.axis['xMax']))
            self.lineEditXInt.setText(str(self.axis['xInt']))
            self.lineEditZMin.setText(str(self.axis['zMin']))
            self.lineEditZMax.setText(str(self.axis['zMax']))
            self.lineEditZInt.setText(str(self.axis['zInt']))

    def check_axis_file(self , fpath):
        if fpath.exists():
            self.read_axis_file(fpath)

    def update_axis_file(self , fpath):
        with open(fpath , 'w') as f:
            json.dump(self.axis , f , indent = 4)

    def update_axis(self):
        self.read_axis()
        self.update_axis_file(self.axis_path)
        if self.mode == 'rhi':
            if self.index is not None:
                self.show_rhi(self.inFilesStr[self.index.row()])
                return
            if self.inPath is not None:
                self.show_rhi(self.inPath.name)

    def update_listview(self):
        idx = self.comboBox.currentIndex()
        self.inFilesStr = self.list_files_from_datetime(self.inDir , self.selTimes[idx] , INEXT3)
        self.list_view(self.listView , self.inFilesStr)

    def update_scene(self):
        if self.mode == 'rhi':
            self.show_rhi(self.inPath.name)

    def clicked(self , index):
        self.index = index
        if self.mode == 'cv':
            self.show_cv(self.selTimes[index.row()])
        elif self.mode == 'rhi':
            self.show_rhi(self.inFilesStr[index.row()])

    # def keyPressEvent(self , e):
    #     print(e)

    def read_axis(self):
        try:
            self.xMin = float(self.lineEditXMin.text())
            self.xMax = float(self.lineEditXMax.text())
            self.xInt = float(self.lineEditXInt.text())
            self.zMin = float(self.lineEditZMin.text())
            self.zMax = float(self.lineEditZMax.text())
            self.zInt = float(self.lineEditZInt.text())
        except ValueError:
            self.check_axis_file(self.axis_path)
        self.axis = {'xMin': self.xMin , 'xMax': self.xMax , 'xInt': self.xInt , 
                     'zMin': self.zMin , 'zMax': self.zMax , 'zInt': self.zInt}

    def update_plotZhh(self):
        self.plotVar = 'DBZ'
        if self.reader:
            Plot_RHI(self.scene , self.reader , self.plotVar , self.axis , width = 6 , height = 5 , dpi = 100 , size = 6)
    def update_plotZdr(self):
        self.plotVar = 'ZDR'
        if self.reader:
            Plot_RHI(self.scene , self.reader , self.plotVar , self.axis , width = 6 , height = 5 , dpi = 100 , size = 6)
    def update_plotKdp(self):
        self.plotVar = 'KDP'
        if self.reader:
            Plot_RHI(self.scene , self.reader , self.plotVar , self.axis , width = 6 , height = 5 , dpi = 100 , size = 6)
    def update_plotPhidp(self):
        self.plotVar = 'PHIDP'
        if self.reader:
            Plot_RHI(self.scene , self.reader , self.plotVar , self.axis , width = 6 , height = 5 , dpi = 100 , size = 6)
    def update_plotRhohv(self):
        self.plotVar = 'RHOHV'
        if self.reader:
            Plot_RHI(self.scene , self.reader , self.plotVar , self.axis , width = 6 , height = 5 , dpi = 100 , size = 6)
    def update_plotVr(self):
        self.plotVar = 'VEL'
        if self.reader:
            Plot_RHI(self.scene , self.reader , self.plotVar , self.axis , width = 6 , height = 5 , dpi = 100 , size = 6)
    def update_plotSw(self):
        self.plotVar = 'WIDTH'
        if self.reader:
            Plot_RHI(self.scene , self.reader , self.plotVar , self.axis , width = 6 , height = 5 , dpi = 100 , size = 6)
    def update_plotZhhAC(self):
        self.plotVar = 'DBZ_AC'
        if self.reader:
            Plot_RHI(self.scene , self.reader , self.plotVar , self.axis , width = 6 , height = 5 , dpi = 100 , size = 6)
    def update_plotZdrAC(self):
        self.plotVar = 'ZDR_AC'
        if self.reader:
            Plot_RHI(self.scene , self.reader , self.plotVar , self.axis , width = 6 , height = 5 , dpi = 100 , size = 6)

    def show_rhi(self , fileStr):
        self.inPath2 = self.inDir/fileStr
        self.statusBar().showMessage(f'處理中：{fileStr}')
        print(f"{fileStr} - Processing...")
        runtime = time.time()
        self.reader = Reader(self.inPath2)
        Plot_RHI(self.scene , self.reader , self.plotVar , self.axis , width = 6 , height = 5 , dpi = 100 , size = 6)
        self.show_button()
        runtime = time.time() - runtime
        self.statusBar().showMessage(f'已完成：{fileStr}（經{runtime:.0f}秒）')
        print(f"{fileStr} - Finish Plotting! (Runtime: {runtime} sec(s))")

        if self.subWindow:
            Plot_RHI(self.subWindow.scene1 , self.reader , 'DBZ_AC' , self.axis , width = 4 , height = 3 , dpi = 100 , size = 4)
            Plot_RHI(self.subWindow.scene2 , self.reader , 'ZDR_AC' , self.axis , width = 4 , height = 3 , dpi = 100 , size = 4)
            Plot_RHI(self.subWindow.scene3 , self.reader , 'KDP' , self.axis , width = 4 , height = 3 , dpi = 100 , size = 4)
            Plot_RHI(self.subWindow.scene4 , self.reader , 'RHOHV' , self.axis , width = 4 , height = 3 , dpi = 100 , size = 4)
            Plot_RHI(self.subWindow.scene5 , self.reader , 'VEL' , self.axis , width = 4 , height = 3 , dpi = 100 , size = 4)
            Plot_RHI(self.subWindow.scene6 , self.reader , 'WIDTH' , self.axis , width = 4 , height = 3 , dpi = 100 , size = 4)

    def show_cv(self , sel_time):
        timeStr = dtdt.strftime(sel_time , '%Y/%m/%d %H:%M:%S')
        self.statusBar().showMessage(f'處理中：{timeStr}')
        print(f"{timeStr} - Processing...")
        runtime = time.time()
        Plot(self.scene , self.inDir , sel_time , 'DBZ_AC' , width = 6 , height = 5 , dpi = 100 , size = 6)
        runtime = time.time() - runtime
        self.statusBar().showMessage(f'已完成：{timeStr}（經{runtime:.0f}秒）')
        print(f"{timeStr} - Finish Plotting! (Runtime: {runtime} sec(s))")

    def show_button(self):
        self.pushButton = QtWidgets.QPushButton(self.frame)
        self.pushButton.setGeometry(QtCore.QRect(500, 470, 100, 30))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.setText("綜合比較")
        self.pushButton.show()
        self.pushButton.clicked.connect(self.clicked_open)

    def clicked_open(self):
        if self.subWindow is None:
            self.subWindow = Sub()
            Plot_RHI(self.subWindow.scene1 , self.reader , 'DBZ_AC' , self.axis , width = 4 , height = 3 , dpi = 100 , size = 4)
            Plot_RHI(self.subWindow.scene2 , self.reader , 'ZDR_AC' , self.axis , width = 4 , height = 3 , dpi = 100 , size = 4)
            Plot_RHI(self.subWindow.scene3 , self.reader , 'KDP' , self.axis , width = 4 , height = 3 , dpi = 100 , size = 4)
            Plot_RHI(self.subWindow.scene4 , self.reader , 'RHOHV' , self.axis , width = 4 , height = 3 , dpi = 100 , size = 4)
            Plot_RHI(self.subWindow.scene5 , self.reader , 'VEL' , self.axis , width = 4 , height = 3 , dpi = 100 , size = 4)
            Plot_RHI(self.subWindow.scene6 , self.reader , 'WIDTH' , self.axis , width = 4 , height = 3 , dpi = 100 , size = 4)
            self.subWindow.show()

class Reader():
    def __init__(self , inPath):
        (self.datetime , NULL , NULL , NULL , 
         self.LATITUDE , self.LONGITUDE , self.ALTITUDE , 
         NULL , NULL , self.fixed_angle , 
         NULL , NULL , 
         self.range , self.azimuth , self.elevation , fields) = reader_corrected_by_radar_constant(inPath)

        self.fields = Fields
        for var in VAR_IN:
            self.fields[var]['data'] = fields[var]['data']
        
        ########## Filters ##########
        for var in ['DBZ' , 'ZDR' , 'PHIDP' , 'KDP' , 'RHOHV' , 'VEL']:
            self.fields[var]['data'] = var_filter(self.fields['DBZ']['data'] , self.fields[var]['data'] , DZ_MIN , None)
        for var in ['DBZ' , 'ZDR' , 'PHIDP' , 'KDP' , 'VEL']:
            self.fields[var]['data'] = var_filter(self.fields['RHOHV']['data'] , self.fields[var]['data'] , RH_MIN , RH_MAX)
        self.fields['DBZ_AC']['data'] , self.fields['ZDR_AC']['data'] = attenuation_correction_X(self.fields['DBZ']['data'] , self.fields['ZDR']['data'] , self.fields['KDP']['data'] , self.range['data'][1] - self.range['data'][0])

class Plot():
    def __init__(self , scene , dir , sel_time , var , parent = None , width = 6 , height = 5 , dpi = 100 , size = 6):
        (datetimes , NULL , NULL , NULL , 
         LATITUDE , LONGITUDE , ALTITUDE , 
         NULL , NULL , Fixed_angle , 
         Sweep_start_ray_index , Sweep_end_ray_index , 
         Range , Azimuth , Elevation , Fields) = read_cv(dir , PRODUCT_ID , sel_time , f'[{INEXT2}]')

        datetimeLST = datetimes['data'][0] + dt.timedelta(hours = 8)
        Fixed_angle = Fixed_angle['data']
        Sweep_start_ray_index = Sweep_start_ray_index['data']
        Sweep_end_ray_index = Sweep_end_ray_index['data']
        Range = Range['data']
        Elevation = Elevation['data']

        # ########## Plot CV ##########
        LonCV , LatCV , NULL = radial_CV_grid(AXIS_REORDER , Fixed_angle , LONGITUDE['data'] , LATITUDE['data'])
        Fields['DBZ_AC']['data'] = radial_CV(AXIS_REORDER , ALTITUDE['data'] , Fixed_angle , Sweep_start_ray_index , Sweep_end_ray_index , Range , Elevation , Fields['DBZ_AC']['data'])
        
        STA_INFO = {'name' : STATION_NAME , 'lon' : LONGITUDE['data'] , 'lat' : LATITUDE['data'] , 'alt' : ALTITUDE['data'] , 'scn' : SCAN_TYPE}
        fig = plot_cv(AXIS_PPI , LonCV , LatCV , Fields[var] , STA_INFO , datetimeLST , SHP_PATH , MAT_PATH , width = width , height = height , dpi = dpi , size = size)

        # fig = plot_test(width = 6 , height = 5 , dpi = 100)

        canvas = FigureCanvasQTAgg(fig)
        proxy_widget = scene.addWidget(canvas)

class Plot_RHI():
    def __init__(self , scene , reader , var , axis , parent = None , width = 6 , height = 5 , dpi = 100 , size = 6):
        datetimeLST = reader.datetime['data'] + dt.timedelta(hours = 8)
        fixed_angle = reader.fixed_angle['data']
        range = reader.range['data']
        elevation = reader.elevation['data']

        for key in VAR_IN:
            Fields[key]['data'] = reader.fields[key]['data']
            Fields[key]['units'] = reader.fields[key]['units']
        
        range = np.append(range , range[-1] + (range[-1] - range[-2]))     # Units: km
        elevation = np.append(elevation - np.append(elevation[1] - elevation[0] , elevation[1:] - elevation[:-1]) / 2 , elevation[-1] + (elevation[-1] - elevation[-2]) / 2)
        Dis , Hgt = equivalent_earth_model_by_elevations(elevation , range , reader.ALTITUDE['data'])

        STA_INFO = {'name' : STATION_NAME , 'lon' : reader.LONGITUDE['data'] , 'lat' : reader.LATITUDE['data'] , 'alt' : reader.ALTITUDE['data'] , 'scn' : SCAN_TYPE}
        fig = plot_rhi(axis , Dis , Hgt , Fields[var] , STA_INFO , fixed_angle , datetimeLST , 'X' , width = width , height = height , dpi = dpi , size = size)

        canvas = FigureCanvasQTAgg(fig)
        proxy_widget = scene.addWidget(canvas)

class Ui_SubWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("SubWindow")
        MainWindow.resize(1280, 720)
        MainWindow.setWindowTitle('綜合比較 - NTU Radar Viewer')
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        MainWindow.setCentralWidget(self.centralwidget)
        self.graphicsView1 = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView1.setGeometry(QtCore.QRect(25, 50, 400, 300))
        self.graphicsView1.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.graphicsView1.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.graphicsView1.setLineWidth(1)
        self.graphicsView1.setObjectName("graphicsView1")
        self.graphicsView2 = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView2.setGeometry(QtCore.QRect(440, 50, 400, 300))
        self.graphicsView2.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.graphicsView2.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.graphicsView2.setLineWidth(1)
        self.graphicsView2.setObjectName("graphicsView2")
        self.graphicsView3 = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView3.setGeometry(QtCore.QRect(855, 50, 400, 300))
        self.graphicsView3.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.graphicsView3.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.graphicsView3.setLineWidth(1)
        self.graphicsView3.setObjectName("graphicsView3")
        self.graphicsView4 = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView4.setGeometry(QtCore.QRect(25, 370, 400, 300))
        self.graphicsView4.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.graphicsView4.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.graphicsView4.setLineWidth(1)
        self.graphicsView4.setObjectName("graphicsView4")
        self.graphicsView5 = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView5.setGeometry(QtCore.QRect(440, 370, 400, 300))
        self.graphicsView5.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.graphicsView5.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.graphicsView5.setLineWidth(1)
        self.graphicsView5.setObjectName("graphicsView5")
        self.graphicsView6 = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView6.setGeometry(QtCore.QRect(855, 370, 400, 300))
        self.graphicsView6.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.graphicsView6.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.graphicsView6.setLineWidth(1)
        self.graphicsView6.setObjectName("graphicsView6")

        self.scene1 = QtWidgets.QGraphicsScene()
        self.scene1.setSceneRect(0, 0, 400, 300)
        self.scene1.setObjectName("scene1")
        self.graphicsView1.setScene(self.scene1)
        self.scene2 = QtWidgets.QGraphicsScene()
        self.scene2.setSceneRect(0, 0, 400, 300)
        self.scene2.setObjectName("scene2")
        self.graphicsView2.setScene(self.scene2)
        self.scene3 = QtWidgets.QGraphicsScene()
        self.scene3.setSceneRect(0, 0, 400, 300)
        self.scene3.setObjectName("scene3")
        self.graphicsView3.setScene(self.scene3)
        self.scene4 = QtWidgets.QGraphicsScene()
        self.scene4.setSceneRect(0, 0, 400, 300)
        self.scene4.setObjectName("scene4")
        self.graphicsView4.setScene(self.scene4)
        self.scene5 = QtWidgets.QGraphicsScene()
        self.scene5.setSceneRect(0, 0, 400, 300)
        self.scene5.setObjectName("scene5")
        self.graphicsView5.setScene(self.scene5)
        self.scene6 = QtWidgets.QGraphicsScene()
        self.scene6.setSceneRect(0, 0, 400, 300)
        self.scene6.setObjectName("scene6")
        self.graphicsView6.setScene(self.scene6)

class Sub(QtWidgets.QMainWindow , Ui_SubWindow):
    def __init__(self , parent = None):
        super(Sub , self).__init__(parent)
        self.setupUi(self)

class SignalStore(QtCore.QObject):
    progress_update = QtCore.pyqtSignal(int)


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

def read_cv(dir: str , id: str , dt: dtdt , ext: str) -> tuple[dict[str , dtdt] , dict[str , str] , dict[str , float] , str , 
        dict[str , float] , dict[str , float] , dict[str , float] , 
        dict[str , int] , dict[str , str] , dict[str , float] , 
        dict[str , int] , dict[str , int] , 
        dict[str , np.ndarray] , dict[str , np.ndarray] , dict[str , np.ndarray] , dict[str , dict[str , Any]]]:

    files = find_volume_scan_files(dir , id , dt , ext)
    num_file = len(files)
    datetimes = {}
    datetimes['data'] = np.empty((num_file) , dtype = dtdt)
    sweep_numbers = np.zeros((num_file) , dtype = int)
    fixed_angles = np.zeros((num_file))
    sweep_start_ray_indexes = np.zeros((num_file) , dtype = int)
    sweep_end_ray_indexes = np.zeros((num_file) , dtype = int)

    (datetime , metadata , instrument_parameters , scan_type , 
     latitude , longitude , altitude , 
     sweep_number , sweep_mode , fixed_angle , 
     sweep_start_ray_index , sweep_end_ray_index , 
     range , azimuth , elevation , fields) = reader_corrected_by_radar_constant(files[0])

    for key in ['PHIDP' , 'RHOHV' , 'VEL' , 'WIDTH' , 'RRR' , 'QC_INFO']:
        del fields[key]

    datetimes['data'][0] = datetime['data']
    datetimes['units'] = datetime['units']
    sweep_numbers[0] = sweep_number['data']
    fixed_angles[0] = fixed_angle['data']
    sweep_start_ray_indexes[0] = sweep_start_ray_index['data']
    sweep_end_ray_indexes[0] = sweep_end_ray_index['data']
    azimuths = azimuth['data']
    elevations = elevation['data']
    for key in ['DBZ' , 'ZDR' , 'KDP']:
        Fields[key]['data'] = fields[key]['data']
        Fields[key]['units'] = fields[key]['units']

    for cnt_file in np.arange(1 , num_file):
        (datetime , NULL , NULL , NULL , 
         NULL , NULL , NULL , 
         sweep_number , NULL , fixed_angle , 
         sweep_start_ray_index , sweep_end_ray_index , 
         NULL , azimuth , elevation , fields) = reader_corrected_by_radar_constant(files[cnt_file])

        for key in ['PHIDP' , 'RHOHV' , 'VEL' , 'WIDTH' , 'RRR' , 'QC_INFO']:
            del fields[key]

        datetimes['data'][cnt_file] = datetime['data']
        sweep_numbers[cnt_file] = sweep_number['data']
        fixed_angles[cnt_file] = fixed_angle['data']
        sweep_start_ray_indexes[cnt_file] = sweep_end_ray_indexes[cnt_file - 1] + 1
        sweep_end_ray_indexes[cnt_file] = sweep_end_ray_indexes[cnt_file - 1] + sweep_end_ray_index['data'] + 1
        azimuths = np.hstack((azimuths , azimuth['data']))
        elevations = np.hstack((elevations , elevation['data']))
        for key in ['DBZ' , 'ZDR' , 'KDP']:
            Fields[key]['data'] = np.vstack((Fields[key]['data'] , fields[key]['data']))

    sweep_numbers = {'data': sweep_numbers}
    fixed_angles = {'data': fixed_angles}
    sweep_start_ray_indexes = {'data': sweep_start_ray_indexes}
    sweep_end_ray_indexes = {'data': sweep_end_ray_indexes}
    azimuths = {'data': azimuths}
    elevations = {'data': elevations}

    Fields['DBZ_AC']['data'] , NULL = attenuation_correction_X(Fields['DBZ']['data'] , Fields['ZDR']['data'] , Fields['KDP']['data'] , range['data'][1] - range['data'][0])

    return (datetimes , metadata , instrument_parameters , scan_type , 
            latitude , longitude , altitude , 
            sweep_numbers , sweep_mode , fixed_angles , 
            sweep_start_ray_indexes , sweep_end_ray_indexes , 
            range , azimuths , elevations , Fields)

def plot_rhi(axis , X , Z , var , staInfo , azi , datetimeLST , band , width = 6 , height = 5 , dpi = 100 , size = 6):
    import cfg.color as cfgc
    from matplotlib.colors import ListedColormap , BoundaryNorm

    ########## Grid ##########
    xTick = np.arange(axis['xMin'] * 10 , axis['xMax'] * 10 + axis['xInt'] * 10 , axis['xInt'] * 10) / 10
    zTick = np.arange(axis['zMin'] * 10 , axis['zMax'] * 10 + axis['zInt'] * 10 , axis['zInt'] * 10) / 10
    ########## Color ##########
    colors , levels , ticks , tickLabels , cmin , cmax = cfgc.colors(var['name'] , band)
    ########## Plot ##########
    plt.close()
    fig , ax = plt.subplots(figsize = (width , height))
    fig.set_dpi(dpi)
    ax.text(0.125 , 0.930 , staInfo['name'] , fontsize = size + 4 , ha = 'left' , color = '#666666' , transform = fig.transFigure)
    ax.text(0.125 , 0.890 , var['plotname'] , fontsize = size + 6 , ha = 'left' , transform = fig.transFigure)
    ax.text(0.744 , 0.930 , f'Azi. {azi:.2f}$^o$ RHI' , fontsize = size + 4 , ha = 'right' , transform = fig.transFigure)
    ax.text(0.744 , 0.890 , dtdt.strftime(datetimeLST , '%Y/%m/%d %H:%M:%S LST') , fontsize = size + 6 , ha = 'right' , transform = fig.transFigure)
    ax.axis([axis['xMin'] , axis['xMax'] , axis['zMin'] , axis['zMax']])
    ax.scatter(0 , staInfo['alt'] , s = size * 20 , c = 'k' , marker = '^')
    if azi >= 180:
        ax.invert_xaxis()
    plt.xticks(xTick , size = size + 2)
    plt.yticks(zTick , size = size + 2)
    plt.xlabel('Distance from Radar (km)' , size = size + 4)
    plt.ylabel('Altitude (km)' , size = size + 4)
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(levels , cmap.N)
    PC = ax.pcolormesh(X , Z , var['data'] , shading = 'flat' , cmap = cmap , norm = norm , alpha = 1)
    PC.set_clim(cmin , cmax)
    ax.grid(visible = True , color = '#bbbbbb' , linewidth = .5 , alpha = .5 , zorder = 0)
    cbar = plt.colorbar(PC , orientation = 'vertical' , ticks = ticks , extend = 'both')
    cbar.ax.set_yticklabels(tickLabels , fontdict = {'fontsize': size + 2})
    cbar.set_label(var['units'] , size = size + 4)
    return fig

def plot_cv(axis , Lon , Lat , var , staInfo , datetimeLST , shpPath , matPath , width = 6 , height = 5 , dpi = 100 , size = 6):
    import cfg.color as cfgc
    import cartopy.crs as ccrs
    from scipy import io
    from cartopy.io.shapereader import Reader as shprd
    from cartopy.feature import ShapelyFeature as shpft
    from matplotlib.colors import ListedColormap , BoundaryNorm

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
    colors , levels , ticks , tickLabels , cmin , cmax = cfgc.colors(var['name'] , 'S')
    ########## Terrain ##########
    terrain = io.loadmat(matPath)
    ########## Plot ##########
    plt.close()
    fig , ax = plt.subplots(figsize = (width , height) , subplot_kw = {'projection' : ccrs.PlateCarree()})
    fig.set_dpi(dpi)
    ax.text(0.125 , 0.905 , staInfo['name'] , fontsize = size + 4 , ha = 'left' , color = '#666666' , transform = fig.transFigure)
    ax.text(0.125 , 0.875 , var['plotname'] , fontsize = size + 6 , ha = 'left' , transform = fig.transFigure)
    ax.text(0.744 , 0.905 , 'Composite ' + staInfo['scn'].upper() , fontsize = size + 4 , ha = 'right' , transform = fig.transFigure)
    ax.text(0.744 , 0.875 , dtdt.strftime(datetimeLST , '%Y/%m/%d %H:%M:%S LST') , fontsize = size + 6 , ha = 'right' , transform = fig.transFigure)
    ax.set_extent([axis['xMin'] , axis['xMax'] , axis['yMin'] , axis['yMax']])
    ax.gridlines(xlocs = X , ylocs = Y , color = '#bbbbbb' , linewidth = 0.5 , alpha = 0.5 , draw_labels = False)
    ax.add_feature(shpft(shprd(shpPath).geometries() , ccrs.PlateCarree() , 
                facecolor = (1 , 1 , 1 , 0) , edgecolor = (0 , 0 , 0 , 1) , linewidth = 1 , zorder = 11))
    ax.contour(terrain['blon'] , terrain['blat'] , terrain['QPEterrain'] , levels = [500 , 1500 , 3000] , colors = '#C0C0C0' , linewidths = [0.5 , 1 , 1.5] , zorder = 10)
    ax.scatter(staInfo['lon'] , staInfo['lat'] , s = 50 , c = 'k' , marker = '^')
    plt.xticks(X , XStr , size = size)
    plt.yticks(Y , YStr , size = size)
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(levels , cmap.N)
    PC = ax.pcolormesh(Lon , Lat , var['data'] , shading = 'flat' , cmap = cmap , norm = norm , alpha = 1)
    PC.set_clim(cmin , cmax)
    cbar = plt.colorbar(PC , orientation = 'vertical' , ticks = ticks)
    cbar.ax.set_yticklabels(tickLabels)
    cbar.ax.tick_params(labelsize = size + 2)
    cbar.set_label(var['units'] , size = size + 2)
    return fig

def plot_test(width = 6 , height = 5 , dpi = 100):
    fig , ax = plt.subplots(figsize = (width , height))
    fig.set_dpi(dpi)
    # fig = Figure(figsize = (width , height) , dpi = dpi)
    # ax = fig.gca()
    ax.set_title(f'Plot')
    x = np.linspace(1, 10)
    y = np.linspace(1, 10)
    y1 = np.linspace(11, 20)
    ax.plot(x, y, "-k", label="first one")
    ax.plot(x, y1, "-b", label="second one")
    ax.legend()
    ax.grid(True)
    return fig

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Main()
    window.show()
    sys.exit(app.exec_())
