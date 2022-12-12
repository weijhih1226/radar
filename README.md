# radar

## 畫圖程式

### Radar Viewer（GUI介面）
- radar_viewer.py - 目前可讀.rhi及.rhi.gz

### 臺大雷達（二進位檔格式）
- plot_furuno_wr2100_archive.py - 直接讀二進位原始檔

### 五分山雷達（Level II格式）
- plot_nexrad_levelII.py - 使用ARM Py-ART套件讀檔

---
## 處理雷達資料函式

### 臺大雷達讀檔
- read_furuno_wr2100_archive.py - 讀二進位檔

### 畫圖型態
- plot_radar.py - PPI、RHI、Cross Section(CS)
- plot_consistency - 自洽法、相關性

### 網格座標轉換
- convert_grid.py
  - equivalent_earth_model - 相當地球模型
  - equivalent_earth_model_by_elevations - 各仰角相當地球模型
  - polar_to_lonlat - 極座標轉經緯度座標
  - convert_grid_ppi - 
  - convert_grid_cs

### 濾波器
- filter.py
  - var_filter - 利用某參數上下限濾其他參數
  - ZD_filter - 透過Zdr濾雜波
  - KD_filter - 從Phidp輸出Kdp

### 衰減修正（適用C或X波段）
- attenuation_correction.py
  - attenuation_correction - 自行輸入衰減修正參數
  - attenuation_correction_C - C波段適用
    - 參考Bringi et al. 1990 (B90)
  - attenuation_correction_X - X波段適用
    - 參考FURUNO WR2100手冊

## FURUNO WR2100所附執行檔

放在`exe`目錄下，僅可供**Windows系統**執行：

- RainPlay - 雷達參數使用者介面
  - RainPlay.exe - 執行檔
  - RainPlay.ini - 配置文件
  - *.bmp - 地圖文件

- SCN2CfRadial_Converter - NetCDF轉檔程式
  - SCN2CfRadial_Converter.exe - 執行檔
  - config.txt - 配置文件
    - 包含輸入／輸出路徑、其他欲輸出資訊
  - quant_map_const.txt - 常數配置文件
    - 包含偏移量、增益、無效值

- SCN2HDF5_Converter - H5轉檔程式
  - SCN2HDF5_Converter.exe - 執行檔
  - config.txt - 配置文件
    - 包含輸入／輸出路徑、其他欲輸出資訊
  - quant_map_const.txt - 常數配置文件
    - 包含偏移量、增益、無效值
