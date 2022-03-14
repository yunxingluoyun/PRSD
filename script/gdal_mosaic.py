# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 10:25:35 2021

@author: xiaohuihui
"""
import os
from osgeo import gdal
# import subprocess
# 生成影像列表
# command = "dir /b *.tif > imglist.txt"
# subprocess.run(command)
os.system("dir /b *.tif > imglist.txt")

# 读取影像列表
f = open(r'imglist.txt','r')
L = f.readlines()
f.close()

# 影像镶嵌
files_to_mosaic = [ i.rstrip().split(':')[0] for i in L] # However many you want.
g = gdal.Warp("output.tif", files_to_mosaic, format="GTiff",
            options=["COMPRESS=LZW", "TILED=YES"]) # if you want
g = None # Close file and flush to disk