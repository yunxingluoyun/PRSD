# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 11:54:56 2021

@author: 陨星落云
"""
import os
from osgeo import gdal
from osgeo.gdalconst import GA_ReadOnly,GA_Update
import numpy as np

    
def imread(file_path):
    '''
    读取数据
    Parameters
    ----------
    file_path : str
        DESCRIPTION.

    Raises
    ------
    ValueError
        无法打开该文件.

    Returns
    -------
    img_arr : array
        DESCRIPTION.
    geotrans : TYPE
        DESCRIPTION.
    proj : TYPE
        DESCRIPTION.

    '''
    gdal.AllRegister()
    inDataset = gdal.Open(file_path,GA_ReadOnly)
    if inDataset==None:
        raise ValueError("文件无法打开")
    #width = inDataset.RasterXSize  # 宽
    #height = inDataset.RasterYSize  # 高
    #bands = inDataset.RasterCount  # 获取波段数
    geotrans = inDataset.GetGeoTransform() 
    proj = inDataset.GetProjectionRef()# 获取投影信息
    img_arr = inDataset.ReadAsArray() # (c,h,w)
    # print(dir(inDataset))
    # print('--'*16)
    # print("width:",width)
    # print("height:",height)
    # print("bands:",bands)
    # print("geotransform:",geotrans)
    # print("projection:",proj)
    # print("img(array):\n",inDataset.ReadAsArray().shape) 
    
    return img_arr,geotrans,proj
 

def imsave(path,img_arr,geotrans=None,proj=None):
    '''
    保存数据，只支持tif格式
    Parameters
    ----------
    img_arr : array
        DESCRIPTION.
    path : str
        保存路径.
    geotrans : TYPE, optional
        仿射变换. The default is None.
    proj : TYPE, optional
        投影. The default is None.

    Returns
    -------
    None.

    '''
    
    if 'int8' in img_arr.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in img_arr.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(img_arr.shape) == 3:
        bands, height, width = img_arr.shape
    elif len(img_arr.shape) == 2:
        img_arr = np.array([img_arr])
        bands, height, width = img_arr.shape
    if os.path.exists(path):
        #print("该影像已存在,正在修改该影像......")
        dataset = gdal.Open(path,GA_Update)      
    else:
        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(path, int(width), int(height), int(bands), datatype)
    
    if(dataset!= None):
        if geotrans!=None:
            dataset.SetGeoTransform(geotrans) #写入仿射变换参数
        if proj!=None:
            dataset.SetProjection(proj) #写入投影
    for i in range(bands):
        dataset.GetRasterBand(i + 1).WriteArray(img_arr[i])
    del dataset  


if __name__ == '__main__':
    
    file_path = 'E:\landsat2017-20210401T013451Z-001\landsat2017\LC08_123032_20171014.tif'
    img,geotrans,proj = imread(file_path)
    print(type(img.dtype))
    imsave(img,r"D:\PRSD\test_data\test1.tif")



