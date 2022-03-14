# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 15:51:09 2021

@author: 陨星落云

"""
from osgeo import gdal
from osgeo import gdalconst


def imageClip(indata_path,inshp_path,outdata_path):
    '''
    影像裁剪
    Parameters
    ----------
    indata_path : str
        待裁剪的影像路径.
    inshp_path : str
        裁剪的矢量shp路径.
    outdata_path : str
        裁剪后的影像路径.

    Returns
    -------
    None.

    '''
    gdal.Warp(outdata_path,indata_path,cutlineDSName=inshp_path,cropToCutline=True,dstNodata=-9999)
    
def imageResample(indata_path,outdata_path,w,h,Alg='near'):
    '''
    影像重采样

    Parameters
    ----------
    indata_path : str
        输入路径.
    outdata_path : str
        输出路径.
    w : int
        宽.
    h : int
        高.
    Alg : str
        near ：近邻重采样（默认、最快算法、最差插值质量）。
        bilinear ：双线性重采样。
        cubic ：立方重采样。
        cubicspline ：三次样条线重采样。
        lanczos ：Lanczos窗口sinc重新采样。
        average ：平均重采样，计算所有非节点数据贡献像素的加权平均值。

    Returns
    -------
    None.

    '''
    resampleAlgs ={'near':gdalconst.GRA_NearestNeighbour,
                  'bilinear':gdalconst.GRIORA_Bilinear,
                  'cubic':gdalconst.GRA_Cubic,
                  'cubicspline':gdalconst.GRA_CubicSpline,
                  'lanczos':gdalconst.GRA_Lanczos,
                  'average':gdalconst.GRA_Average}
    if Alg in resampleAlgs.keys():
        resampleAlg =resampleAlgs[Alg]
    else:
        raise ValueError("不存在该采样算法")
    gdal.Warp(outdata_path,indata_path,width=w, height=h, resampleAlg=resampleAlg)
