# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 16:12:02 2021

@author: 陨星落云
"""

import numpy as np

def NDVI(Nir,R):
    '''
    #归一化植被指数
    Parameters
    ----------
    Nir : array
        近红外波段.
    R : array
        红波段.

    Returns
    -------
    array
        NDVI指数.

    '''
    return (Nir-R)/(Nir+R)

def RVI(Nir,R):
    '''
    #比值植被指数
    Parameters
    ----------
    Nir : array
        近红外波段.
    R : array
        红波段.

    Returns
    -------
    array
        RVI指数.

    '''
    return Nir/R

def NDWI(G,Nir):
    '''
    #归一化水体指数
    Parameters
    ----------
    G : array
        绿波段.
    Nir : array
        近红外波段.
        
    Returns
    -------
    array
        NDWI指数.

    '''
    return (G-Nir)/(G+Nir)

def SAVI(Nir,R,L=0.5):
    '''
    #土壤调节植被指数

    Parameters
    ----------
    Nir : array
        近红外波段.
    R : array
        红波段.
    L : float, optional
        #L是随着植被密度变化的参数，取值范围从0-1;
        当植被覆盖度很高时为0，很低时为1。The default is 0.5.

    Returns
    -------
    array
        SAVI指数.

    '''
    return (Nir-R)*(1+L)/(Nir+R+L)

