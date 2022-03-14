#!/usr/bin/env python
#-*- coding:utf-8 -*-

#############################################
# File Name: setup.py
# Author: yunxingluoyun
# Mail: 672319707@qq.com
# Created Time:  2021-12-24 19:17:34
#############################################

from setuptools import setup, find_packages            #这个包没有的可以pip一下

setup(
    name = "prsd",      #这里是pip项目发布的名称
    version = "0.12.24",  #版本号，数值大的会优先被pip
    keywords = ("pip", "prsd","remote sensing"),
    description = "remote sensing image processing",
    long_description = "remote sensing image processing",
    license = "MIT Licence",

    url = "",     #项目相关文件地址，一般是github
    author = "yunxingluoyun",
    author_email = "672319707@qq.com",

    packages = find_packages(),
    include_package_data = True,
    platforms = "any",
    install_requires = ["numpy","gdal","pytorch"]          #这个项目需要的第三方库
)