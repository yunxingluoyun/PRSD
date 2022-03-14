

一、遥感数据处理（prsd）

可以使用下面命令打包一个源代码的包:

```
python setup.py sdist build
```

这样在当前目录的dist文件夹下，就会多出一个以tar.gz结尾的包了：

也可以打包一个wheels格式的包，使用下面的命令搞定：

```
python setup.py bdist_wheel --universal
```

这样会在dist文件夹下生成一个whl文件

#### 二、安装

环境配置

```
conda create -n RSDataProcessing python=3.8
conda activate RSDataProcessing
```

安装gdal

```
conda install gdal==3.0.2
```

安装prsd

```
pip install prsd-0.12.24-py2.py3-none-any.whl
```

目前prsd子模块

```
>>> from prsd import io,index,utils
>>> img,geotrans,proj = io.imread(r"E:\landsat2017-20210401T013451Z-001\landsat2017\LC08_123032_20170115.tif")
>>> io.imsave("test.tif",img,geotrans,proj)
```

