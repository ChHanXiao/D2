'''
Date: 2021-10-19 21:16:26
Author: ChHanXiao
Github: https://github.com/ChHanXiao
LastEditors: ChHanXiao
LastEditTime: 2021-10-19 21:24:25
FilePath: /D2/data/transforms/__init__.py
'''
from detectron2.data.transforms import *
from .transform import *

__all__ = [k for k in globals().keys() if not k.startswith("_")]