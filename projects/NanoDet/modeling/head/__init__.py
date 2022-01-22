'''
Date: 2021-10-19 21:15:08
Author: ChHanXiao
Github: https://github.com/ChHanXiao
LastEditors: ChHanXiao
LastEditTime: 2021-12-29 22:20:51
FilePath: /D2/projects/NanoDet/modeling/head/__init__.py
'''
import copy

from .gfl_head import GFLHead
from .nanodet_head import NanoDetHead
from .nanodet_plus_head import NanoDetPlusHead
from .simple_conv_head import SimpleConvHead

def build_head(cfg):
    head_cfg = copy.deepcopy(cfg)
    name = head_cfg.pop("name")
    if name == "GFLHead":
        return GFLHead(**head_cfg)
    elif name == "NanoDetHead":
        return NanoDetHead(**head_cfg)
    elif name == "NanoDetPlusHead":
        return NanoDetPlusHead(**head_cfg)
    elif name == "SimpleConvHead":
        return SimpleConvHead(**head_cfg)
    else:
        raise NotImplementedError
