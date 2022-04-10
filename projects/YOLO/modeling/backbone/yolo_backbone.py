# Copyright (c) Facebook, Inc. and its affiliates.
import torch
import torch.nn as nn
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init
from copy import deepcopy
import yaml
from pathlib import Path
from detectron2.config import configurable
from detectron2.modeling import BACKBONE_REGISTRY, Backbone
from detectron2.layers import ShapeSpec, BatchNorm2d, Conv2d
from ..module.common import *
from ..head import Detect, DetectX, DetectYoloX
from ..util.general import make_divisible

__all__ = ["YOLO_BACKBONE"]

class YOLO_BACKBONE(nn.Module):

    @configurable
    def __init__(self, cfg, ch=3):
        super().__init__()
        self.yaml = cfg
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        self.kp = 0
        self.model, self.save = self.parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', False)
       # Build strides, anchors
        m = self.model[-1]  # Detect()
        self.model_type = 'yolov5'
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self.initialize_biases()  # only run once
        elif isinstance(m, (DetectX, DetectYoloX)):
            m.inplace = self.inplace
            self.stride = torch.tensor(m.stride)
            m.initialize_biases()  # only run once
            self.model_type = 'yolox'
        
        # Init weights, biases
        self.initialize_weights()

    @classmethod
    def from_config(cls, cfg):
        model_yaml_file = cfg.MODEL.YAML
        with open(model_yaml_file) as f:
            model_yaml = yaml.safe_load(f)  # model dict
        in_channels = 3
        return {
            "cfg": model_yaml, 
            "ch": in_channels}

    def forward(self, x):
        return self.forward_once(x)  # augmented inference, None

    def forward_once(self, x):
        y = []
        for m in self.model:
            if m.f != -1:   # Not the previous layer
                if isinstance(m.f, int):
                    x = y[m.f]
                else:
                    x = [x if j == -1 else y[j] for j in m.f]

            x = m(x)  # run
            y.append(x if m.i in self.save else None)

        return x

    def initialize_weights(self):
        for m in self.modules():
            t = type(m)
            if t is nn.Conv2d:
                pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif t is nn.BatchNorm2d:
                m.eps = 1e-3
                m.momentum = 0.03
            elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
                m.inplace = True

    def initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


    def parse_model(self, d, ch):  # model_dict, input_channels(3)
        anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
        keypoints = d.get('keypoints', 0)
        self.kp = keypoints
        na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
        no = na * (nc + 5 + 2 * keypoints*2)  # number of outputs = anchors * (classes + 5 + keypoints*2)

        layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
        for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
            m = eval(m) if isinstance(m, str) else m  # eval strings
            for j, a in enumerate(args):
                try:
                    args[j] = eval(a) if isinstance(a, str) else a  # eval strings
                except NameError:
                    pass
            if m == Conv:
                n_ = n
            else:
                n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
            if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, Focus,
                    BottleneckCSP, C3, C3TR, C3SPP, C3Ghost]:
                c1, c2 = ch[f], args[0]
                if c2 != no:  # if not output
                    c2 = make_divisible(c2 * gw, 8)

                args = [c1, c2, *args[1:]]
                if m in [BottleneckCSP, C3, C3TR, C3Ghost]:
                    args.insert(2, n)  # number of repeats
                    n = 1
            elif m is nn.BatchNorm2d:
                args = [ch[f]]
            elif m is Concat:
                c2 = sum([ch[x] for x in f])
            elif m is Detect:
                args.append([ch[x] for x in f])
                if isinstance(args[1], int):  # number of anchors
                    args[1] = [list(range(args[1] * 2))] * len(f)
            elif m in {DetectX, DetectYoloX}:
                args.append([ch[x] for x in f])
            elif m is Contract:
                c2 = ch[f] * args[0] ** 2
            elif m is Expand:
                c2 = ch[f] // args[0] ** 2
            else:
                c2 = ch[f]

            m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
            t = str(m)[8:-2].replace('__main__.', '')  # module type
            np = sum([x.numel() for x in m_.parameters()])  # number params
            m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
            save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
            layers.append(m_)
            if i == 0:
                ch = []
            ch.append(c2)
        return nn.Sequential(*layers), sorted(save)


