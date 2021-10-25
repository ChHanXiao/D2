# Copyright (c) Facebook, Inc. and its affiliates.
from pathlib import Path
import fvcore.nn.weight_init as weight_init
import torch.nn.functional as F
import torch.nn as nn
from detectron2.config import configurable
from detectron2.modeling import BACKBONE_REGISTRY, Backbone
from detectron2.layers import ShapeSpec, BatchNorm2d, Conv2d
from ..module.common import *
from ..util.general import make_divisible
from copy import deepcopy
import torch


class Detect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4] or self.onnx_dynamic:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)])
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (self.anchors[i].clone() * self.stride[i]) \
            .view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid



class DarkNet(Backbone):

    @configurable
    def __init__(self, cfg, ch=3, norm="BN", activation="nn.LeakyReLU"):
        super().__init__()
        self.yaml = cfg
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        head = [x for x in self.yaml['head'] if 'Detect' not in x]
        self.out_features = -1
        self.out_feature_channels = {}
        self.out_feature_strides = {}
        activation = eval(activation) if isinstance(activation, str) else activation
        post_conv = {'norm': norm,
                     'act': activation}
        if 'Detect' in self.yaml['head'][-1]:
            self.out_features = self.yaml['head'][-1][0]
            print("Detection Head found")


        self.model, self.save = self.parse_model(post_conv, deepcopy(self.yaml), ch=[ch])  # model, savelist

        print(self.model)
        print(self.save)

        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', False)

        # Init weights, biases
        self.initialize_weights()

    @classmethod
    def from_config(cls, cfg):
        model_yaml_file = cfg.MODEL.YAML
        import yaml  # for torch hub
        with open(model_yaml_file) as f:
            model_yaml = yaml.safe_load(f)  # model dict
        in_channels = 3
        norm = cfg.MODEL.YOLO.NORM
        activation = cfg.MODEL.YOLO.ACTIVATION
        return {
            "cfg": model_yaml, 
            "ch": in_channels, 
            "norm": norm, 
            "activation": activation}

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

        return {j: y[j] for j in self.out_features}

    def initialize_weights(self):
        for m in self.modules():
            t = type(m)
            if t is Conv2d:
                pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif t is BatchNorm2d:
                m.eps = 1e-3
                m.momentum = 0.03

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self.out_feature_channels[name],
                stride=self.out_feature_strides[name]
            )
            for name in self.out_features
        }
    def parse_model(self, post_conv, d, ch):  # model_dict, input_channels(3)
        anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
        na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
        no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)
        head = [x for x in d['head'] if 'Detect' not in x]
        if 'Detect' in d['head'][-1]:
            self.out_features = d['head'][-1][0]
            print("Detection Head found")
        layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
        for i, (f, n, m, args) in enumerate(d['backbone'] + head):  # from, number, module, args
            m = eval(m) if isinstance(m, str) else m  # eval strings
            for j, a in enumerate(args):
                try:
                    args[j] = eval(a) if isinstance(a, str) else a  # eval strings
                except NameError:
                    pass

            n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
            if m in [Conv, Bottleneck, SPP, SPPF, DWConv, Focus, BottleneckCSP, C3]:
                c1, c2 = ch[f], args[0]
                if c2 != no:  # if not output
                    c2 = make_divisible(c2 * gw, 8)
                kwargs = post_conv
                args = [c1, c2, *args[1:]]
                if m in [BottleneckCSP, C3]:
                    args.insert(2, n)  # number of repeats
                    n = 1
            elif m is nn.BatchNorm2d:
                args = [ch[f]]
                kwargs = {}
            elif m is Concat:
                c2 = sum([ch[x] for x in f])
                kwargs = {}
            elif m is Detect:
                args.append([ch[x] for x in f])
                if isinstance(args[1], int):  # number of anchors
                    args[1] = [list(range(args[1] * 2))] * len(f)
            elif m is Contract:
                c2 = ch[f] * args[0] ** 2
                kwargs = {}
            elif m is Expand:
                c2 = ch[f] // args[0] ** 2
                kwargs = {}
            else:
                c2 = ch[f]
                kwargs = {}

            m_ = nn.Sequential(*[m(*args, **kwargs) for _ in range(n)]) if n > 1 else m(*args, **kwargs)  # module
            t = str(m)[8:-2].replace('__main__.', '')  # module type
            np = sum([x.numel() for x in m_.parameters()])  # number params
            m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
            save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
            layers.append(m_)
            if i == 0:
                ch = []
            ch.append(c2)
            if i in self.out_features:
                self.out_feature_channels[i] = c2
                self.out_feature_strides[i] = 1
            save.extend(x for x in self.out_features if self.out_features != -1)

        return nn.Sequential(*layers), sorted(save)

