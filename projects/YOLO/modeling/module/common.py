# YOLOv3 common modules
""" Directly Imported from yolov3 - needed for the yolo.py script"""
import math
import warnings
import torch
import torch.nn as nn

from detectron2.layers import CNNBlockBase, Conv2d, get_norm


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(CNNBlockBase):
    # Standard convolution
    # ch_in, ch_out, kernel, stride, padding, groups
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, norm="BN", act=nn.LeakyReLU):
        super(Conv, self).__init__(in_channels=c1, out_channels=c2, stride=s)
        if isinstance(act, nn.LeakyReLU):
            act = nn.LeakyReLU(0.1)
        else:
            act = act()
        norm = get_norm(norm, c2)
        bias = norm is None
        self.conv = Conv2d(c1, c2, k, s, autopad(k, p), groups=g,
                           bias=bias, norm=norm, activation=act)

    def forward(self, x):
        return self.conv(x)

class DWConv(Conv):
    # Depth-wise convolution class
    def __init__(self, c1, c2, k=1, s=1, **kwargs):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), **kwargs)

class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*[TransformerLayer(c2, num_heads) for _ in range(num_layers)])
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).unsqueeze(0).transpose(0, 3).squeeze(3)
        return self.tr(p + self.linear(p)).unsqueeze(3).transpose(0, 3).reshape(b, self.c2, w, h)




class Bottleneck(CNNBlockBase):
    # Standard bottleneck
    # ch_in, ch_out, shortcut, groups, expansion
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5, **kwargs):
        super(Bottleneck, self).__init__(in_channels=c1, out_channels=c2, stride=1)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, **kwargs)
        self.cv2 = Conv(c_, c2, 3, 1, g=g, **kwargs)
        self.add = False
        if shortcut and c1 == c2:
            self.add = True

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(CNNBlockBase):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, **kwargs):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__(in_channels=c1, out_channels=c2, stride=1)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, **kwargs)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1, **kwargs)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0, **kwargs) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))

class C3(CNNBlockBase):
    # CSP Bottleneck with 3 convolutions
    # ch_in, ch_out, number, shortcut, groups, expansion
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, **kwargs):
        super(C3, self).__init__(in_channels=c1, out_channels=c2, stride=1)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, **kwargs)
        self.cv2 = Conv(c1, c_, 1, 1, **kwargs)
        self.cv3 = Conv(2 * c_, c2, 1, **kwargs)  # act=FReLU(c2, **kwargs)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0, **kwargs)
                               for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)], **kwargs)

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class CSPOSA(nn.Module):
    # CSP OSA Net with PCB as proposed for tiny-Yolov4
    def __init__(self, c1, c2, **kwargs):
        super(CSPOSA, self).__init__()
        assert c1 % 2 == 0
        assert c2 == c1 * 2
        g = c1 // 2
        self.g = g
        self.cv1 = Conv(c1, 2 * g, 3, **kwargs)
        self.cv2 = Conv(g, g, 3, **kwargs)
        self.cv3 = Conv(g, g, 3, **kwargs)
        self.cv4 = Conv(2 * g, 2 * g, 1, **kwargs)

    def forward(self, x):
        x = self.cv1(x)
        x1, x2 = torch.split(x, self.g, dim=1)
        x3 = self.cv2(x2)
        x4 = self.cv3(x3)
        x3 = torch.cat([x3, x4], dim=1)
        x3 = self.cv4(x3)
        return torch.cat([x1, x2, x3], dim=1)


class SPP(CNNBlockBase):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13), **kwargs):
        super(SPP, self).__init__(in_channels=c1, out_channels=c2, stride=1)
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, **kwargs)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1, **kwargs)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(CNNBlockBase):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5, **kwargs):  # equivalent to SPP(k=(5, 9, 13))
        super(SPPF, self).__init__(in_channels=c1, out_channels=c2, stride=1)
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, **kwargs)
        self.cv2 = Conv(c_ * 4, c2, 1, 1, **kwargs)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class Focus(CNNBlockBase):
    # Focus wh information into c-space
    # ch_in, ch_out, kernel, stride, padding, groups
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, **kwargs):
        super(Focus, self).__init__(in_channels=c1 * 4, out_channels=c2, stride=s)
        self.conv = Conv(c1 * 4, c2, k, s, p, g, **kwargs)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(
            torch.cat(
                [x[..., :: 2, :: 2],
                 x[..., 1:: 2, :: 2],
                 x[..., :: 2, 1:: 2],
                 x[..., 1:: 2, 1:: 2]],
                1))
        # return self.conv(self.contract(x))


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert (h / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(b, c, h // s, s, w // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(b, c * s * s, h // s, w // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(b, s, s, c // s ** 2, h, w)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(b, c // s ** 2, h * s, w * s)  # x(1,16,160,160)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)
