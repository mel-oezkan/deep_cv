from torch import nn

def m(in_channels, out_channels, k, d=1):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, k, padding=d if d>1 else k//2, dilation=d, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())
