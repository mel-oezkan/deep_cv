import torch
from torch import nn

from models.Utils import m

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels = 256, rates = [12, 24, 36]):
        super(ASPP, self).__init__()

        self.c1 = m(in_channels, out_channels, 1)
        self.c2 = m(in_channels, out_channels, 3, rates[0])
        self.c3 = m(in_channels, out_channels, 3, rates[1])
        self.c4 = m(in_channels, out_channels, 3, rates[2])
        self.cg = nn.Sequential(nn.AdaptiveAvgPool2d(1), m(in_channels, out_channels, 1))
        self.project = m(4 * out_channels, out_channels, 1)
        self.projectg = m(out_channels, out_channels, 1)
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        c1 = self.c1(x)
        c2 = self.c2(x)
        c3 = self.c3(x)
        c4 = self.c4(x)
        cg = self.cg(x)
        c14 = self.project(torch.cat([c1, c2, c3, c4], dim=1))
        cg = self.projectg(cg)
        return self.drop(c14 + cg)