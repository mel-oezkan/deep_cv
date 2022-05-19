import torch
from torch import nn

from models.ObjectAttentionBlock import ObjectAttentionBlock2D


class SpatialOCR_Module(nn.Module):
    def __init__(self, in_channels, key_channels, out_channels, dropout=0.1):
        super(SpatialOCR_Module, self).__init__()
        self.object_context_block = ObjectAttentionBlock2D(
            in_channels, key_channels)
        self.conv_bn_dropout = nn.Sequential(nn.Conv2d(
            2 * in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(), nn.Dropout2d(dropout))

    def forward(self, feats, proxy_feats):
        context = self.object_context_block(feats, proxy_feats)
        return self.conv_bn_dropout(torch.cat([context, feats], 1))
