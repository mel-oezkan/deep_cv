import torch
from torch import nn


class GenEfficientNet(nn.Module):
    def __init__(self, block_args, num_classes=1000, in_chans=3, num_features=1280, stem_size=32, fix_stem=False, channel_multiplier=1.0, channel_divisor=8, channel_min=None,
                 pad_type='', act_layer=nn.ReLU, drop_connect_rate=0., se_kwargs=None, norm_layer=nn.BatchNorm2d, norm_kwargs=None, weight_init='goog', dilations=[False, False, False, False]):
        super(GenEfficientNet, self).__init__()

        stem_size = round_channels(
            stem_size, channel_multiplier, channel_divisor, channel_min)
        self.conv_stem = select_conv2d(
            in_chans, stem_size, 3, stride=2, padding=pad_type)
        self.bn1 = norm_layer(stem_size, **norm_kwargs)
        self.act1 = act_layer(inplace=True)

        builder = EfficientNetBuilder(channel_multiplier, channel_divisor, channel_min,
                                      pad_type, act_layer, se_kwargs, norm_layer, norm_kwargs, drop_connect_rate)
        self.blocks = nn.Sequential(*builder(stem_size, block_args, dilations))

        self.conv_head = select_conv2d(
            builder.in_chs, num_features, 1, padding=pad_type)
        self.bn2 = norm_layer(num_features, **norm_kwargs)
        self.act2 = act_layer(inplace=True)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(num_features, num_classes)

        for n, m in self.named_modules():
            if weight_init == 'goog':
                initialize_weight_goog(m, n)
            else:
                initialize_weight_default(m, n)
