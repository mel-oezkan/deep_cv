import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import Dataset, DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F

from models.ASPP import ASPP

class Unet(nn.Module):
    def __init__(self, extra_num = 1, dec_ch = [32, 64, 128, 256, 1024], stride = 32, net='b5', bot1x1=False, glob=False, bn = False, aspp=False, ocr=False, aux = False):
        super().__init__()

        self.extra_num = extra_num
        self.stride = stride
        self.bot1x1 = bot1x1
        self.glob = glob
        self.bn = bn
        self.aspp = aspp
        self.ocr = ocr
        self.aux = aux

        #! Select which weigths to fetch based on the selected EffNet
        if net == 'b4':
            channel_multiplier=1.4
            depth_multiplier=1.8
            url = 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b4_ns-d6313a46.pth'
            enc_ch = [24, 32, 56, 160, 1792]
        
        if net == 'b5':
            channel_multiplier=1.6
            depth_multiplier=2.2
            url = 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b5_ns-6f26d0cf.pth'
            enc_ch = [24, 40, 64, 176, 2048]
        
        if net == 'b6':
            channel_multiplier=1.8
            depth_multiplier=2.6
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b6_ns-51548356.pth'
            enc_ch = [32, 40, 72, 200, 2304]
        
        if net == 'b7':
            channel_multiplier=2.0
            depth_multiplier=3.1
            url = 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b7_ns-1dbc32de.pth'
            enc_ch = [32, 48, 80, 224, 2560]
        
        if net == 'l2':
            channel_multiplier=4.3
            depth_multiplier=5.3
            url = 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_l2_ns-df73bb44.pth'
            enc_ch = [72, 104, 176, 480, 5504]

        dilations = [False, False, False, False]
        if stride == 16:
            dec_ch[4] = enc_ch[4]
            dilations = [False, False, False, True]
        elif stride == 8:
            dec_ch[3] = enc_ch[4]
            dilations = [False, False, True, True]

        def mod(cin, cout, k=3):
            if bn:
                return nn.Sequential(nn.Conv2d(cin, cout, k, padding=k//2, bias=False), torch.nn.BatchNorm2d(cout), nn.ReLU(inplace=True))
            else:
                return nn.Sequential(nn.Conv2d(cin, cout, k, padding=k//2), nn.ReLU(inplace=True))

        if self.aspp:
            self.asppc = ASPP(enc_ch[4], 256)
            enc_ch[4] = 256
        if self.ocr:
            midc = 512
            keyc = 256
            numcl = 4 * 4 * 3
            enc_ch[4] = 512
            dec_ch[2] = midc
            inpc = sum(enc_ch[1:])
            self.aux_head = nn.Sequential(nn.Conv2d(inpc, inpc, 3, padding=1, bias=False), nn.BatchNorm2d(inpc), nn.ReLU(inplace=True), nn.Conv2d(inpc, numcl, 1))
            self.conv3x3_ocr = nn.Sequential(nn.Conv2d(inpc, midc, 3, padding=1, bias=False), nn.BatchNorm2d(midc), nn.ReLU(inplace=True))
            self.ocr_gather_head = SpatialGather_Module(numcl)
            self.ocr_distri_head = SpatialOCR_Module(in_channels=midc, key_channels=keyc, out_channels=midc, dropout=0.05)
        if self.glob:
            self.global_f = nn.Sequential(nn.AdaptiveAvgPool2d(1), mod(enc_ch[4], dec_ch[4], 1))

        self.bot0extra = mod(206, enc_ch[4])
        self.bot1extra = mod(206, dec_ch[4])
        self.bot2extra = mod(206, dec_ch[3])
        self.bot3extra = mod(206, dec_ch[2])
        self.bot4extra = mod(206, dec_ch[1])
        self.bot5extra = mod(206, 6)

        self.dec0 = mod(enc_ch[4], dec_ch[4])
        self.dec1 = mod(dec_ch[4], dec_ch[3])
        self.dec2 = mod(dec_ch[3], dec_ch[2])
        self.dec3 = mod(dec_ch[2], dec_ch[1])
        self.dec4 = mod(dec_ch[1], dec_ch[0])

        if self.bot1x1:
            self.bot1x10 = mod(enc_ch[3], enc_ch[3], 1)
            self.bot1x11 = mod(enc_ch[2], enc_ch[2], 1)
            self.bot1x12 = mod(enc_ch[1], enc_ch[1], 1)
            self.bot1x13 = mod(enc_ch[0], enc_ch[0], 1)

        self.bot0 = mod(enc_ch[3] + dec_ch[4], dec_ch[4])
        self.bot1 = mod(enc_ch[2] + dec_ch[3], dec_ch[3])
        self.bot2 = mod(enc_ch[1] + dec_ch[2], dec_ch[2])
        self.bot3 = mod(enc_ch[0] + dec_ch[1], dec_ch[1])

        self.up = nn.Upsample(scale_factor=2)
        self.upps = nn.PixelShuffle(upscale_factor=2)
        self.final = nn.Conv2d(dec_ch[0], 6, 1)
        if self.aux:
            aux_c = max(enc_ch[3], 16 * 16 * 3)
            self.aux_final = nn.Sequential(nn.Conv2d(enc_ch[3], aux_c, 3, padding=1, bias=False), nn.BatchNorm2d(aux_c), nn.ReLU(inplace=True), nn.Conv2d(aux_c, 16 * 16 * 3, 1))

        self._initialize_weights()

        arch_def = [
            ['ds_r1_k3_s1_e1_c16_se0.25'],
            ['ir_r2_k3_s2_e6_c24_se0.25'],
            ['ir_r2_k5_s2_e6_c40_se0.25'],
            ['ir_r3_k3_s2_e6_c80_se0.25'],
            ['ir_r3_k5_s1_e6_c112_se0.25'],
            ['ir_r4_k5_s2_e6_c192_se0.25'],
            ['ir_r1_k3_s1_e6_c320_se0.25']]
        enc = GenEfficientNet(in_chans=3, block_args=decode_arch_def(arch_def, depth_multiplier), num_features=round_channels(1280, channel_multiplier, 8, None), stem_size=32,
            channel_multiplier=channel_multiplier, act_layer=resolve_act_layer({}, 'swish'), norm_kwargs=resolve_bn_args({'bn_eps': BN_EPS_TF_DEFAULT}), pad_type='same', dilations=dilations)
        state_dict = load_state_dict_from_url(url)
        enc.load_state_dict(state_dict, strict=True)

        stem_size = round_channels(32, channel_multiplier, 8, None)
        conv_stem = select_conv2d(4, stem_size, 3, stride=2, padding='same')
        _w = enc.conv_stem.state_dict()
        _w['weight'] = torch.cat([_w['weight'], _w['weight'][:,1:2] ], 1)
        conv_stem.load_state_dict(_w)

        self.enc0 = nn.Sequential(conv_stem, enc.bn1, enc.act1, enc.blocks[0])
        self.enc1 = nn.Sequential(enc.blocks[1])
        self.enc2 = nn.Sequential(enc.blocks[2])
        self.enc3 = nn.Sequential(enc.blocks[3], enc.blocks[4])
        self.enc4 = nn.Sequential(enc.blocks[5], enc.blocks[6], enc.conv_head, enc.bn2, enc.act2)
        if self.ocr:
            self.enc4 = nn.Sequential(enc.blocks[5], enc.blocks[6])


    def forward(self, x, strip, direction, coord):
        enc0 = self.enc0(x)
        enc1 = self.enc1(enc0)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        if self.bot1x1:
            enc3 = self.bot1x10(enc3)
            enc2 = self.bot1x11(enc2)
            enc1 = self.bot1x12(enc1)
            enc0 = self.bot1x13(enc0)

        ex = torch.cat([strip, direction, coord], 1)
        x = enc4
        if self.aspp:
            x = self.asppc(x)
        elif self.ocr:
            enc1 = enc1
            enc2 = self.up(enc2)
            enc3 = self.up(self.up(enc3))
            enc4 = self.up(self.up(self.up(enc4)))
            feats = torch.cat([enc4, enc3, enc2, enc1], 1)

            out_aux = self.aux_head(feats)
            feats = self.conv3x3_ocr(feats)
            cont = self.ocr_gather_head(feats, out_aux)
            feats = self.ocr_distri_head(feats, cont)
  
            x = self.dec3(feats)
            x = torch.cat([x, enc0], dim=1)
            x = self.bot3(x)
            x = self.dec4(x)
            return self.final(x), self.upps(self.upps(out_aux))

        if self.stride == 32:
            x = self.dec0(self.up(x + (0 if self.extra_num <= 0 else self.bot0extra(ex)))) + (self.global_f(x) if self.glob else 0)
            x = torch.cat([x, enc3], dim=1)
            x = self.bot0(x) 
        if self.stride == 32 or self.stride == 16:
            x = self.dec1(self.up(x + (0 if self.extra_num <= 1 else self.bot1extra(ex))))
            x = torch.cat([x, enc2], dim=1)
            x = self.bot1(x)
        x = self.dec2(self.up(x + (0 if self.extra_num <= 2 else self.bot2extra(ex))))
        x = torch.cat([x, enc1], dim=1)
        x = self.bot2(x)
        x = self.dec3(self.up(x + (0 if self.extra_num <= 3 else self.bot3extra(ex))))
        x = torch.cat([x, enc0], dim=1)
        x = self.bot3(x) 
        x = self.dec4(self.up(x + (0 if self.extra_num <= 4 else self.bot4extra(ex))))
        x = self.final(x) + (0 if self.extra_num <= 5 else self.bot5extra(ex))
        if self.aux:
            return x, self.upps(self.upps(self.upps(self.upps(self.aux_final(enc3)))))
        else:
            return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
