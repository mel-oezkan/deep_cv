import torch
from torch import nn


class SpatialGather_Module(nn.Module):
    def __init__(self, cls_num=0):
        super(SpatialGather_Module, self).__init__()
        self.cls_num = cls_num

    def forward(self, feats, probs):
        batch_size, c, h, w = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
        probs = probs.view(batch_size, c, -1)
        feats = feats.view(batch_size, feats.size(1), -1)
        feats = feats.permute(0, 2, 1) # batch x hw x c 
        probs = F.softmax(probs, dim=2)# batch x k x hw
        return torch.matmul(probs, feats).permute(0, 2, 1).unsqueeze(3).contiguous() # batch x k x c x 1
