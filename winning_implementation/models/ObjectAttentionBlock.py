import torch
from torch import nn

class ObjectAttentionBlock2D(nn.Module):
    def __init__(self, inc, keyc, bn_type=None):
        super(ObjectAttentionBlock2D, self).__init__()
        self.keyc = keyc
        self.f_pixel = nn.Sequential(nn.Conv2d(inc, keyc, 1, bias=False), nn.BatchNorm2d(keyc), nn.ReLU(), nn.Conv2d(keyc, keyc, 1, bias=False), nn.BatchNorm2d(keyc), nn.ReLU())
        self.f_object = nn.Sequential(nn.Conv2d(inc, keyc, 1, bias=False), nn.BatchNorm2d(keyc), nn.ReLU(), nn.Conv2d(keyc, keyc, 1, bias=False), nn.BatchNorm2d(keyc), nn.ReLU())
        self.f_down = nn.Sequential(nn.Conv2d(inc, keyc, 1, bias=False), nn.BatchNorm2d(keyc), nn.ReLU())
        self.f_up = nn.Sequential(nn.Conv2d(keyc, inc, 1, bias=False), nn.BatchNorm2d(inc), nn.ReLU())

    def forward(self, x, proxy):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        query = self.f_pixel(x).view(batch_size, self.keyc, -1)
        query = query.permute(0, 2, 1)
        key = self.f_object(proxy).view(batch_size, self.keyc, -1)
        value = self.f_down(proxy).view(batch_size, self.keyc, -1)
        value = value.permute(0, 2, 1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.keyc**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)   

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.keyc, *x.size()[2:])
        context = self.f_up(context)
        return context

