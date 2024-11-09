import torch
import torch.nn as nn
from utils.Projection import Equirec2Cube, Cube2Equirec


class CETransform(nn.Module):
    def __init__(self, equ_h_lst):
        super(CETransform, self).__init__()

        self.c2e = nn.ModuleDict()
        self.e2c = nn.ModuleDict()

        for h in equ_h_lst:
            a = Equirec2Cube(h//2, h, FoV=90)
            self.e2c['(%d,%d)'%(h, h*2)] = a

            b = Cube2Equirec(h//2, h)
            self.c2e['(%d)'%(h//2)] = b


    def E2C(self, x, mode='bilinear'):
        [bs, c, h, w] = x.shape
        assert w == h*2
        key = '(%d,%d)' % (h, w)
        assert key in self.e2c and mode in ['nearest', 'bilinear']
        return self.e2c[key](x, mode=mode)

    def C2E(self, x, mode='bilinear'):
        [bs, c, h, w] = x.shape
        assert h == w
        key = '(%d)' % (h)
        assert key in self.c2e and h == w and mode in ['nearest', 'bilinear']
        return self.c2e[key](x, mode=mode)

    def forward(self, equi, cube):
        return self.e2c(equi), self.c2e(cube)
