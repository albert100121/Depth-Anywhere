import sys
import numpy as np
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
import torchvision.models as models
import functools
import math
from .BaseModule import BaseModule
from .CETransform import CETransform


def weights_init(m):
    # Initialize filters with Gaussian random weights
    if isinstance(m, nn.Conv2d): 
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None: 
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None: 
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


class ResNet(nn.Module):
    def __init__(self, layers, in_channels=3, pretrained=True):
        assert layers in [18, 34, 50, 101, 152]
        super().__init__()
        pretrained_model = models.__dict__['resnet{}'.format(layers)](weights='ResNet%d_Weights.DEFAULT'%layers)
        if in_channels == 3:
            self.conv1 = pretrained_model._modules['conv1']
            self.bn1 = pretrained_model._modules['bn1']
        else:
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            weights_init(self.conv1)
            weights_init(self.bn1)
        
        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.layer1 = pretrained_model._modules['layer1']
        self.layer2 = pretrained_model._modules['layer2']
        self.layer3 = pretrained_model._modules['layer3']
        self.layer4 = pretrained_model._modules['layer4']

        # clear memory
        del pretrained_model

    def forward(self, x):
        # resnet
        x = self.conv1(x)
        x = self.bn1(x)
        x0 = self.relu(x)
        x = self.maxpool(x0)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        out = {
            'l0': x0,
            'l1': x1,
            'l2': x2,
            'l3': x3,
            'l4': x4
        }

        return out
    
    def preforward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x0 = self.relu(x)
        x = self.maxpool(x0)

        return x


class FusionModule(nn.Module):
    def __init__(self, num_channels, CE):
        super().__init__()
        self.CE = CE
        self.conv_equi_cat = nn.Sequential(
            nn.Conv2d(num_channels*2, num_channels, kernel_size=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
        )
        self.conv_cube_cat = nn.Sequential(
            nn.Conv2d(num_channels*2, num_channels, kernel_size=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
        )
        self.conv_cat = nn.Sequential(
            nn.Conv2d(num_channels*2, num_channels, kernel_size=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
        )
    
    def forward(self, equi, cube):
        f_equi = equi
        f_cube = self.CE.C2E(cube)
        f_cat = torch.cat([f_equi, f_cube], dim=1)
        f_equi = f_equi + self.conv_equi_cat(f_cat)
        f_cube = f_cube + self.conv_cube_cat(f_cat)
        f_fusion = self.conv_cat(f_cat)

        return f_equi, self.CE.E2C(f_cube), f_fusion

class ResUNet(nn.Module):
    def __init__(self, layers, CE_equi_h):
        super().__init__()
        num_channels = 512 if layers <= 34 else 2048
        self.resnet_equi = ResNet(layers, in_channels=3, pretrained=True)
        self.resnet_cube = ResNet(layers, in_channels=3, pretrained=True)
        self.ce = CETransform(CE_equi_h)

        self.f1 = FusionModule(num_channels//8, self.ce)
        self.f2 = FusionModule(num_channels//4, self.ce)
        self.f3 = FusionModule(num_channels//2, self.ce)
        self.f4 = FusionModule(num_channels//1, self.ce)
        
        planes = [num_channels, num_channels//2, num_channels//4, num_channels//8]
        def create_conv(in_ch, out_ch, kernel_size):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=(kernel_size-1)//2),
                nn.BatchNorm2d(out_ch),
                nn.ReLU()
            )
        self.deconv4 = nn.Sequential(
            create_conv(planes[0], planes[1]*4, kernel_size=3),
            nn.PixelShuffle(2)
        )
        self.deconv3 = nn.Sequential(
            create_conv(planes[0], planes[2]*4, kernel_size=3),
            nn.PixelShuffle(2)
        )
        self.deconv2 = nn.Sequential(
            create_conv(planes[1], planes[3]*4, kernel_size=3),
            nn.PixelShuffle(2)
        )
        self.deconv1 = nn.Sequential(
            create_conv(planes[2], planes[3]//2*4, kernel_size=3),
            nn.PixelShuffle(2),
            nn.Conv2d(planes[3]//2, 1, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

        
    
    def forward(self, x):        
        f_equi = self.resnet_equi.preforward(x)
        f_cube = self.resnet_cube.preforward(self.ce.E2C(x))
        
        f_equi_l1 = self.resnet_equi.layer1(f_equi)
        f_cube_l1 = self.resnet_cube.layer1(f_cube)
        f_equi_l1, f_cube_l1, fusion_1 = self.f1(f_equi_l1, f_cube_l1)

        f_equi_l2 = self.resnet_equi.layer2(f_equi_l1)
        f_cube_l2 = self.resnet_cube.layer2(f_cube_l1)
        f_equi_l2, f_cube_l2, fusion_2 = self.f2(f_equi_l2, f_cube_l2)

        f_equi_l3 = self.resnet_equi.layer3(f_equi_l2)
        f_cube_l3 = self.resnet_cube.layer3(f_cube_l2)
        f_equi_l3, f_cube_l3, fusion_3 = self.f3(f_equi_l3, f_cube_l3)

        f_equi_l4 = self.resnet_equi.layer4(f_equi_l3)
        f_cube_l4 = self.resnet_cube.layer4(f_cube_l3)
        f_equi_l4, f_cube_l4, fusion_4 = self.f4(f_equi_l4, f_cube_l4)

        
        feat = self.deconv4(fusion_4)
        
        feat = torch.cat([feat, fusion_3], dim=1)
        feat = self.deconv3(feat)

        feat = torch.cat([feat, fusion_2], dim=1)
        feat = self.deconv2(feat)

        feat = torch.cat([feat, fusion_1], dim=1)
        depth = self.deconv1(feat)

        return [depth]


class PoseNet(nn.Module):
    def __init__(self, layers, nb_tgts):
        super().__init__()
        self.nb_tgts = nb_tgts
        num_channels = 512 if layers <= 34 else 2048
        self.resnet = ResNet(layers, 3*(nb_tgts+1), pretrained=True)
        self.pose_pred = nn.Sequential(
            nn.Conv2d(num_channels, num_channels//2, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_channels//2),
            nn.Conv2d(num_channels//2, 6*nb_tgts, kernel_size=1)
        )
        self.exp_layers = nn.ModuleList([])
        for i in range(4):
            c = num_channels // (2**i)
            l = nn.Sequential(
                nn.Conv2d(c, nb_tgts, kernel_size=3, stride=1, padding=1,bias=False),
                nn.Sigmoid(),
                nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
            )
            self.exp_layers.append(l)
        self.exp_layers = self.exp_layers[::-1]

        self.pose_pred.apply(weights_init)
        self.exp_layers.apply(weights_init)
    
    def forward(self, ref, tgts: list):
        x = torch.cat([ref] + tgts, dim=1)
        res_out = self.resnet(x)
        pose = self.pose_pred(res_out['l4']).mean(-1).mean(-1)
        pose = 0.01 * pose.view(-1, self.nb_tgts, 6)

        exp_lst = []
        for i, (key, val) in enumerate(res_out.items()):
            if key == 'l0': continue
            exp_lst.append(self.exp_layers[i-1](val))
        
        return exp_lst, pose
    

class SupervisedCombinedModel(BaseModule):

    def __init__(self, save_path, dnet_args, pnet_args=None):
        super().__init__(save_path)
        self.dnet = ResUNet(**dnet_args)

    def forward(self, batch):
        depth = self.dnet(batch)

        return {'pred_depth': depth[0]}
