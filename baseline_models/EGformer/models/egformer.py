# ------------------------------------------
# Equirectangular geometry-biased Transformer
# ------------------------------------------

# These codes are based on the following codes :
# CSwin-Transformer (https://github.com/microsoft/CSWin-Transformer), 
# Panoformer (https://github.com/zhijieshen-bjtu/PanoFormer),
# and others.

# We thank the authors providing the codes availble

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

import timm
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from einops.layers.torch import Rearrange
import torch.utils.checkpoint as checkpoint
import numpy as np
import time
from models.Panoformer.model import *



class Interpolate(nn.Module):
    """Interpolation module."""

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.

        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )

        return x



VRPE_LUT=[]
HRPE_LUT=[]


def VRPE(num_heads, height,width,split_size): ## Used for vertical attention  / (theta,phi) -> (theta,phi') / This part is coded assuming that the split_size (stripe_with) is 1 to reduce the computational cost. To use larger split_size, ERPE must be calculated accoringly.

    H = height // split_size # Base height
    pi = torch.acos(torch.zeros(1)).item() * 2
    
    base_x = torch.linspace(0,H,H) * pi / H
    base_x = base_x.unsqueeze(0).repeat(H,1)

    base_y = torch.linspace(0,H,H) * pi / H
    base_y = base_y.unsqueeze(1).repeat(1,H)

    base = base_x - base_y
    pn = torch.where(base>0,1,-1)
    
    base =  torch.sqrt(2 * (1 - torch.cos(base))) # H x H 
    base = pn * base
    return (base.unsqueeze(0).unsqueeze(0)).repeat(width * split_size,num_heads,1,1) 

def HRPE(num_heads, height, width, split_size): ## Used for Horizontal attention  / (theta,phi) -> (theta',phi) / This part is coded assuming that the split_size (stripe_with) is 1 to reduce the computational cost. To use larger split_size, ERPE must be calculated accoringly.


    W = width // split_size # Base width
    pi = torch.acos(torch.zeros(1)).item() * 2

    base_x = torch.linspace(0,W,W) *2*pi / W
    base_x = base_x.unsqueeze(0).repeat(W,1)

    base_y = torch.linspace(0,W,W)*2*pi / W
    base_y = base_y.unsqueeze(1).repeat(1,W)
    base = base_x - base_y
    pn = torch.where(base>0,1,-1)
    base = base.unsqueeze(0).repeat(height,1,1)

    for k in range(0,height):
        base[k,:,:] = torch.sin(torch.as_tensor(k*pi/height)) * torch.sqrt(2 * (1 - torch.cos(base[k,:,:]))) # height x W x W  
    
    if True: # Unlike the vertical direction, EIs are cyclic along the horizontal direction. Set to 'False' to reflect this cyclic characteristic / Refer to discussions in repo for more details. 
        base = pn * base
    return base.unsqueeze(1).repeat(split_size,num_heads,1,1) 


# LUT should be updated if input resolution is not 512x1024. 

VRPE_LUT.append(VRPE(1,256,512,1))
HRPE_LUT.append(HRPE(1,256,512,1))

VRPE_LUT.append(VRPE(2,128,256,1))
HRPE_LUT.append(HRPE(2,128,256,1))

VRPE_LUT.append(VRPE(4,64,128,1))
HRPE_LUT.append(HRPE(4,64,128,1))

VRPE_LUT.append(VRPE(8,32,64,1))
HRPE_LUT.append(HRPE(8,32,64,1))

VRPE_LUT.append(VRPE(16,16,32,1))
HRPE_LUT.append(HRPE(16,16,32,1))


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class EGAttention(nn.Module):
    def __init__(self, dim, resolution, idx, split_size=7, dim_out=None, num_heads=8, attn_drop=0., proj_drop=0., qk_scale=None,attention=0,depth_index=0):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.resolution = resolution
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        
        self.scale=1
        self.bias_level = 0.1

        self.sigmoid = nn.Sigmoid()
        self.d_idx = depth_index
        self.idx = idx 
        self.relu = nn.ReLU()
        if attention == 0:
            self.attention = 'L'
        
        if self.attention == 'L':
            # We assume split_size (stripe_with) is 1   
            assert self.split_size == 1, "split_size is not 1" 

            if idx == 0:  # Horizontal Self-Attention
                W_sp, H_sp = self.resolution[1], self.split_size
                self.RPE = HRPE_LUT[self.d_idx]
            elif idx == 1:  # Vertical Self-Attention
                H_sp, W_sp = self.resolution[0], self.split_size
                self.RPE = VRPE_LUT[self.d_idx]
            else:
                print ("ERROR MODE", idx)
                exit(0)


        self.H_sp = H_sp
        self.W_sp = W_sp
        stride = 1
        self.attn_drop = nn.Dropout(attn_drop)

    def im2hvwin(self, x):
        B, N, C = x.shape
        # H = W = int(np.sqrt(N))
        H, W = self.resolution[0], self.resolution[1]
        x = x.transpose(-2,-1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp* self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()  # B, H//H_sp, W//W_sp, H_sp * W_sp, C -> B, H//H_sp, W//W_sp, H_sp*W_sp, heads, C//heads
        return x

    def get_v(self, x): # LePE is not used for EGformer
        B, N, C = x.shape
        # H = W = int(np.sqrt(N))
        H, W = self.resolution[0], self.resolution[1]
        x = x.transpose(-2,-1).contiguous().view(B, C, H, W)

        H_sp, W_sp = self.H_sp, self.W_sp
        x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp) ### B', C, H', W'

        x = x.reshape(-1, self.num_heads, C // self.num_heads, self.H_sp* self.W_sp).permute(0, 1, 3, 2).contiguous()
        return x

    def forward(self, qkv,res_x):
        """
        x: B L C
        """
        q,k,v = qkv[0], qkv[1], qkv[2]
        
        pi = torch.acos(torch.zeros(1)).item() * 2

        ### Img2Window
        # H = W = self.resolution
        H, W = self.resolution[0], self.resolution[1]
        B, L, C = q.shape
        assert L == H * W, "flatten img_tokens has wrong size"
        
        q = self.im2hvwin(q)
        k = self.im2hvwin(k)
        v = self.get_v(v)

        if self.attention == 'L': 
            self.RPE = self.RPE.cuda(q.get_device())
            
            Re = int(q.size(0) / self.RPE.size(0))
    
            attn = q @ k.transpose(-2, -1)
            
            # ERPE
            attn = attn + self.bias_level * self.RPE.repeat(Re,1,1,1)
 
            M = torch.abs(attn) # Importance level of each local attention

            # DAS
            attn = F.normalize(attn,dim=-1) * pi/2
            attn = (1 - torch.cos(attn)) # Square of the distance from the baseline point. By setting the baseline point as (1/sqrt(2),0,pi/2), DAS can get equal score range (0,1) for both vertical & horizontal direction. 

            # EaAR
            M = torch.mean(M,dim=(1,2,3),keepdim=True)   # Check this part to utilize batch size > 1 per GPU.
           
            M = M / torch.max(M)
            M = torch.clamp(M, min=0.5)

            attn = attn * M  

            attn = self.attn_drop(attn)

            x = (attn @ v) 

        x = x.transpose(1, 2).reshape(-1, self.H_sp* self.W_sp, C)  # B head N N @ B head N C

        x = windows2img(x, self.H_sp, self.W_sp, H, W).view(B, -1, C)  # B H' W' C
        
        # EaAR
        res_x = res_x.reshape(-1,self.H_sp*self.W_sp,C).unsqueeze(1)
        res_x = res_x * (1 - M)
        res_x = res_x.view(B,-1,C)


        return x + res_x


class EGBlock(nn.Module):

    def __init__(self, dim, reso, num_heads,
                 split_size=7, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 last_stage=False,attention=0,idx=0,depth_index=0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.patches_resolution = reso
        self.split_size = split_size
        self.mlp_ratio = mlp_ratio
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm1 = norm_layer(dim)
        
        self.branch_num = 1
       
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        self.idx = idx
        self.attention = attention
       
        self.attns = nn.ModuleList([
            EGAttention(
                dim, resolution=self.patches_resolution, idx = self.idx,
                split_size=split_size, num_heads=num_heads, dim_out=dim,
                qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,attention=self.attention, depth_index = depth_index)
            for i in range(self.branch_num)])
        

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """

        # H = W = self.patches_resolution
        H, W = self.patches_resolution[0], self.patches_resolution[1]
        B, L, C = x.shape
        assert L == H * W, "flatten img_tokens has wrong size"
        img = self.norm1(x)
        qkv = self.qkv(img).reshape(B, -1, 3, C).permute(2, 0, 1, 3)
        
        attened_x = self.attns[0](qkv,x)

        attened_x = self.proj(attened_x)
        x = x + self.drop_path(attened_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

def img2windows(img, H_sp, W_sp):
    """
    img: B C H W
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp* W_sp, C)  # B, H//H_sp, W//W_sp, H_sp * W_sp, C
    return img_perm

def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    img_splits_hw: B' H W C
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img

class To_BCHW(nn.Module):
    def __init__(self, resolution, norm_layer=nn.LayerNorm):
        super().__init__()
        
        self.resolution = resolution
        self.gelu = nn.GELU()
    def forward(self, x):
        B, new_HW, C = x.shape
        # H = W = int(np.sqrt(new_HW))
        H, W = self.resolution[0], self.resolution[1]
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        B, C = x.shape[:2]
        x = x.view(B, C, -1).transpose(-2, -1).contiguous()
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        
        return x

class Tune_Block_Final(nn.Module):
    def __init__(self, dim, dim_out, resolution, norm_layer=nn.LayerNorm):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim_out, 3, 1, 1)
        self.norm = norm_layer(dim_out)
        self.resolution = resolution
        self.gelu = nn.GELU()
    def forward(self, x):
        B, new_HW, C = x.shape
#        # H = W = int(np.sqrt(new_HW))
        H, W = self.resolution[0], self.resolution[1]
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = self.conv(x)
        B, C = x.shape[:2]
        x = x.view(B, C, -1).transpose(-2, -1).contiguous()
        x = self.gelu(x)
        x = self.norm(x)
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        
        return x

class Tune_Block(nn.Module):
    def __init__(self, dim, dim_out, resolution, norm_layer=nn.LayerNorm):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim_out, 3, 1, 1)
        self.norm = norm_layer(dim_out)
        self.resolution = resolution
        self.gelu = nn.GELU()
    def forward(self, x):
        H, W = self.resolution[0], self.resolution[1]
        x = self.conv(x)
        B, C = x.shape[:2]
        x = x.view(B, C, -1).transpose(-2, -1).contiguous()
        x = self.gelu(x)
        x = self.norm(x)
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        
        return x

class Downdim(nn.Module):
    def __init__(self, in_channel, out_channel, reso=None):
        super().__init__()
        self.input_resolution = reso
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        B, L, C = x.shape
        H, W = self.input_resolution
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.conv(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        return out

class Merge_Block(nn.Module):
    def __init__(self, dim, dim_out, resolution, norm_layer=nn.LayerNorm,scale_factor=0.5):
        super().__init__()

        if scale_factor < 1.:
            self.conv = nn.Conv2d(dim, dim_out, 3, 2, 1)
        elif scale_factor > 1.:
            self.conv = nn.Conv2d(dim, dim_out, 3, 1, 1)
        elif scale_factor == 1.:
            self.conv = nn.Conv2d(dim, dim_out, 3, 1, 1)
        self.scale_factor = scale_factor   
        self.norm = norm_layer(dim_out)
        self.resolution = resolution
        self.gelu = nn.GELU()
 
    def forward(self, x):
        B, new_HW, C = x.shape
        # H = W = int(np.sqrt(new_HW))
        H, W = self.resolution[0], self.resolution[1]
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        if self.scale_factor >1.:
            x = F.interpolate(x,scale_factor=self.scale_factor)
        x = self.conv(x)
        x = self.gelu(x)
        B, C = x.shape[:2]
        x = x.view(B, C, -1).transpose(-2, -1).contiguous()
        x = self.norm(x)
        
        return x

class EGTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    Args:
        depth       : Number of blocks in each stage
        split_size  : Width(Height) of stripe size in each stage
        num_heads   : Number of heads in each stage
        hybrid      : Whether to use hybrid patch embedding (ResNet50)/ Not used
    """
    def __init__(self, img_size=[512, 1024], patch_size=16, in_chans=3, num_classes=1000, embed_dim=96, depth=[2,2,6,2], split_size = [3,5,7],
                 num_heads=12, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, use_chk=False, hybrid=False):
        super().__init__()
        self.use_chk = use_chk
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        heads=num_heads
        

        self.patch_embed= nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, 3, 2, 1),
            Rearrange('b c h w -> b (h w) c', h = img_size[0]//2, w = img_size[1]//2),
            nn.LayerNorm(embed_dim)
        )

        non_negative=True 

        #### Panoformer variables
        img_size_pano=256; in_chans=3; embed_dim=32; depths=[2, 2, 2, 2, 2, 2, 2, 2, 2]; num_heads=[1, 2, 4, 8, 16, 16, 8, 4, 2]
        win_size=8; mlp_ratio=4.; qkv_bias=True; qk_scale=None; drop_rate=0.; attn_drop_rate=0.; drop_path_rate=0.1
        norm_layer=nn.LayerNorm; patch_norm=True; use_checkpoint=False; token_projection='linear'; token_mlp='leff'; se_layer=False
        dowsample=Downsample; upsample=Upsample

        self.num_enc_layers = len(depths) // 2
        self.num_dec_layers = len(depths) // 2
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.token_projection = token_projection
        self.mlp = token_mlp
        self.win_size = win_size
        self.ref_point256x512 = genSamplingPattern(256, 512, 3, 3).cuda()
        self.ref_point128x256 = genSamplingPattern(128, 256, 3, 3).cuda()
        self.ref_point64x128 = genSamplingPattern(64, 128, 3, 3).cuda()
        self.ref_point32x64 = genSamplingPattern(32, 64, 3, 3).cuda()
        self.ref_point16x32 = genSamplingPattern(16, 32, 3, 3).cuda()

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[:self.num_enc_layers]))]
        conv_dpr = [drop_path_rate] * depths[4]
        dec_dpr = enc_dpr[::-1]


        curr_dim = embed_dim
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, np.sum(depth))]  # stochastic depth decay rule
 
        self.stage1 = nn.ModuleList([
            BasicPanoformerLayer(dim=curr_dim, output_dim=curr_dim,
                                                input_resolution=(img_size[0]//2, img_size[1]//2),
                                                depth=depths[0],
                                                num_heads=num_heads[0],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=enc_dpr[int(sum(depths[:0])):int(sum(depths[:1]))],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                se_layer=se_layer, ref_point=self.ref_point256x512, flag = 0) if i%2==0 else 
            EGBlock(
                dim=curr_dim, num_heads=heads[0], reso=[img_size[0]//2, img_size[1]//2], mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[0],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer,attention = 0, idx= i%2, depth_index = 0)
            for i in range(depth[0])])
        self.downsample1 = Merge_Block(curr_dim, curr_dim *2 , resolution = [img_size[0]//2, img_size[1]//2])
       
        # Tuning into decoder dimension

        curr_dim = curr_dim*2
        self.stage2 = nn.ModuleList([
            BasicPanoformerLayer(dim=curr_dim, output_dim=curr_dim,
                                                input_resolution=(img_size[0]//4, img_size[1]//4),
                                                depth=depths[1],
                                                num_heads=num_heads[1],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=enc_dpr[sum(depths[:1]):sum(depths[:2])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                se_layer=se_layer, ref_point=self.ref_point128x256, flag = 0) if i%2==0 else 
            EGBlock(
                dim=curr_dim, num_heads=heads[1], reso=[img_size[0]//4, img_size[1]//4], mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[1],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:1])+i], norm_layer=norm_layer,attention = 0, idx= i%2, depth_index = 1)
            for i in range(depth[1])])
        self.downsample2 = Merge_Block(curr_dim, curr_dim*2, resolution = [img_size[0]//4, img_size[1]//4])
       
        # Tuning into decoder dimension
        curr_dim = curr_dim*2

        self.stage3 = nn.ModuleList([
            EGBlock(
                dim=curr_dim, num_heads=heads[2], reso=[img_size[0]//8, img_size[1]//8], mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[2],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:2])+i], norm_layer=norm_layer,attention = 0 , idx= i%2, depth_index = 2)
            for i in range(depth[2])])

        self.downsample3 = Merge_Block(curr_dim, curr_dim*2, resolution = [img_size[0]//8, img_size[1]//8])

        curr_dim = curr_dim*2
        self.stage4 = nn.ModuleList([
            EGBlock(
                dim=curr_dim, num_heads=heads[3], reso=[img_size[0]//16, img_size[1]//16], mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[3],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:3])+i], norm_layer=norm_layer, last_stage=False,attention = 0 , idx= i%2, depth_index = 3)
            for i in range(depth[3])])
 
        self.downsample4 = Merge_Block(curr_dim, curr_dim * 2, resolution = [img_size[0]//16, img_size[1]//16])
        curr_dim = curr_dim*2
 
        self.bottleneck = nn.ModuleList([
            EGBlock(
                dim=curr_dim, num_heads=heads[-1], reso=[img_size[0]//32, img_size[1]//32], mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[-1],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:-1])+i], norm_layer=norm_layer, last_stage=False,attention = 0, idx= i%2, depth_index = 4)
            for i in range(depth[-1])])
 
        self.upsample5 = Merge_Block(curr_dim, curr_dim // 2, resolution = [img_size[0]//32, img_size[1]//32],scale_factor=2.)
        curr_dim = curr_dim // 2



        self.red_ch = []
        self.set_dim = []
        self.rearrange = []
        curr_dim = curr_dim
        self.dec_stage5 = nn.ModuleList(
            [EGBlock(
                dim=curr_dim, num_heads=heads[4], reso=[img_size[0]//16, img_size[1]//16], mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[4],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:4])+i], norm_layer=norm_layer, last_stage=False, attention = 0, idx= i%2, depth_index = 3)
            for i in range(depth[4])])

        self.upsample6 = Merge_Block(curr_dim, curr_dim // 2, resolution = [img_size[0]//16, img_size[1]//16],scale_factor=2.)

        self.tune5 = Tune_Block(curr_dim * 2  ,curr_dim, resolution = [img_size[0]//16, img_size[1]//16]) # Tune_5
        self.set_dim.append(To_BCHW(resolution = [img_size[0]//16, img_size[1]//16])) # BCHW_5
        self.rearrange.append(Rearrange('b c h w -> b (h w) c', h = img_size[0]//16, w = img_size[1]//16))


        
        curr_dim = curr_dim // 2
        self.dec_stage6 = nn.ModuleList(
            [EGBlock(
                dim=curr_dim, num_heads=heads[5], reso=[img_size[0]//8, img_size[1]//8], mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[5],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:5])+i], norm_layer=norm_layer,attention = 0, idx= i%2, depth_index = 2)
            for i in range(depth[5])])

        self.upsample7 = Merge_Block(curr_dim , curr_dim //2, resolution = [img_size[0]//8, img_size[1]//8],scale_factor=2.)
 
        self.tune6 = Tune_Block(curr_dim * 2  ,curr_dim, resolution = [img_size[0]//8, img_size[1]//8]) # Tune_6

        self.set_dim.append(To_BCHW(resolution = [img_size[0]//8, img_size[1]//8])) # BCHW_6
        self.rearrange.append(Rearrange('b c h w -> b (h w) c', h = img_size[0]//8, w = img_size[1]//8))
        
        
        curr_dim = curr_dim // 2
        self.dec_stage7 = nn.ModuleList([
            BasicPanoformerLayer(dim=curr_dim, output_dim=curr_dim,
                                                input_resolution=(img_size[0]//4, img_size[1]//4),
                                                depth=depths[7],
                                                num_heads=num_heads[7],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=dec_dpr[sum(depths[5:7]):sum(depths[5:8])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                se_layer=se_layer,ref_point=self.ref_point128x256, flag = 1) if i%2==0 else
            EGBlock(
                dim=curr_dim, num_heads=heads[6], reso=[img_size[0]//4, img_size[1]//4], mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[6],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:6])+i], norm_layer=norm_layer,attention=0, idx= i%2, depth_index = 1)
            for i in range(depth[6])])
        
        self.upsample8 = Merge_Block(curr_dim , curr_dim//2, resolution = [img_size[0]//4, img_size[1]//4],scale_factor=2.)
        
        self.tune7 = Tune_Block(curr_dim * 2  ,curr_dim, resolution = [img_size[0]//4, img_size[1]//4]) # Tune_7
        self.set_dim.append(To_BCHW(resolution = [img_size[0]//4, img_size[1]//4])) # BCHW_7
        self.rearrange.append(Rearrange('b c h w -> b (h w) c', h = img_size[0]//4, w = img_size[1]//4))


        curr_dim = curr_dim // 2
        self.dec_stage8 = nn.ModuleList([
            BasicPanoformerLayer(dim=curr_dim, output_dim=curr_dim,
                                                input_resolution=(img_size[0]//2, img_size[1]//2),
                                                depth=depths[8],
                                                num_heads=num_heads[8],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=dec_dpr[sum(depths[5:8]):sum(depths[5:9])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                se_layer=se_layer, ref_point=self.ref_point256x512, flag = 1) if i%2==0 else
            EGBlock(
                dim=curr_dim, num_heads=heads[7], reso=[img_size[0]//2, img_size[1]//2], mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[7],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:7])+i], norm_layer=norm_layer,attention=0, idx= i%2, depth_index = 0)
            for i in range(depth[7])])

        
        self.tune8 = Tune_Block(curr_dim * 2  ,curr_dim, resolution = [img_size[0]//2, img_size[1]//2]) # Tune_8
        self.set_dim.append(To_BCHW(resolution = [img_size[0]//2, img_size[1]//2])) # BCHW_8
        self.rearrange.append(Rearrange('b c h w -> b (h w) c', h = img_size[0]//2, w = img_size[1]//2))
    
        self.tune_final = Tune_Block_Final(curr_dim,curr_dim, resolution = [img_size[0]//2, img_size[1]//2])
        # Tuning into decoder dimension

        self.norm = norm_layer(curr_dim)

        self.output_conv = nn.Sequential(
            nn.Conv2d(curr_dim, curr_dim, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(curr_dim, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            # nn.Sigmoid() if non_negative else nn.Identity(),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        features = []
        B = x.shape[0]
        features= []
       

    ########## Encoder       
        x = self.patch_embed(x)
        
        for blk in self.stage1:
            x = blk(x)
        features.append(x)
        x = self.downsample1(x)

        for blk in self.stage2:
            x = blk(x)
        features.append(x)
        x = self.downsample2(x)

        for blk in self.stage3:
            x = blk(x)
        features.append(x)
        x = self.downsample3(x)
        
        for blk in self.stage4:
            x = blk(x)
        features.append(x)
        x = self.downsample4(x)
        
        for blk in self.bottleneck:
            x = blk(x)

    ######## Decoder
        x = self.upsample5(x)
        for blk in self.dec_stage5:
            x = blk(x)
        x = torch.cat((self.set_dim[0](features[3]), self.set_dim[0](x)), dim=1)
        x = self.tune5(x)
        x = self.rearrange[0](x)

        x = self.upsample6(x)
        for blk in self.dec_stage6:
            x = blk(x)
        x = torch.cat((self.set_dim[1](features[2]), self.set_dim[1](x)), dim=1)
        x = self.tune6(x)
        x = self.rearrange[1](x)

        x = self.upsample7(x)
        for blk in self.dec_stage7:
            x = blk(x)
        x = torch.cat((self.set_dim[2](features[1]), self.set_dim[2](x)), dim=1)
        x = self.tune7(x)
        x = self.rearrange[2](x)

        x = self.upsample8(x)
        for blk in self.dec_stage8:
            x = blk(x)
        x = torch.cat((self.set_dim[3](features[0]), self.set_dim[3](x)), dim=1)
        x = self.tune8(x)
        x = self.rearrange[3](x)

        # EGformer Output Projection
        x = self.tune_final(x)
        x = self.output_conv(x)

        return x

    def forward(self, x):
        out = self.forward_features(x)
        return {'pred_depth': out}




@register_model
def EGDepthModel(pretrained=False, split_size=[1,1,1,1,1,1,1,1,1], img_size=[512, 1024], num_heads=[1,2,4,8,8,4,2,1,16], **kwargs):
    model = EGTransformer(patch_size=4, embed_dim=32, depth=[2,2,2,2,2,2,2,2,2],
        split_size=split_size, num_heads=num_heads, mlp_ratio=4., img_size=img_size, **kwargs)
    return model

