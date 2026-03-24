"""model.py - Model and module class for ViT.
   They are built to mirror those in the official Jax implementation.
"""

from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F
import math

from .transformer_layer import Transformer_ST_TDC_gra_sharp
from .pfe_function import MLP
import utils as utils

import pdb


def as_tuple(x):
    return x if isinstance(x, tuple) else (x, x)

'''
Temporal Center-difference based Convolutional layer (3D version)
theta: control the percentage of original convolution and centeral-difference convolution
'''
class CDC_T(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.6):

        super(CDC_T, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            # pdb.set_trace()
            [C_out, C_in, t, kernel_size, kernel_size] = self.conv.weight.shape

            # only CD works on temporal kernel size>1
            if self.conv.weight.shape[2] > 1:
                kernel_diff = self.conv.weight[:, :, 0, :, :].sum(2).sum(2) + self.conv.weight[:, :, 2, :, :].sum(
                    2).sum(2)
                kernel_diff = kernel_diff[:, :, None, None, None]
                out_diff = F.conv3d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride,
                                    padding=0, dilation=self.conv.dilation, groups=self.conv.groups)
                return out_normal - self.theta * out_diff

            else:
                return out_normal

# stem_3DCNN + ST-ViT with local Depthwise Spatio-Temporal MLP
class ViT_ST_ST_Compact3_TDC_gra_sharp(nn.Module):

    def __init__(
        self, 
        name: Optional[str] = None, 
        pretrained: bool = False, 
        patches: int = 16,
        dim: int = 768,
        ff_dim: int = 3072,
        num_heads: int = 12,
        num_layers: int = 12,
        attention_dropout_rate: float = 0.0,
        dropout_rate: float = 0.2,
        representation_size: Optional[int] = None,
        load_repr_layer: bool = False,
        classifier: str = 'token',
        #positional_embedding: str = '1d',
        in_channels: int = 3, 
        frame: int = 160,
        theta: float = 0.2,
        image_size: Optional[int] = None,
        pfe_hidden: int = 128,
        pfe_output_size: int = 4,
    ):
        super().__init__()


        self.image_size = image_size  
        self.frame = frame  
        self.dim = dim              

        # Image and patch sizes
        t, h, w = as_tuple(image_size)  # tube sizes
        ft, fh, fw = as_tuple(patches)  # patch sizes, ft = 4 ==> 160/4=40
        gt, gh, gw = t//ft, h // fh, w // fw  # number of patches
        seq_len = gh * gw * gt

        # Patch embedding    [4x16x16]conv
        self.patch_embedding = nn.Conv3d(dim, dim, kernel_size=(ft, fh, fw), stride=(ft, fh, fw))
        
        # Transformer
        self.transformer1 = Transformer_ST_TDC_gra_sharp(num_layers=num_layers//3, dim=dim, num_heads=num_heads, 
                                       ff_dim=ff_dim, dropout=dropout_rate, theta=theta)
        # Transformer
        self.transformer2 = Transformer_ST_TDC_gra_sharp(num_layers=num_layers//3, dim=dim, num_heads=num_heads, 
                                       ff_dim=ff_dim, dropout=dropout_rate, theta=theta)
        # Transformer
        self.transformer3 = Transformer_ST_TDC_gra_sharp(num_layers=num_layers//3, dim=dim, num_heads=num_heads, 
                                       ff_dim=ff_dim, dropout=dropout_rate, theta=theta)
        
        
        
        self.Stem0 = nn.Sequential(
            nn.Conv3d(3, dim//4, [1, 5, 5], stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(dim//4),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )
        
        self.Stem1 = nn.Sequential(
            nn.Conv3d(dim//4, dim//2, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(dim//2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )
        self.Stem2 = nn.Sequential(
            nn.Conv3d(dim//2, dim, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )
        
        #self.normLast = nn.LayerNorm(dim, eps=1e-6)
        
        self.pfe_hidden = pfe_hidden
        self.pfe_output_size = pfe_output_size
        imnet_in_dim = dim * 9 + 2
        self.mlp_pfe = MLP(
            in_dim=imnet_in_dim,
            out_dim=dim,
            hidden_list=[pfe_hidden, pfe_hidden],
        )
        self.pfe_proj = nn.Sequential(
            nn.Conv3d(dim, dim // 2, kernel_size=1),
            nn.BatchNorm3d(dim // 2),
            nn.ELU(),
        )
        self._pfe_coord_cache = {}
 
        self.ConvBlockLast = nn.Conv1d(dim//2, 1, 1,stride=1, padding=0)
        
        self.ConvBlock1 = nn.Sequential( 
            nn.Conv3d(3, 3, [1, 5, 5], stride=1, padding=[0, 2, 2]),
            nn.BatchNorm3d(3),
            nn.ReLU(inplace=True),
        )
        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))

        # Initialize weights
        self.init_weights()
        
    @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # _trunc_normal(m.weight, std=0.02)  # from .initialization import _trunc_normal
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)  # nn.init.constant(m.bias, 0)
        self.apply(_init)


    def forward(self, x, gra_sharp,size):
        # downsample raw view1 view2
        # print("Input size",x.shape)
        # print("Input size", size)
        if size == 32: 
            x = self.ConvBlock1(x)  # [B, 16, T, H, W] -> [B, 3, T, H, W] 
            x = self.MaxpoolSpa(x)  # [B, 16, T, H/2, W/2] -> [B, 3, T, H/2, W/2]
            x = self.ConvBlock1(x)  # [B, 16, T, H, W] -> [B, 3, T, H, W] 
            x = self.MaxpoolSpa(x)  # [B, 16, T, H/2, W/2] -> [B, 3, T, H/2, W/2]
        elif size == 64:
            x = self.ConvBlock1(x)  # [B, 16, T, H, W] -> [B, 3, T, H, W] 
            x = self.MaxpoolSpa(x)  # [B, 16, T, H/2, W/2] -> [B, 3, T, H/2, W/2]
        else:
            pass 
        
        b, c, t, fh, fw = x.shape
        # >> 128: (1, 3, 160, 128, 128) | 64: (1, 3, 160, 64, 64) |32: (1, 3, 160, 32, 32)
        
        # Stem Module
        x = self.Stem0(x); x = self.Stem1(x); x = self.Stem2(x)
        x = self.patch_embedding(x)
        x = x.flatten(2).transpose(1, 2)
        
        # Transfomer Module
        Trans_features, Score1 =  self.transformer1(x, gra_sharp)
        # >> 128: [1, 640, 96] | 64: [1, 160, 96] |32: [1, 40, 96]
        Trans_features2, Score2 =  self.transformer2(Trans_features, gra_sharp)
        # >> 128: [1, 640, 96] | 64: [1, 160, 96] |32: [1, 40, 96]
        Trans_features3, Score3 =  self.transformer3(Trans_features2, gra_sharp)
        # >> 128: [1, 640, 96] | 64: [1, 160, 96] |32: [1, 40, 96]
        
        # downsample後的shape無法通過原版的transpose整除判斷，會出現shape對不上問題
        # features_last = Trans_features3.transpose(1, 2).view(b, self.dim, t//4, 4, 4) # [B, 64, 40, 4, 4]  = Not Working!!!!
        # 修正downsample後的shape問題，通過彈性調整在保持self.dim是64情況下，後面的shape可以對上
        wh_size = {128:4 ,64:2, 32:1}
        wh_size = wh_size[size]
        features_last = Trans_features3.transpose(1, 2).view(b, self.dim, -1, wh_size, wh_size) # fixed
        # >> 128: [1, 96, 40, 4, 4] | 64: [1, 96, 40, 2, 2] |32: [1, 96, 40, 1, 1]
        
        # Feature Upsample via PFE
        features_last = self._pfe_upsample(features_last, self.pfe_output_size)
        features_last = self.pfe_proj(features_last)
        temporal_scale = self.frame / features_last.shape[2]
        if temporal_scale != 1:
            features_last = F.interpolate(
                features_last,
                scale_factor=(temporal_scale, 1, 1),
                mode='trilinear',
                align_corners=False,
            )
        features_last = torch.mean(features_last,3)
        features_last = torch.mean(features_last,3)
        rPPG = self.ConvBlockLast(features_last)

        rPPG = rPPG.squeeze(1)
        return rPPG, Score1, Score2, Score3 ,Trans_features, Trans_features2

    def _get_pfe_coord(self, output_size: int, device: torch.device) -> torch.Tensor:
        cache_key = (output_size, device)
        coord = self._pfe_coord_cache.get(cache_key)
        if coord is None:
            coord = utils.make_coord([output_size, output_size]).to(device)
            self._pfe_coord_cache[cache_key] = coord
        return coord

    def _pfe_query(self, x, coord, cell=None):
        feat = F.unfold(x, 3, padding=1).view(
            x.shape[0], x.shape[1] * 9, x.shape[2], x.shape[3]
        )
        coord_ = coord.clone()
        q_feat = F.grid_sample(
            feat, coord_.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False
        )[:, :, 0, :].permute(0, 2, 1)

        rel_cell = cell.clone()
        rel_cell[:, :, 0] *= feat.shape[-2]
        rel_cell[:, :, 1] *= feat.shape[-1]
        inp = torch.cat([q_feat, rel_cell], dim=-1)

        bs, q = coord.shape[:2]
        pred = self.mlp_pfe(inp.view(bs * q, -1)).view(bs, q, -1)
        return pred

    def _pfe_upsample(self, x, output_size: int):
        batch, channel, length, width, height = x.shape
        coord = self._get_pfe_coord(output_size, x.device)
        coord = coord.expand(batch * length, -1, -1)

        cell = torch.ones_like(coord)
        cell[:, 0] *= 2 / output_size
        cell[:, 1] *= 2 / output_size
        cell = cell.expand(batch * length, -1, -1)

        feat = (
            x.permute(0, 2, 1, 3, 4)
            .contiguous()
            .view(-1, channel, width, height)
        )
        ret = self._pfe_query(feat, coord, cell=cell)
        ret = (
            ret.permute(0, 2, 1)
            .contiguous()
            .view(batch * length, channel, output_size, output_size)
            .view(batch, length, channel, output_size, output_size)
        )
        return ret.permute(0, 2, 1, 3, 4).contiguous()
