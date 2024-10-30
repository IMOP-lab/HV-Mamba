from .unet_parts import *

import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import trunc_normal_, DropPath
import os
import sys
import torch.fft
from timm import create_model

import math
from .vmamba import SS2D
from .vmamba import VSSBlock
import traceback
import torch_geometric
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from collections import OrderedDict
from .encoder import FeatureExtractor
# from.upsample import IFA

class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, padding=1, stride=1, dilation=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding, 
                      stride=stride, dilation=dilation, groups=dim_in)
        self.norm_layer = nn.GroupNorm(4, dim_in)
        self.conv2 = nn.Conv2d(dim_in, dim_out, kernel_size=1)

    def forward(self, x):
        return self.conv2(self.norm_layer(self.conv1(x)))

if 'DWCONV_IMPL' in os.environ:
    try:
        sys.path.append(os.environ['DWCONV_IMPL'])
        from DepthwiseConv2d.depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM
        def get_dwconv(dim, kernel, bias):
            return DepthWiseConv2dImplicitGEMM(dim, kernel, bias)
    except:
        print(traceback.format_exc())
        def get_dwconv(dim, kernel, bias):
            return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel-1)//2 ,bias=bias, groups=dim)
else:
    def get_dwconv(dim, kernel, bias):
            return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel-1)//2 ,bias=bias, groups=dim)

class H_SS2D(nn.Module):
    def __init__(self, dim, order=5, gflayer=None, h=14, w=8, s=1.0, d_state=16):
        super().__init__()
        self.order = order
        self.dims = [dim // 2 ** i for i in range(order)]
        self.dims.reverse()
        self.proj_in = nn.Conv2d(dim, 2*dim, 1)

        if gflayer is None:
            self.dwconv = get_dwconv(sum(self.dims), 7, True)
        else:
            self.dwconv = gflayer(sum(self.dims), h=h, w=w)
        
        self.proj_out = nn.Conv2d(dim, dim, 1)

        self.pws = nn.ModuleList(
            [nn.Conv2d(self.dims[i], self.dims[i+1], 1) for i in range(order-1)]
        )

        num = len(self.dims)
        if num == 2:
            self.ss2d_1 = SS2D(d_model=self.dims[1], dropout=0, d_state=16) 
        elif num == 3:
            self.ss2d_1 = SS2D(d_model=self.dims[1], dropout=0, d_state=16) 
            self.ss2d_2 = SS2D(d_model=self.dims[2], dropout=0, d_state=16) 
        elif num == 4:
            self.ss2d_1 = SS2D(d_model=self.dims[1], dropout=0, d_state=16) 
            self.ss2d_2 = SS2D(d_model=self.dims[2], dropout=0, d_state=16) 
            self.ss2d_3 = SS2D(d_model=self.dims[3], dropout=0, d_state=16) 
        elif num == 5:
            self.ss2d_1 = SS2D(d_model=self.dims[1], dropout=0, d_state=16) 
            self.ss2d_2 = SS2D(d_model=self.dims[2], dropout=0, d_state=16) 
            self.ss2d_3 = SS2D(d_model=self.dims[3], dropout=0, d_state=16) 
            self.ss2d_4 = SS2D(d_model=self.dims[4], dropout=0, d_state=16) 

        self.ss2d_in = SS2D(d_model=self.dims[0], dropout=0, d_state=16)

        self.scale = s

        print('[H_SS2D]', order, 'order with dims=', self.dims, 'scale=%.4f'%self.scale)


    def forward(self, x, mask=None, dummy=False):
        B, C, H, W = x.shape

        fused_x = self.proj_in(x)
        pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)

        dw_abc = self.dwconv(abc) * self.scale

        dw_list = torch.split(dw_abc, self.dims, dim=1)
        x = pwa * dw_list[0]
        x = x.permute(0, 2, 3, 1)
        x = self.ss2d_in(x)
        x = x.permute(0, 3, 1, 2)

        for i in range(self.order -1):
            x = self.pws[i](x) * dw_list[i+1]
            if i == 0 :
                x = x.permute(0, 2, 3, 1)
                x = self.ss2d_1(x)
                x = x.permute(0, 3, 1, 2)
            elif i == 1 :
                x = x.permute(0, 2, 3, 1)
                x = self.ss2d_2(x)
                x = x.permute(0, 3, 1, 2)
            elif i == 2 :
                x = x.permute(0, 2, 3, 1)
                x = self.ss2d_3(x)
                x = x.permute(0, 3, 1, 2)
            elif i == 3 :
                x = x.permute(0, 2, 3, 1)
                x = self.ss2d_4(x)
                x = x.permute(0, 3, 1, 2)                
        x = self.proj_out(x)
        return x
   
class self_SelectiveKernelUnit(nn.Module):
    def __init__(self, channels, num_heads, kernels=[1, 3, 5, 7], reduction=16, group=1, L=32, dynamic_ratio=4):
        super().__init__()
        self.num_heads = num_heads
        self.channels = channels
        self.head_dim = channels // num_heads
        self.dynamic_ratio = dynamic_ratio
        self.scale = 1 / (self.head_dim ** 0.5)
        self.qkv_conv = nn.Conv2d(channels, 3 * channels, kernel_size=1, bias=False)
        self.output_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.dynamic_weights = nn.Parameter(torch.Tensor(num_heads, self.head_dim // dynamic_ratio, self.head_dim))
        nn.init.xavier_uniform_(self.dynamic_weights)
        self.d = max(L, channels // reduction)
        self.convs = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(channels, channels, kernel_size=k, padding=k//2, groups=group)),
                ('bn', nn.BatchNorm2d(channels)),
                ('relu', nn.ReLU())
            ])) for k in kernels
        ])
        self.fc = nn.Linear(channels, self.d)
        self.fcs = nn.ModuleList([nn.Linear(self.d, channels) for _ in kernels])
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        bsz, ch, ht, wd = x.shape
        qkv = self.qkv_conv(x)
        q, k, v = torch.split(qkv, self.channels, dim=1)
        q = q.view(bsz, self.num_heads, self.head_dim, ht * wd).permute(0, 1, 3, 2)
        k = k.view(bsz, self.num_heads, self.head_dim, ht * wd).permute(0, 1, 3, 2)
        v = v.view(bsz, self.num_heads, self.head_dim, ht * wd).permute(0, 1, 3, 2)
        scores = torch.einsum('bnid,bnjd->bnij', q, k) * self.scale
        attention_probs = torch.softmax(scores, dim=-1)
        context = torch.einsum('bnij,bnjd->bnid', attention_probs, v)
        context = context.permute(0, 1, 3, 2).contiguous().view(bsz, -1, ht, wd)
        output = self.output_proj(context)
        conv_outs = [conv(output) for conv in self.convs]
        feats = torch.stack(conv_outs, dim=0)
        U = torch.sum(feats, dim=0)
        S = U.mean([-2, -1]) 
        Z = self.fc(S)
        weights = [fc(Z).view(bsz, ch, 1, 1) for fc in self.fcs]
        attention_weights = self.softmax(torch.stack(weights, dim=0))
        V = torch.einsum('nbcwh,nbcwh->bcwh', attention_weights, feats)
        return V

class HarmonicStateSpaceModule(nn.Module):
    def __init__(self, channel):
        super(HarmonicStateSpaceModule, self).__init__()
        self.vss_block = VSSBlock(64, d_state=16)
        self.adjust_channels = nn.Conv2d(channel * 2, 64, kernel_size=1)

    def forward(self, x):
        x_fft = torch.fft.fftn(x, dim=(-2, -1))
        x_fft_shifted = torch.fft.fftshift(x_fft, dim=(-2, -1))
        x_fft_real_imag = torch.cat((x_fft_shifted.real, x_fft_shifted.imag), dim=1)
        x_fft_real_imag= self.adjust_channels(x_fft_real_imag)
        x_fft_attended = self.vss_block(x_fft_real_imag)
        real_part = x_fft_attended[:, :x_fft_attended.size(1)//2, :, :]
        imag_part = x_fft_attended[:, x_fft_attended.size(1)//2:, :, :]
        x_fft_attended_complex = torch.complex(real_part, imag_part)
        x_ifft_shifted = torch.fft.ifftshift(x_fft_attended_complex, dim=(-2, -1))
        y_ifft = torch.fft.ifftn(x_ifft_shifted, dim=(-2, -1))
        y = y_ifft.real
        return y
    
class DynamicRoutingLayer(nn.Module):
    def __init__(self, num_iterations=3):
        super().__init__()
        self.num_iterations = num_iterations

    def forward(self, capsules):
        batch_size, num_capsules, num_features = capsules.size()
        b_ij = torch.zeros(batch_size, num_capsules, num_capsules, device=capsules.device)

        for iteration in range(self.num_iterations):
            c_ij = F.softmax(b_ij, dim=2)
            s_j = (c_ij.unsqueeze(2) * capsules.unsqueeze(1)).sum(dim=1)
            v_j = self.squash(s_j)

            if iteration < self.num_iterations - 1:
                b_ij = b_ij + (capsules.unsqueeze(1) * v_j.unsqueeze(2)).sum(dim=-1)

        return v_j

    @staticmethod
    def squash(s):
        norm_sq = (s ** 2).sum(dim=-1, keepdim=True)
        scale = norm_sq / (1 + norm_sq) / torch.sqrt(norm_sq + 1e-8)
        return scale * s

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x   
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)      
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        return out

class FrequencyAttentionWithFeedback(nn.Module):
    def __init__(self, in_channels, num_iterations=3,reduction=32):
        super(FrequencyAttentionWithFeedback, self).__init__()
        self.in_channels = in_channels
        self.num_iterations = num_iterations
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.coord_att_real = CoordAtt(in_channels, in_channels, reduction=reduction)
        self.coord_att_imag = CoordAtt(in_channels, in_channels, reduction=reduction)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        fft_x = torch.fft.rfftn(x, dim=(-2, -1), norm='ortho')
        fft_x_real = fft_x.real
        fft_x_imag = fft_x.imag
        fft_x_real = self.coord_att_real(fft_x_real)
        fft_x_imag = self.coord_att_imag(fft_x_imag)
        for _ in range(self.num_iterations):
            attn_real = self.conv1(fft_x_real)
            attn_imag = self.conv1(fft_x_imag)
            attn = self.sigmoid(attn_real + attn_imag)
            fft_x_real = fft_x_real * attn
            fft_x_imag = fft_x_imag * attn
        fft_x = torch.complex(fft_x_real, fft_x_imag)
        ifft_x = torch.fft.irfftn(fft_x, s=x.shape[-2:], dim=(-2, -1), norm='ortho')
        return ifft_x

class HybridCoord_FreqAttention(nn.Module):
    def __init__(self, in_channels, reduction=32, num_iterations=3):
        super(HybridCoord_FreqAttention, self).__init__()
        self.coord_att = CoordAtt(in_channels, in_channels, reduction=reduction)
        self.freq_att = FrequencyAttentionWithFeedback(in_channels, num_iterations=num_iterations)

    def forward(self, x):
        x = self.freq_att(x)
        return x
    
class DilatedAttentionDecoder(nn.Module):
    def __init__(self, channel_list, attention_coefficients, final_out_channels=32, dilation_rate=2):       #
        super(DilatedAttentionDecoder, self).__init__()

        self.channel_conv = nn.ModuleList([
            nn.Conv2d(in_channels, final_out_channels, kernel_size=3, padding=dilation_rate, dilation=dilation_rate) for in_channels in channel_list
        ])

        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=True)
        self.attention_blocks = nn.ModuleList([
            AttentionBlock_attunet(final_out_channels, final_out_channels, attention_coefficients[i], dilation_rate=dilation_rate)
            for i in range(len(channel_list))
        ])

    def forward(self, features):
        processed_features = []
        for i, feature in enumerate(features):
            feature = self.channel_conv[i](feature)
            if i > 0:  
                attention_feature = self.attention_blocks[i](feature, processed_features[i-1])
            else:
                attention_feature = feature
            attention_feature = self.upsample(attention_feature)
            processed_features.append(attention_feature)
        out_feat = sum(processed_features) / len(processed_features)
        return out_feat 
    
class AttentionBlock_attunet(nn.Module):
    def __init__(self, F_g, F_l, n_coefficients, dilation_rate=2):
        super(AttentionBlock_attunet, self).__init__()
        self.W_gate = nn.Sequential(
            nn.Conv2d(F_g, n_coefficients, kernel_size=3, stride=1, padding=dilation_rate, dilation=dilation_rate, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, n_coefficients, kernel_size=3, stride=1, padding=dilation_rate, dilation=dilation_rate, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(n_coefficients, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip_connection):
        if gate.size()[2:] != skip_connection.size()[2:]:
            gate = F.interpolate(gate, size=skip_connection.size()[2:], mode='bilinear', align_corners=True)
        g1 = self.W_gate(gate)
        x1 = self.W_x(skip_connection)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return skip_connection * psi
    
class SimpleImplicitFeaturizer(torch.nn.Module):
    def __init__(self, n_freqs=20):
        super().__init__()
        self.n_freqs = n_freqs
        self.dim_multiplier = 2

    def forward(self, original_image):
        b, c, h, w = original_image.shape
        
        grid_h = torch.linspace(-1, 1, h, device=original_image.device)
        grid_w = torch.linspace(-1, 1, w, device=original_image.device)
        feats = torch.cat([t.unsqueeze(0) for t in torch.meshgrid([grid_h, grid_w])], dim=0).unsqueeze(0)

        feats = feats.expand(b, -1, -1, -1)

        feats = feats.unsqueeze(1)
        freqs = torch.exp(torch.linspace(-2, 10, self.n_freqs, device=original_image.device)).reshape(1, self.n_freqs, 1, 1, 1)
        feats = feats * freqs

        feats = feats.reshape(b, self.n_freqs * self.dim_multiplier, h, w)
        all_feats = [torch.sin(feats), torch.cos(feats), original_image]

        return torch.cat(all_feats, dim=1)

class IFA(torch.nn.Module):
    def __init__(self, feat_dim, num_scales=20):
        super().__init__()
        self.scales = 2 * torch.exp(torch.arange(1, num_scales + 1, dtype=torch.float32))
        self.feat_dim = feat_dim
        self.sin_feats = SimpleImplicitFeaturizer()
        
        self.mlp = nn.Sequential(
            nn.Conv2d(feat_dim + (num_scales * 4) + 2, feat_dim, 1),
            nn.BatchNorm2d(feat_dim),
            nn.LeakyReLU(),
            nn.Conv2d(feat_dim, feat_dim, 1),
        )

    def forward(self, source, guidance):
        b, c, h, w = source.shape
        
        up_source = F.interpolate(source, (h * 2, w * 2), mode="bilinear", align_corners=True)
        lr_cord = torch.linspace(0, h-1, steps=h, device=source.device)
        hr_cord = torch.linspace(0, h-1, steps=2 * h, device=source.device)
        lr_coords = torch.cat([x.unsqueeze(0) for x in torch.meshgrid(lr_cord, lr_cord)], dim=0).unsqueeze(0)
        hr_coords = torch.cat([x.unsqueeze(0) for x in torch.meshgrid(hr_cord, hr_cord)], dim=0).unsqueeze(0)
        
        up_lr_coords = F.interpolate(lr_coords, (h * 2, w * 2), mode="bilinear", align_corners=True)
        coord_diff = up_lr_coords - hr_coords

        coord_diff_feats = self.sin_feats(coord_diff)
        c2 = coord_diff_feats.shape[1]

        bcast_coord_feats = coord_diff_feats.expand(b, -1, -1, -1)
        
        return self.mlp(torch.cat([up_source, bcast_coord_feats], dim=1))

class IFAUpsampler(torch.nn.Module):
    def __init__(self, feat_dim, bilinear=False):
        super(IFAUpsampler, self).__init__()
        self.ifa = IFA(feat_dim)
        
        # Convolutional layers to refine the upsampled features
        self.conv = nn.Sequential(
            nn.Conv2d(feat_dim + feat_dim // 2, feat_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim // 2, feat_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.bilinear = bilinear

    def forward(self, x1, x2):
        # Perform the IFA upsampling
        x1 = self.ifa(x1, x2)
        
        # Choose the interpolation method
        if self.bilinear:
            x1 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)
        else:
            x1 = F.interpolate(x1, scale_factor=2, mode='nearest')
        
        # Calculate padding to match dimensions
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate and process with convolutional layers
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
from torch.nn import Module, Conv2d, Parameter, Softmax
class Spatio_ChannelConfluentModule(nn.Module):
    def __init__(self, in_channels, groups=2): 
        super().__init__()
        self.groups = groups
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.cweight = Parameter(torch.zeros(1, in_channels, 1, 1))
        self.cbias = Parameter(torch.ones(1, in_channels, 1, 1))
        
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels // 2, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=self.groups)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        
    @staticmethod
    def channel_shuffle(x, groups):
        batch_size, num_channels, height, width = x.shape
        channels_per_group = num_channels // groups
        x = x.view(batch_size, groups, channels_per_group, height, width)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(batch_size, -1, height, width)
        return x
    
    def forward(self, x):
        x = self.channel_shuffle(x, self.groups)
        
        max_out_channel = self.max_pool(x)
        avg_out_channel = self.avg_pool(x)
        channel_attention = self.cweight * (max_out_channel + avg_out_channel) + self.cbias
        channel_attention = self.sigmoid(channel_attention)
        x = x * channel_attention
        
        spatial_attention = self.conv1(x)
        spatial_attention = self.relu(spatial_attention)
        spatial_attention = self.conv2(spatial_attention)
        spatial_attention = self.sigmoid(spatial_attention)
        x = x * spatial_attention
        
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        
        return x
class hv_mamba(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(hv_mamba, self).__init__()
    
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = DoubleConv(32, 64)
        self.inc1 = DoubleConv(3, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(2048, 1024 // factor, bilinear)      
        self.up2 = Up(1024, 512 // factor, bilinear)
        self.up3 = Up(512, 256 // factor, bilinear)
        self.up4 = Up(256, 128, bilinear)
        self.outc = OutConv(128, n_classes)
        self.fft_out = HarmonicStateSpaceModule(channel=3)
        self.combined_attention = self_SelectiveKernelUnit(
            channels=1024,
            num_heads=8,
            kernels=[1, 3, 5, 7],
            reduction=16,
            group=1,
            L=32,
            dynamic_ratio=4
        )
        self.freq_attn = HybridCoord_FreqAttention(64)
        self.freq_attn128 = HybridCoord_FreqAttention(128)
        self.freq_attn256 = HybridCoord_FreqAttention(256)
        self.freq_attn512 = HybridCoord_FreqAttention(512)

        self.freq_attn1024 = HybridCoord_FreqAttention(1024)
        self.feature_extractor = FeatureExtractor(
            input_channels=32,
            n_stages=5,
            features_per_stage=[64, 128, 256, 512,1024],
            conv_op=nn.Conv2d,
            kernel_sizes=[3, 3, 3, 3,3],
            strides=[1, 2, 2, 2,2],
            n_blocks_per_stage=[2, 2, 2, 2,2],
            norm_op=nn.BatchNorm2d,
            norm_op_kwargs={'eps': 1e-5, 'affine': True},
            nonlin=nn.ReLU,
            nonlin_kwargs={'inplace': True}
        )
        self.decoder = DilatedAttentionDecoder(
        channel_list=[64, 128, 256, 512, 1024], 
        attention_coefficients=[32, 32,64, 128, 256]
                                )
        self.softmax = nn.Softmax(dim=1)
        self.finall_conv= nn.Conv2d(32, n_classes, 1)
        self.upsampling= nn.UpsamplingBilinear2d(scale_factor=2)
        self.Spatio_ChannelConfluentModule1=Spatio_ChannelConfluentModule(2048)
        self.Spatio_ChannelConfluentModule2=Spatio_ChannelConfluentModule(1024)
        self.Spatio_ChannelConfluentModule3=Spatio_ChannelConfluentModule(512)
        self.Spatio_ChannelConfluentModule4=Spatio_ChannelConfluentModule(256)
        self.Spatio_ChannelConfluentModule5=Spatio_ChannelConfluentModule(128)        
        # self.asem = ASEM(axial_dim=2, in_channels=n_classes, head=1)

    def forward(self, x):
        y1 = self.inc1(x)
        y2 = self.down1(y1)
        y3 = self.down2(y2)
        y4 = self.down3(y3)
        y5 = self.down4(y4)
        xfft = self.fft_out(x)
        features = self.feature_extractor(xfft)
        x1=features[0]
        x_attn1 = self.freq_attn(x1)
        e1=torch.cat([y1,x_attn1],dim=1)
        e1=self.Spatio_ChannelConfluentModule5(e1)
        x2=features[1]
        x_attn2 = self.freq_attn128(x2)
        e2=torch.cat([y2,x_attn2],dim=1)
        e2=self.Spatio_ChannelConfluentModule4(e2)
        x3=features[2]
        x_attn3 = self.freq_attn256(x3)
        e3=torch.cat([y3,x_attn3],dim=1)
        e3=self.Spatio_ChannelConfluentModule3(e3)
        x4=features[3]
        x_attn4 = self.freq_attn512(x4)
        e4=torch.cat([y4,x_attn4],dim=1)
        e4=self.Spatio_ChannelConfluentModule2(e4)
        x5=features[4]
        x5 = self.combined_attention(x5)
        e5=torch.cat([y5,x5],dim=1)
        e5=self.Spatio_ChannelConfluentModule1(e5)
        y1 = self.up1(e5, e4)
        y2 = self.up2(y1, e3)
        y3 = self.up3(y2, e2)
        y4 = self.up4(y3, e1)
        logits = self.outc(y4)
        return logits