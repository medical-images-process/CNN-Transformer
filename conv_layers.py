import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
import pdb

__all__ = [
    'ConvNormAct',
    'SingleConv',
    'BasicBlock',
    'Bottleneck',
    'DepthwiseSeparableConv',
    'SEBlock',
    'DropPath',
    'MBConv',
    'FusedMBConv',
    'ConvNeXtBlock',
    'LayerNorm'
]


class cross_att(nn.Module):
    def __init__(self, d_model, head=1, dim_mlp=512, dropout=0.0):
        super().__init__()

        self.cross_attn = nn.MultiheadAttention(d_model,
                                                head,
                                                dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_mlp)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_mlp, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.LeakyReLU()

    def forward(self, larger_feat, smaller_feat):
        """
        larger_feat shape: B, C, H, W
        smaller_feat shape: B, 1, C
        """

        B, C, h, w = larger_feat.shape
        tgt = larger_feat.view(B, C, h * w).permute(2, 0, 1)  # shape: L, B, C

        src = smaller_feat.permute(1, 0, 2)  # 1, B, C

        fusion_feature = self.cross_attn(query=tgt, key=src, value=src)[0]
        tgt = tgt + self.dropout1(fusion_feature)
        tgt = self.norm1(tgt)

        tgt1 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt1)
        tgt = self.norm2(tgt)
        return tgt.permute(1, 2, 0).view(B, C, h, w)


class ConvNormAct(nn.Module):
    """
    Layer grouping a convolution, normalization and activation funtion
    normalization includes BN and IN
    """

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=0,
                 groups=1, dilation=1, bias=False, norm=nn.BatchNorm2d, act=nn.GELU, preact=False):

        super().__init__()
        assert norm in [nn.BatchNorm2d, nn.InstanceNorm2d, True, False]
        assert act in [nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, True, False]

        self.conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            dilation=dilation,
            bias=bias
        )
        if preact:
            self.norm = norm(in_ch) if norm else nn.Identity()
        else:
            self.norm = norm(out_ch) if norm else nn.Identity()
        self.act = act() if act else nn.Identity()
        self.preact = preact

    def forward(self, x):
        if self.preact:
            out = self.conv(self.act(self.norm(x)))
        else:
            out = self.act(self.norm(self.conv(x)))

        return out


class SingleConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, norm=nn.BatchNorm2d, act=nn.GELU, preact=False):
        super().__init__()
        assert norm in [nn.BatchNorm2d, nn.InstanceNorm2d, True, False]
        assert act in [nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, True, False]

        self.conv = ConvNormAct(in_ch, out_ch, 3, stride=stride, padding=1,
                                norm=norm, act=act, preact=preact)

    def forward(self, x):
        return self.conv(x)


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, norm=nn.BatchNorm2d, act=nn.ReLU, preact=True):
        super().__init__()
        assert norm in [nn.BatchNorm2d, nn.InstanceNorm2d, True, False]
        assert act in [nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, True, False]

        self.conv1 = ConvNormAct(in_ch, out_ch, 3, stride=stride, padding=1,
                                 norm=norm, act=act, preact=preact)
        self.conv2 = ConvNormAct(out_ch, out_ch, 3, stride=1, padding=1,
                                 norm=norm, act=act, preact=preact)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = ConvNormAct(in_ch, out_ch, 3, stride=stride, padding=1,
                                        norm=norm, act=act, preact=preact)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)

        out += self.shortcut(residual)

        return out


class Bottleneck(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, groups=1, dilation=1, norm=nn.BatchNorm2d, act=nn.ReLU, preact=True):
        super().__init__()
        assert norm in [nn.BatchNorm2d, nn.InstanceNorm2d, True, False]
        assert act in [nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, True, False]
        self.expansion = 4
        self.conv1 = ConvNormAct(in_ch, out_ch // self.expansion, 1, stride=1, padding=0, norm=norm, act=act,
                                 preact=preact)
        self.conv2 = ConvNormAct(out_ch // self.expansion, out_ch // self.expansion, 3, stride=stride, padding=1,
                                 norm=norm, act=act, groups=groups, dilation=dilation, preact=preact)

        self.conv3 = ConvNormAct(out_ch // self.expansion, out_ch, 1, stride=1, padding=0, norm=norm, act=act,
                                 preact=preact)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = ConvNormAct(in_ch, out_ch, 3, stride=stride, padding=1, norm=norm, act=act, preact=preact)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        out += self.shortcut(residual)

        return out


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, kernel_size=3, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_ch,
            bias=bias
        )
        self.pointwise = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=bias
        )

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)

        return out


class SEBlock(nn.Module):
    def __init__(self, in_ch, ratio=8, act=nn.ReLU):
        super().__init__()

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // ratio, kernel_size=1),
            # act(),
            nn.Conv2d(in_ch // ratio, in_ch, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.squeeze(x)
        out = self.excitation(out)

        return x * out


class DropPath(nn.Module):
    """
    Drop connection with pobability p
    """

    def __init__(self, p=0):
        super().__init__()

        self.p = p

    def forward(self, x):
        if (not self.p) or (not self.training):
            return x

        batch_size = x.shape[0]
        random_tensor = torch.rand(batch_size, 1, 1, 1).to(x.device)
        binary_mask = self.p < random_tensor

        x = x.div(1 - self.p)
        x = x * binary_mask

        return x


class channel_att(nn.Module):
    def __init__(self, d_model, head=1, dim_mlp=512, dropout=0.0):
        super().__init__()

        self.cross_attn = nn.MultiheadAttention(d_model,
                                                head,
                                                dropout=dropout)

    def forward(self, larger_feat, smaller_feat):
        """
        larger_feat shape: B, C, H, W
        smaller_feat shape: B, 1, C
        """
        B, C, h, w = larger_feat.shape
        tgt = larger_feat.view(B, C, h * w).permute(2, 0, 1)  # shape: L, B, C
        src = smaller_feat.permute(1, 0, 2)  # 1, B, C

        fusion_feature = self.cross_attn(query=tgt, key=src, value=src)[0]
        tgt = tgt + fusion_feature
        return tgt.permute(1, 2, 0).view(B, C, h, w)


class SEBlockV2(nn.Module):
    def __init__(self, in_ch, ratio=8):
        super().__init__()

        self.squeeze = nn.AdaptiveAvgPool2d(1)

        self.channel_att = channel_att(d_model=in_ch, head=1, dim_mlp=in_ch * 2, dropout=0.0)
        # self.excitation = nn.Sequential(
        #     nn.Conv2d(in_ch, in_ch // ratio, kernel_size=1),
        #     nn.Conv2d(in_ch // ratio, in_ch, kernel_size=1),
        #     nn.Sigmoid()
        # )

    def forward(self, x):
        small_feat = self.squeeze(x)
        # out = self.excitation(out)
        # small_feat = self.gap(x)  # B,C,1,1
        feat_size = small_feat.size()
        small_feat = small_feat.view(feat_size[0], feat_size[1], 1)
        small_feat = small_feat.permute(0, 2, 1)
        " larger_feat shape: B, C, H, W smaller_feat shape: B, 1, C"
        channel_w = self.channel_att(larger_feat=x, smaller_feat=small_feat)
        channel_w = torch.sigmoid(channel_w)

        return x * channel_w


class MBConv(nn.Module):  # FFN in our Transformer
    def __init__(self, in_ch, out_ch, expansion=4, kernel_size=3, act=False, norm=nn.GELU, stride=1, ratio=4, p=0):
        super().__init__()

        padding = (kernel_size - 1) // 2
        expanded = expansion * in_ch

        self.expand_proj = nn.Conv2d(in_ch, expanded, kernel_size=1)

        self.depthwise = nn.Conv2d(expanded, expanded, kernel_size=kernel_size,
                                   stride=stride, padding=padding,
                                   groups=expanded)
        if expanded >= 128:
            ratio = 8
        # elif 32 < expanded < 128:
        #     ratio = 8
        else:
            ratio = 4

        # self.gap = nn.AdaptiveAvgPool2d(1)
        # self.channel_att = cross_att(d_model=expanded, head=1, dim_mlp=expanded * 2, dropout=0.0)
        self.se_block = SEBlock(expanded, ratio=ratio)
        # self.se_block = SEBlockV2(expanded, ratio=ratio)

        self.pointwise = nn.Conv2d(expanded, out_ch, kernel_size=1)
        self.drop_path = DropPath(p)

        self.shortcut = nn.Sequential()
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size))

    def forward(self, x):
        residual = x

        x = self.expand_proj(x)
        x = self.depthwise(x)

        x = self.se_block(x)

        # small_feat = self.gap(x)  # B,C,1,1
        # feat_size = small_feat.size()
        # small_feat = small_feat.view(feat_size[0], feat_size[1], 1)
        # small_feat = small_feat.permute(0, 2, 1)
        " larger_feat shape: B, C, H, W smaller_feat shape: B, 1, C"
        # x = self.channel_att(larger_feat=x, smaller_feat=small_feat)

        x = self.pointwise(x)
        x = self.drop_path(x)
        x = x + self.shortcut(residual)

        return x


class FusedMBConv(nn.Module):
    """
    MBConv with an expansion factor of N, and squeeze-and-excitation module
    """

    def __init__(self, in_ch, out_ch, expansion=4, kernel_size=3, stride=1, ratio=4, p=0, se=True, norm=nn.BatchNorm2d,
                 act=nn.ReLU):
        super().__init__()

        padding = (kernel_size - 1) // 2
        expanded = expansion * in_ch

        self.stride = stride
        self.se = se

        self.conv3x3 = ConvNormAct(in_ch, expanded, kernel_size=kernel_size, stride=stride, padding=padding, groups=1,
                                   norm=norm, act=act, preact=True)

        if self.se:
            self.se_block = SEBlock(expanded, ratio=ratio)

        self.pointwise = ConvNormAct(expanded, out_ch, kernel_size=1, padding=0, norm=norm, act=False, preact=True)

        self.drop_path = DropPath(p)

        self.shortcut = nn.Sequential()
        if in_ch != out_ch or stride != 1:
            self.shortcut = nn.Sequential(
                ConvNormAct(in_ch, out_ch, 3, stride=stride, padding=1, norm=False, act=False))

    def forward(self, x):
        residual = x

        x = self.conv3x3(x)
        if self.se:
            x = self.se_block(x)
        x = self.pointwise(x)

        x = self.drop_path(x)

        x = x + self.shortcut(residual)

        return x


class ConvNeXtBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, kernel_size=7, drop_path=0.1, layer_scale_init_value=1e-6):
        super().__init__()
        self.padding = math.ceil((kernel_size - 1) / 2)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=self.padding, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)

        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.

    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 

    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 

    with shape (batch_size, channels, height, width).

    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format

        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]

            return x


class ConvNeXtUpBlock(nn.Module):
    """
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, out_ch, stride=1, kernel_size=7, norm=None, act=None, preact=None, drop_path=0.,
                 layer_scale_init_value=1e-6):
        super().__init__()
        padding = kernel_size // 2
        self.upconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)

        return x


if __name__ == '__main__':
    img = torch.randn(2, 3, 256, 256)
    depth_conv = DepthwiseSeparableConv(3, 32)

    out = depth_conv(img)
    print(out.shape)
