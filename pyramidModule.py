# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile, clever_format

from lib.ch5revised.v11.conv_layers import SEBlockV2


# class ChannelShuffle(nn.Module):
#     def __init__(self, groups):
#         super(ChannelShuffle, self).__init__()
#         self.groups = groups
#
#     def forward(self, x):
#         batch_size, num_channels, height, width = x.size()
#         channels_per_group = num_channels // self.groups
#
#         # Reshape input tensor
#         x = x.view(batch_size, self.groups, channels_per_group, height, width)
#
#         # Perform channel shuffle
#         x = torch.transpose(x, 1, 2).contiguous()
#
#         # Reshape output tensor
#         x = x.view(batch_size, -1, height, width)
#         return x


class MLP(nn.Module):
    def __init__(self, n_feats):
        super().__init__()

        self.norm = LayerNorm(n_feats, data_format='channels_first')
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

        if n_feats >= 64:
            i_feats = n_feats // 8
        else:
            i_feats = n_feats // 4

        self.fc1 = nn.Conv2d(n_feats, i_feats, 1, 1, 0)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(i_feats, n_feats, 1, 1, 0)

    def forward(self, x):
        shortcut = x.clone()
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)

        return x * self.scale + shortcut


# multi-scale convolution attention with large kernel convolution
class MLKA_4k_256(nn.Module):  # 256=128+128
    def __init__(self, dim_feats):
        super().__init__()

        self.norm = LayerNorm(dim_feats, data_format='channels_first')
        # self.scale = nn.Parameter(torch.zeros((1, dim_feats, 1, 1)), requires_grad=True)

        self.LKA = nn.ModuleList()
        self.LKA.append(
            nn.Sequential(
                nn.Conv2d(128, 128,
                          3, 1, 1, groups=8),
                nn.Conv2d(128, 128, 1, 1, 0)
            )
        )
        self.LKA.append(
            nn.Sequential(
                nn.Conv2d(128, 128,
                          5, 1, 5 // 2, groups=64),
                nn.Conv2d(128, 128,
                          7, stride=1, padding=(7 // 2) * 3,
                          groups=64,
                          dilation=3),
                nn.Conv2d(128, 128, 1, 1, 0)
            )
        )
        # self.channel_shuffle = ChannelShuffle(dim_feats)
        # self.se_block = SEBlockV2(in_ch=dim_feats, ratio=16)

        # self.proj_first = nn.Sequential(
        #     nn.Conv2d(dim_feats, dim_feats, 1, 1, 0))

        self.proj_last = nn.Sequential(
            nn.Conv2d(dim_feats, dim_feats, 1, 1, 0))

        # channel attention sub-block
        # self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        # self.MLP = nn.Sequential(nn.Conv2d(dim_feats, dim_feats // 8, kernel_size=1),
        #                          nn.Conv2d(dim_feats // 8, dim_feats, kernel_size=1),
        #                          nn.Sigmoid()
        #                          )

    def forward(self, x):
        shortcut0 = x.clone()
        # x = self.channel_shuffle(x)
        x = self.norm(x)
        # x = self.proj_first(x)
        #
        # pool = self.pooling(x)
        # channel_weight = self.MLP(pool)

        shortcut1 = x.clone()

        # 256=32+32+64+128
        x_1, x_2 = torch.chunk(x, 2, dim=1)
        # x_1, x_2, x_3, x_4 = x[:, 0:32, :, :], \
        #                      x[:, 32:(32 + 32), :, :], \
        #                      x[:, (32 + 32):(32 + 32 + 64), :, :], \
        #                      x[:, (32 + 32 + 64):, :, :]

        x = torch.cat([self.LKA[0](x_1) * x_1,
                       self.LKA[1](x_2) * x_2
                       ],
                      dim=1)

        # x = self.proj_last(x * shortcut1) * channel_weight + shortcut0
        x = self.proj_last(x * shortcut1) + shortcut0
        # x = self.se_block(x)
        return x


class MLKA_3k_128(nn.Module):
    def __init__(self, dim_feats):  # 128=64+64
        super().__init__()
        self.norm = LayerNorm(dim_feats, data_format='channels_first')

        self.LKA = nn.ModuleList()
        self.LKA.append(
            nn.Sequential(
                nn.Conv2d(64, 64,
                          3, 1, 3 // 2, groups=4),
                nn.Conv2d(64, 64, 1, 1, 0)
            )
        )
        self.LKA.append(
            nn.Sequential(
                nn.Conv2d(64, 64,
                          5, 1, 5 // 2, groups=16),
                nn.Conv2d(64, 64,
                          7, stride=1, padding=(7 // 2) * 3, groups=16,
                          dilation=3),
                nn.Conv2d(64, 64, 1, 1, 0)
            )
        )
        # self.channel_shuffle = ChannelShuffle(dim_feats)
        # self.se_block = SEBlockV2(in_ch=dim_feats, ratio=8)

        # self.LKA.append(
        #     nn.Sequential(
        #         nn.Conv2d(64, 64,
        #                   7, 1, 7 // 2, groups=64),
        #         nn.Conv2d(64, 64,
        #                   9, stride=1, padding=(9 // 2) * 4, groups=64,
        #                   dilation=4),
        #         nn.Conv2d(64, 64, 1, 1, 0)
        #     )
        # )

        self.proj_first = nn.Sequential(
            nn.Conv2d(dim_feats, dim_feats, 1, 1, 0))

        self.proj_last = nn.Sequential(
            nn.Conv2d(dim_feats, dim_feats, 1, 1, 0))

        # channel attention sub-block
        # self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        # self.MLP = nn.Sequential(nn.Conv2d(dim_feats, dim_feats // 8, kernel_size=1),
        #                          nn.Conv2d(dim_feats // 8, dim_feats, kernel_size=1),
        #                          nn.Sigmoid()
        #                          )

    def forward(self, x):
        shortcut0 = x.clone()
        # x = self.channel_shuffle(x)
        x = self.norm(x)
        x = self.proj_first(x)

        # pool = self.pooling(x)
        # channel_weight = self.MLP(pool)

        shortcut1 = x.clone()

        # x_1, x_2 = torch.chunk(x, 2, dim=1)
        x_1, x_2 = x[:, 0:64, :, :], \
                   x[:, 64:, :, :]
        # x[:, (32 + 32):, :, :]

        x = torch.cat([self.LKA[0](x_1) * x_1, self.LKA[1](x_2) * x_2], dim=1)
        # x = self.proj_last(x * shortcut1) * channel_weight + shortcut0
        x = self.proj_last(x * shortcut1) + shortcut0
        # x = self.se_block(x)
        return x


class MLKA_3k_64(nn.Module):
    def __init__(self, dim_feats):  # 64=32+32
        super().__init__()
        self.norm = LayerNorm(dim_feats, data_format='channels_first')
        # self.scale = nn.Parameter(torch.zeros((1, dim_feats, 1, 1)), requires_grad=True)
        # self.channel_shuffle = ChannelShuffle(dim_feats)
        self.se_block = SEBlockV2(in_ch=dim_feats, ratio=4)

        self.LKA = nn.ModuleList()
        self.LKA.append(
            nn.Sequential(
                nn.Conv2d(32, 32,
                          3, 1, 3 // 2, groups=8),
                nn.Conv2d(32, 32, 1, 1, 0)
            )
        )
        # self.LKA.append(
        #     nn.Sequential(
        #         nn.Conv2d(16, 16,
        #                   7, 1, 7 // 2, groups=8),
        #         nn.Conv2d(16, 16, 1, 1, 0)
        #     )
        # )
        self.LKA.append(
            nn.Sequential(
                nn.Conv2d(32, 32,
                          5, 1, 5 // 2, groups=32),
                nn.Conv2d(32, 32,
                          7, stride=1, padding=(7 // 2) * 3, groups=32,
                          dilation=3),
                nn.Conv2d(32, 32, 1, 1, 0)
            )
        )

        self.proj_first = nn.Sequential(
            nn.Conv2d(dim_feats, dim_feats, 1, 1, 0))

        self.proj_last = nn.Sequential(
            nn.Conv2d(dim_feats, dim_feats, 1, 1, 0))

        # channel attention sub-block
        # self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        # self.MLP = nn.Sequential(nn.Conv2d(dim_feats, dim_feats // 8, kernel_size=1),
        #                          nn.Conv2d(dim_feats // 8, dim_feats, kernel_size=1),
        #                          nn.Sigmoid()
        #                          )

    def forward(self, x):
        shortcut0 = x.clone()
        # x = self.channel_shuffle(x)
        x = self.norm(x)
        x = self.proj_first(x)

        # pool = self.pooling(x)
        # channel_weight = self.MLP(pool)

        shortcut1 = x.clone()

        # x_1, x_2 = torch.chunk(x, 2, dim=1)
        x_1, x_2 = x[:, 0:32, :, :], x[:, 32:, :, :]
        x = torch.cat([self.LKA[0](x_1) * x_1, self.LKA[1](x_2) * x_2], dim=1)
        # x = self.proj_last(x * shortcut1) * channel_weight + shortcut0
        x = self.proj_last(x * shortcut1) + shortcut0
        # x = self.se_block(x)
        return x


class MLKA_2k_32(nn.Module):
    def __init__(self, dim_feats):  # 32=16+16
        super().__init__()

        self.norm = LayerNorm(dim_feats, data_format='channels_first')
        self.LKA = nn.ModuleList()
        self.LKA.append(
            nn.Sequential(
                nn.Conv2d(16, 16,
                          3, 1, 3 // 2, groups=4),
                nn.Conv2d(16, 16, 1, 1, 0)
            )
        )
        # self.channel_shuffle = ChannelShuffle(dim_feats)
        # self.se_block = SEBlockV2(in_ch=dim_feats, ratio=2)

        self.LKA.append(
            nn.Sequential(
                nn.Conv2d(16, 16,
                          5, 1, 5 // 2, groups=16),
                nn.Conv2d(16, 16, 1, 1, 0)
            )
        )

        self.proj_first = nn.Sequential(
            nn.Conv2d(dim_feats, dim_feats, 1, 1, 0))

        self.proj_last = nn.Sequential(
            nn.Conv2d(dim_feats, dim_feats, 1, 1, 0))

        # # channel attention sub-block
        # self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        # self.MLP = nn.Sequential(nn.Conv2d(dim_feats, dim_feats // 8, kernel_size=1),
        #                          nn.Conv2d(dim_feats // 8, dim_feats, kernel_size=1),
        #                          nn.Sigmoid()
        #                          )

    def forward(self, x):
        shortcut0 = x.clone()
        # x = self.channel_shuffle(x)

        x = self.norm(x)
        x = self.proj_first(x)

        # pool = self.pooling(x)
        # channel_weight = self.MLP(pool)

        x_1, x_2 = x[:, 0:16, :, :], x[:, 16:, :, :]
        x_c = torch.cat([self.LKA[0](x_1) * x_1, self.LKA[1](x_2) * x_2], dim=1)

        # x = self.proj_last(x_c * x) * channel_weight + shortcut0
        x = self.proj_last(x_c * x) + shortcut0
        # x = self.se_block(x)

        return x


class PyLargeConv2d(nn.Module):
    """Modified Pyramidal Convolution code from https://github.com/iduta/pyconv by Xiaowei Liu
    <<Pyramidal Convolution: Rethinking Convolutional Neural Networks for Visual Recognition>>
    https://arxiv.org/pdf/2006.11538.pdf
    Args:
        in_channels (int): Number of channels in the input image
        out_channels (list): Number of channels for each pyramid level produced by the convolution
        pyconv_kernels (list): Spatial size of the kernel for each pyramid level
        pyconv_groups (list): Number of blocked connections from input channels to output channels for each pyramid level
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``False``

    Example::

        >>> # PyConv with two pyramid levels, kernels: 3x3, 5x5
        >>> m = PyConv2d(in_channels=64, out_channels=[32, 32], pyconv_kernels=[3, 5], pyconv_groups=[1, 4])
        >>> input = torch.randn(4, 64, 56, 56)
        >>> output = m(input)

        >>> # PyConv with three pyramid levels, kernels: 3x3, 5x5, 7x7
        >>> m = PyConv2d(in_channels=64, out_channels=[16, 16, 32], pyconv_kernels=[3, 5, 7], pyconv_groups=[1, 4, 8])
        >>> input = torch.randn(4, 64, 56, 56)
        >>> output = m(input)
    """

    def __init__(self, in_channels, out_channels, pyconv_kernels, pyconv_groups, stride=1, dilation=1, bias=False):
        super(PyLargeConv2d, self).__init__()

        assert len(out_channels) == len(pyconv_kernels) == len(pyconv_groups)

        # self.pwconv1 = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.pyconv_levels = [None] * len(pyconv_kernels)
        self.pyconv_gates = [None] * len(pyconv_kernels)
        for i in range(len(pyconv_kernels)):
            # if pyconv_kernels[i] == 36:
            #     self.pyconv_levels[i] = nn.Sequential(
            #         nn.Conv2d(in_channels, out_channels[i], 7,
            #                   1, 7 // 2, groups=pyconv_groups[i]),
            #         nn.Conv2d(out_channels[i], out_channels[i], 9,
            #                   stride=1, padding=(9 // 2) * 4,
            #                   groups=pyconv_groups[i],
            #                   dilation=4),
            #         nn.Conv2d(out_channels[i], out_channels[i], 1, 1, 0))
            #     self.pyconv_gates[i] = nn.Conv2d(in_channels, out_channels[i], 7, 1, 7 // 2,
            #                                      groups=pyconv_groups[i])

            if pyconv_kernels[i] > 7:  # 21
                self.pyconv_levels[i] = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels[i], 5, 1,
                              5 // 2,
                              groups=pyconv_groups[i]),
                    nn.Conv2d(out_channels[i], out_channels[i], 7, stride=1,
                              padding=(7 // 2) * 3,
                              groups=pyconv_groups[i],
                              dilation=3),
                    nn.Conv2d(out_channels[i], out_channels[i], 1, 1, 0))
                self.pyconv_gates[i] = nn.Conv2d(in_channels, out_channels[i], 5, 1, 5 // 2,
                                                 groups=pyconv_groups[i])
            else:  # kernel < 21
                self.pyconv_levels[i] = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels[i], kernel_size=pyconv_kernels[i],
                              stride=stride, padding=pyconv_kernels[i] // 2,
                              groups=pyconv_groups[i],
                              dilation=dilation, bias=bias),
                    nn.Conv2d(out_channels[i], out_channels[i], 1, 1, 0)
                )
                self.pyconv_gates[i] = nn.Conv2d(in_channels, out_channels[i],
                                                 3, 1, 3 // 2,
                                                 groups=pyconv_groups[i])
        self.pyconv_levels = nn.ModuleList(self.pyconv_levels)
        self.pyconv_gates = nn.ModuleList(self.pyconv_gates)
        self.pwconv2 = nn.Conv2d(in_channels, in_channels, 1, 1, 0)

    def forward(self, x):
        shortcut = x.clone()
        # x = self.pwconv1(x)
        # if self.is_channel_dby_3 is True:
        #     x = self.channel_reduction(x)
        out = []
        for level, gate in zip(self.pyconv_levels, self.pyconv_gates):
            out.append(level(x) * gate(x))

        out = torch.cat(out, 1)
        # if self.is_channel_dby_3 is True:
        #     out = self.channel_restore(out)
        out = self.pwconv2(out)

        f_out = shortcut + out
        return f_out


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
