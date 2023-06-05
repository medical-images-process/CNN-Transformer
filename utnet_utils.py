import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.ch5revised.v11.conv_layers import DepthwiseSeparableConv, BasicBlock, MBConv, FusedMBConv, ConvNormAct
from einops import rearrange
from .pyramidModule import PyLargeConv2d


class Attention(nn.Module):
    def __init__(self, feat_dim, out_dim, dim_head=64, attn_drop=0.,
                 proj_drop=0., proj_type='depthwise', pooling_ratio=None):
        super().__init__()

        if pooling_ratio is None:
            pooling_ratio = 2  # [1, 3, 5]

        self.feat_dim = feat_dim
        self.heads = feat_dim // dim_head
        self.scale = dim_head ** (-0.5)
        self.dim_head = dim_head
        self.pooling_ratio = pooling_ratio

        assert proj_type in ['linear', 'depthwise']

        if proj_type == 'linear':  # 一般来说，feat_dim==self.inner_dim
            self.feat_q = nn.Conv2d(feat_dim, self.feat_dim, kernel_size=1, bias=False)
            self.feat_out = nn.Conv2d(self.feat_dim, out_dim, kernel_size=1, bias=False)
            self.feat_kv = nn.Conv2d(feat_dim, 2 * self.feat_dim, kernel_size=3, stride=1, padding=1, groups=feat_dim)
        else:
            self.feat_q = DepthwiseSeparableConv(feat_dim, self.feat_dim)
            self.feat_out = DepthwiseSeparableConv(self.feat_dim, out_dim)
            self.feat_kv = DepthwiseSeparableConv(feat_dim, 2 * self.feat_dim)

        # self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, feat):
        B, C, H, W = feat.shape

        feat_q = self.feat_q(feat)  # B, feat_dim, H, W
        feat_kv = self.feat_kv(feat)
        # B, 2*feat_dim, rs, rs
        kv = F.adaptive_avg_pool2d(feat_kv, (round(H / self.pooling_ratio), round(W / self.pooling_ratio)))

        feat_q = rearrange(feat_q,
                           'b (dim_head heads) h w -> b heads (h w) dim_head',
                           dim_head=self.dim_head, heads=self.heads, h=H, w=W)

        k, v = kv.chunk(2, dim=1)
        b, c, h, w = k.shape
        k, v = map(  # b c h, w--->b heads N c
            lambda t: rearrange(t,
                                'b (dim_head heads) h w -> b heads (h w) dim_head',
                                dim_head=self.dim_head,
                                heads=self.heads, h=h, w=w), [k, v])

        attn = torch.einsum('bhid,bhjd->bhij', feat_q, k)
        attn *= self.scale

        feat_map_attn = F.softmax(attn, dim=-1)
        # add dropout migth cause unstable during training
        # map_feat_attn = self.attn_drop(F.softmax(attn, dim=-2))

        feat_out = torch.einsum('bhij,bhjd->bhid', feat_map_attn, v)
        feat_out = rearrange(feat_out, 'b heads (h w) dim_head -> b (dim_head heads) h w', h=H, w=W,
                             dim_head=self.dim_head, heads=self.heads)

        feat_out = self.proj_drop(self.feat_out(feat_out))
        return feat_out


class AttentionBlock(nn.Module):
    def __init__(self, feat_dim, out_dim,
                 norm=nn.BatchNorm2d,
                 act=nn.GELU, expansion=4,
                 proj_drop=0.,
                 proj_type='depthwise',
                 out_channels=[],
                 pyconv_kernels=[3, 7, 21],
                 pyconv_groups=[]
                 ):
        super().__init__()

        assert norm in [nn.BatchNorm2d, nn.InstanceNorm2d, True, False]
        assert act in [nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, True, False]
        assert proj_type in ['linear', 'depthwise']

        self.norm1 = norm(feat_dim) if norm else nn.Identity()  # norm layer for feature map

        # self.attn = Attention(feat_dim, out_dim, dim_head=dim_head,
        #                       attn_drop=attn_drop, proj_drop=proj_drop,
        #                       proj_type=proj_type, pooling_ratio=pooling_ratio)

        # self.alpha = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        # self.conv = BasicBlock(in_ch=feat_dim, out_ch=feat_dim, stride=1)
        # self.conv = SpatialAttention(feat_dim)
        # self.conv = MLKA(feat_dim)
        self.conv = PyLargeConv2d(in_channels=feat_dim,
                                  out_channels=out_channels,
                                  pyconv_kernels=pyconv_kernels,#[3, 7, 21],
                                  pyconv_groups=pyconv_groups)

        self.shortcut = nn.Sequential()
        if feat_dim != out_dim:
            self.shortcut = ConvNormAct(feat_dim, out_dim, kernel_size=1, padding=0, norm=norm, act=act, preact=True)

        if proj_type == 'linear':
            self.feedforward = FusedMBConv(out_dim, out_dim, expansion=expansion, kernel_size=1, act=act,
                                           norm=norm)  # 2 conv1x1
        else:
            self.feedforward = MBConv(out_dim, out_dim, expansion=expansion, kernel_size=3, act=act, norm=norm,
                                      p=proj_drop)  # depthwise conv

    def forward(self, x):
        feat = self.norm1(x)

        # out = self.attn(feat)
        # out_conv = self.conv(feat)
        # out = torch.sigmoid(self.alpha) * out + (1 - torch.sigmoid(self.alpha)) * out_conv
        # print('----------')
        # print(torch.sigmoid(self.alpha))
        # out += self.shortcut(x)
        # out = self.feedforward(out)

        out = self.conv(feat)
        out += self.shortcut(x)
        out = self.feedforward(out)
        return out


class PatchMerging(nn.Module):
    """
    Modified patch merging layer that works as down-sampling
    """

    def __init__(self, dim, out_dim, norm=nn.BatchNorm2d, proj_type='depthwise'):
        super().__init__()
        self.dim = dim
        if proj_type == 'linear':
            self.reduction = nn.Conv2d(4 * dim, out_dim, kernel_size=1, bias=False)
        else:
            self.reduction = DepthwiseSeparableConv(4 * dim, out_dim)

        self.norm = norm(4 * dim)

    def forward(self, x):
        """
        x: B, C, H, W
        """
        x0 = x[:, :, 0::2, 0::2]
        x1 = x[:, :, 1::2, 0::2]
        x2 = x[:, :, 0::2, 1::2]
        x3 = x[:, :, 1::2, 1::2]

        x = torch.cat([x0, x1, x2, x3], 1)  # B, 4C, H, W

        x = self.norm(x)
        x = self.reduction(x)

        return x


class BasicLayer(nn.Module):
    """
    A basic transformer layer for one stage
    No downsample and upsample operation in this layer, they are wraped in the down_block or up_block
    """

    def __init__(self, feat_dim, out_dim, num_blocks, expansion=1,
                 proj_drop=0., proj_type='depthwise', act=nn.GELU,
                 out_channels=[],
                 pyconv_kernels=[],
                 pyconv_groups=[]
                 ):
        super().__init__()

        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(AttentionBlock(feat_dim, out_dim,
                                              expansion=expansion,
                                              proj_drop=proj_drop,
                                              proj_type=proj_type,
                                              act=act,
                                              out_channels=out_channels,
                                              pyconv_kernels=pyconv_kernels,
                                              pyconv_groups=pyconv_groups
                                              ))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


#######################################################################
# UTNet block that for one stage, which contains conv block and trans block


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, block=BasicBlock, norm=nn.BatchNorm2d, act=nn.GELU):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.conv2 = block(out_ch, out_ch, norm=norm, act=act)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        return out


class down_block(nn.Module):
    def __init__(self, in_ch, out_ch, conv_num, trans_num, conv_block=BasicBlock,
                 expansion=4, proj_drop=0.,
                 proj_type='depthwise', norm=nn.BatchNorm2d, act=nn.GELU,
                 out_channels=[],
                 pyconv_kernels=[],
                 pyconv_groups=[]
                 ):
        super().__init__()
        self.patch_merging = PatchMerging(in_ch, out_ch, proj_type=proj_type, norm=norm)

        block_list = []
        for i in range(conv_num):
            block_list.append(conv_block(out_ch, out_ch, norm=norm, act=act))

        if block_list is not None:
            self.conv_blocks = nn.Sequential(*block_list)
        else:
            self.conv_blocks = None

        if trans_num > 0:
            self.trans_blocks = BasicLayer(out_ch, out_ch, num_blocks=trans_num,
                                           act=act,
                                           expansion=expansion,
                                           proj_drop=proj_drop,
                                           proj_type=proj_type,
                                           out_channels=out_channels,
                                           pyconv_kernels=pyconv_kernels,
                                           pyconv_groups=pyconv_groups
                                           )
        else:
            self.trans_blocks = None

    def forward(self, x):
        x = self.patch_merging(x)

        if self.conv_blocks is not None:
            out = self.conv_blocks(x)

        if self.trans_blocks is not None:
            out = self.trans_blocks(out)
        return out


class up_block(nn.Module):
    def __init__(self, in_ch, out_ch, conv_num, trans_num, conv_block=BasicBlock,
                 proj_type='linear', norm=nn.BatchNorm2d, act=nn.GELU,
                 out_channels=[],
                 pyconv_kernels=[],
                 pyconv_groups=[]
                 ):
        super().__init__()

        self.reduction = nn.Conv2d(in_ch + out_ch, out_ch, kernel_size=1, padding=0, bias=False)
        self.norm = norm(in_ch + out_ch)

        if trans_num > 0:
            self.trans_blocks = BasicLayer(out_ch, out_ch, num_blocks=trans_num,
                                           act=act, expansion=4,
                                           proj_drop=0.0,
                                           proj_type=proj_type,
                                           out_channels=out_channels,
                                           pyconv_kernels=pyconv_kernels,
                                           pyconv_groups=pyconv_groups
                                           )

        # NTB(out_ch, out_ch, path_dropout=0.01, stride=2,
        #                             sr_ratio=2, head_dim=dim_head, mix_block_ratio=1,
        #                             attn_drop=attn_drop, drop=0.0)
        else:
            self.trans_blocks = None

        conv_list = []
        for i in range(conv_num):
            conv_list.append(conv_block(out_ch, out_ch, norm=norm, act=act))

        self.conv_blocks = nn.Sequential(*conv_list)

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, size=x2.shape[-2:], mode='bilinear', align_corners=True)
        feat = torch.cat([x1, x2], dim=1)

        out = self.reduction(self.norm(feat))
        if self.trans_blocks is not None:
            out = self.trans_blocks(out)
        out = self.conv_blocks(out)

        return out


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


# class skip_connection(nn.Module):
#     def __init__(self, in_ch, out_ch, block=BasicBlock, norm=nn.BatchNorm2d, act=nn.GELU):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
#         self.conv2 = block(out_ch, out_ch, norm=norm, act=act)
#
#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.conv2(out)
#
#         return out