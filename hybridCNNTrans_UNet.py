"""
--model cellfusionv10 --dimension 2d --dataset isic --unique_name isic_2d_cellfusion_V9 --start_fold 0 --batch_size 16
1) 将Transformer模块中self-attention换成了大核+小核attention
2) skip connection上添加了MLKA融合模块
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.ch5revised.v11.pyramidModule import MLKA_2k_32, MLKA_4k_256, MLKA_3k_128, MLKA_3k_64
from lib.ch5revised.v11.utnet_utils import inconv, down_block, up_block, cross_att
from model.dim2.utils import get_block


class CellFusionUTNetV11(nn.Module):
    def __init__(self, in_chan, num_classes, base_chan=32, conv_block='BasicBlock',
                 conv_num=[2, 1, 0, 0, 0, 1, 2, 2],
                 trans_num=[0, 1, 2, 2, 2, 1, 0, 0],  # [0, 1, 2, 2, 2, 1, 0, 0]
                 num_heads=[1, 4, 8, 16, 8, 4, 1, 1],
                 expansion=4,
                 attn_drop=0., proj_drop=0., proj_type='depthwise',
                 norm=nn.BatchNorm2d, act=nn.GELU):
        super().__init__()

        chan_num = [2 * base_chan, 4 * base_chan, 8 * base_chan, 16 * base_chan,
                    8 * base_chan, 4 * base_chan, 2 * base_chan, base_chan]
        dim_head = [chan_num[i] // num_heads[i] for i in range(8)]

        conv_block = get_block(conv_block)

        # self.inc and self.down1 forms the conv stem
        self.inc = inconv(in_chan, base_chan, norm=norm, act=act)  # 32 224 224
        self.down1 = down_block(base_chan, chan_num[0], conv_num[0], trans_num[0],
                                conv_block, norm=norm, act=act
                                )  # 64 112 112

        # down2 down3 down4 apply the new designed transformer blocks
        self.down2 = down_block(chan_num[0], chan_num[1], conv_num[1], trans_num[1], conv_block,
                                expansion=expansion, proj_drop=proj_drop,
                                proj_type=proj_type, norm=norm, act=act,
                                # 128 56 56
                                out_channels=[64, 64],
                                pyconv_kernels=[3, 21],
                                pyconv_groups=[64, 64]
                                )
        self.down3 = down_block(chan_num[1], chan_num[2], conv_num[2], trans_num[2], conv_block,
                                expansion=expansion, proj_drop=proj_drop,
                                proj_type=proj_type, norm=norm, act=act,
                                # 256 28 28
                                out_channels=[128, 128],
                                pyconv_kernels=[3, 21],
                                pyconv_groups=[128, 128]
                                )
        self.down4 = down_block(chan_num[2], chan_num[3], conv_num[3], trans_num[3], conv_block,
                                expansion=expansion, proj_drop=proj_drop,
                                proj_type=proj_type, norm=norm, act=act,
                                # 512 14 14
                                out_channels=[256, 256],
                                pyconv_kernels=[3, 21],
                                pyconv_groups=[256, 256]
                                )

        self.up1 = up_block(chan_num[3], chan_num[4], conv_num[4], trans_num[4], conv_block,
                            proj_type=proj_type, norm=norm, act=act,
                            # 256 28 28
                            out_channels=[128, 128],
                            pyconv_kernels=[3, 21],
                            pyconv_groups=[128, 128]
                            )
        self.up2 = up_block(chan_num[4], chan_num[5], conv_num[5], trans_num[5], conv_block,
                            proj_type=proj_type, norm=norm, act=act,
                            # 128 56 56
                            out_channels=[64, 64],
                            pyconv_kernels=[3, 21],
                            pyconv_groups=[64, 64]
                            )

        # up3 up4 form the conv decoder
        self.up3 = up_block(chan_num[5], chan_num[6], conv_num[6], trans_num[6],
                            conv_block, norm=norm, act=act,
                            )  # 112 112
        self.up4 = up_block(chan_num[6], chan_num[7], conv_num[7], trans_num[7],
                            conv_block, norm=norm, act=act)  # 224 224

        # side outputs
        # self.convolution_mapping_2 = nn.Conv2d(in_channels=chan_num[6],
        #                                        out_channels=chan_num[7],
        #                                        kernel_size=(1, 1),
        #                                        stride=(1, 1),
        #                                        padding=(0, 0),
        #                                        bias=True)
        # self.convolution_mapping_4 = nn.Conv2d(in_channels=chan_num[5],
        #                                        out_channels=chan_num[7],
        #                                        kernel_size=(1, 1),
        #                                        stride=(1, 1),
        #                                        padding=(0, 0),
        #                                        bias=True)
        # self.convolution_mapping_8 = nn.Conv2d(in_channels=chan_num[4],
        #                                        out_channels=chan_num[7],
        #                                        kernel_size=(1, 1),
        #                                        stride=(1, 1),
        #                                        padding=(0, 0),
        #                                        bias=True)
        # self.cross_8_4 = cross_att(d_model=chan_num[7], head=1,
        #                            dim_mlp=4 * chan_num[7], dropout=0.0)
        # self.cross_4_2 = cross_att(d_model=chan_num[7], head=1,
        #                            dim_mlp=4 * chan_num[7], dropout=0.0)
        # self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        # self.outc = nn.Conv2d(3*chan_num[7], num_classes, kernel_size=1)

        self.mlka4 = MLKA_4k_256(dim_feats=256)  # 256=32+32+64+128
        self.mlka_d2_57 = MLKA_3k_128(dim_feats=128)  # 128=32+32+64
        self.mlka_d1_57 = MLKA_3k_64(dim_feats=64)  # 64=16+16+32
        self.mlka_d0_5 = MLKA_2k_32(dim_feats=32)  # 32=8+24
        self.outc = nn.Conv2d(chan_num[7], num_classes, kernel_size=1)

    def forward(self, x):
        x0 = self.inc(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        x3 = self.mlka4(x3)
        out = self.up1(x4, x3)

        x2 = self.mlka_d2_57(x2)
        out = self.up2(out, x2)

        x1 = self.mlka_d1_57(x1)
        out = self.up3(out, x1)

        x0 = self.mlka_d0_5(x0)
        out = self.up4(out, x0)

        out = self.outc(out)

        return out


if __name__ == '__main__':
    from thop import profile
    from thop import clever_format

    # mit = mit_new(in_chans=1, base_ch=64, n_cls=4)
    mit = CellFusionUTNetV11(in_chan=1, num_classes=4,
                             conv_num=[2, 1, 1, 0, 1, 1, 2, 2],  # [0,3,4,3, 4,3,0,0] [0, 1, 2, 2, 2, 1, 0, 0]
                             trans_num=[0, 2, 3, 3, 3, 1, 0, 0]  # [0, 2, 3, 3, 3, 1, 0, 0]02333100 [0,1,2,2,  2,1,0,0]
                             ).cuda()
    input = torch.randn(1, 1, 224, 224).cuda()
    y = mit(input)
    print(y.shape)

    flops, params = profile(mit, inputs=(input,))
    macs, params = clever_format([flops, params], "%.3f")
    print(macs, params)
