import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


IMG_SCALE = 352


class UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=2,
                 enc_nf=[8, 16, 16, 32], dec_nf=[32, 32, 16, 16, 8, 8],
                 full_size=True, for_seg=False,
                 use_bn=False, group_num=1, use_separable=False):
        """
        Arguments:
        enc_nf: [int], number of features in each encoder layer.
        dec_nf: [int], number of features in each encoder layer.
        """
        super(UNet, self).__init__()
        self.use_bn = use_bn
        self.use_separable = use_separable
        self.enc_nf = enc_nf
        self.dec_nf = dec_nf
        self.group_num = group_num
        self.for_seg = for_seg

        # Encoder functions
        self.enc = nn.ModuleList()
        for i in range(len(enc_nf)):
            prev_nf = in_channels if i == 0 else enc_nf[i - 1]
            self.enc.append(self.conv_block(
                in_channels=prev_nf, out_channels=enc_nf[i],
                kernel_size=3, stride=2
            ))

        # Decoder functions
        self.dec = nn.ModuleList()
        self.dec.append(self.conv_block(
            in_channels=enc_nf[3], out_channels=dec_nf[0]))  # 1
        self.dec.append(self.conv_block(
            in_channels=dec_nf[0] + enc_nf[2], out_channels=dec_nf[1]))  # 2
        self.dec.append(self.conv_block(
            in_channels=dec_nf[1] + enc_nf[1], out_channels=dec_nf[2]))  # 3
        self.dec.append(self.conv_block(
            in_channels=dec_nf[2] + enc_nf[0], out_channels=dec_nf[3]))  # 4
        self.dec.append(self.conv_block(
            in_channels=dec_nf[3], out_channels=dec_nf[4]))  # 5
        self.dec.append(self.conv_block(
            in_channels=dec_nf[4] + in_channels, out_channels=dec_nf[5]))  # 6

        # upsample layer
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # last_conv layer to get the flow field or segment map.
        if self.use_separable:
            conv_fn = SeparableConv3d
        else:
            conv_fn = nn.Conv3d
        if for_seg:
            # segmenter network
            self.last_conv = conv_fn(
                in_channels=dec_nf[-1], out_channels=out_channels,
                kernel_size=3, padding=1)
        else:
            # regression network
            flow = conv_fn(
                in_channels=dec_nf[-1],
                out_channels=3,  # dimension=3
                kernel_size=3, padding=1)

            # Make flow weights + bias small. Not sure this is necessary.
            nd = Normal(0, 1e-5)
            flow.weight = nn.Parameter(nd.sample(flow.weight.shape))
            flow.bias = nn.Parameter(torch.zeros(flow.bias.shape))

            if self.use_bn:
                last_bn = nn.BatchNorm3d(3)
                self.last_conv = nn.Sequential(flow, last_bn)
            else:
                self.last_conv = flow

    def conv_block(self, in_channels, out_channels,
                   kernel_size=3, stride=1, padding=1):

        if self.use_separable:
            conv_fn = SeparableConv3d
        else:
            conv_fn = nn.Conv3d

        if in_channels % self.group_num == 0:
            groups = self.group_num
        else:
            groups = 1

        if self.use_bn:
            bn_fn = nn.BatchNorm3d
            layer = nn.Sequential(
                conv_fn(in_channels, out_channels, kernel_size,
                        stride=stride, padding=padding, groups=groups),
                bn_fn(out_channels),
                nn.LeakyReLU(0.2)
            )
        else:
            layer = nn.Sequential(
                conv_fn(in_channels, out_channels, kernel_size,
                        stride=stride, padding=padding, groups=groups),
                nn.LeakyReLU(0.2)
            )
        return layer

    def forward(self, x):

        if isinstance(x, list):
            x = torch.stack(x, 1)

        # Encoder
        x_enc = [x]
        for i, l in enumerate(self.enc):
            x = l(x_enc[-1])
            x_enc.append(x)

        # Decoder
        y = x_enc[-1]
        # 3 conv + upsample + concatenate series
        for i in range(3):
            y = self.dec[i](y)
            y = self.upsample(y)
            y = torch.cat([y, x_enc[-(i + 2)]], dim=1)
        # 2 convs at full_size/2 res
        y = self.dec[3](y)
        y = self.dec[4](y)
        # Upsample to full res, concatenate and conv
        y = self.upsample(y)
        y = torch.cat([y, x_enc[0]], dim=1)
        y = self.dec[5](y)

        # Last conv layer
        y = self.last_conv(y)
        return y


class SeparableConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(SeparableConv3d, self).__init__()

        self.depthwise = nn.Conv3d(
            in_channels, out_channels, kernel_size,
            stride, padding, dilation, groups=groups, bias=bias
        )
        self.pointwise = nn.Conv3d(
            in_channels, out_channels, kernel_size=1,
            stride=1, padding=0, dilation=1, groups=1, bias=bias
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class SpatialTransformation(nn.Module):
    def __init__(self, size=[64, IMG_SCALE, IMG_SCALE], mode='bilinear'):
        super(SpatialTransformation, self).__init__()
        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

        self.mode = mode

    def forward(self, src, flow):
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * \
                (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, mode=self.mode)


class ShapeMorph3d(nn.Module):
    def __init__(self, enc_nf, dec_nf,
                 in_channels=4, num_mods=2,
                 use_bn=False, group_num=1, use_separable=False):
        super(ShapeMorph3d, self).__init__()
        ###################################################
        # Regression UNet
        self.unet = UNet(
            in_channels=in_channels,
            for_seg=False,
            use_bn=use_bn,
            group_num=group_num,
            use_separable=use_separable,
            enc_nf=enc_nf,
            dec_nf=dec_nf,
        )
        ###################################################
        self.spatial_transform = SpatialTransformation()
        self.num_mods = num_mods

    def forward(self, moving_image, moving_seg, fixed_image, fixed_seg):
        if isinstance(moving_image, list):
            moving_image = torch.cat(moving_image, 1)
        # if isinstance(moving_seg,list):
        #    moving_seg=torch.stack(moving_seg,1)
        R_I = [fixed_image]
        R_S = [fixed_seg]
        DM = []
        for i in range(self.num_mods-1):
            x = torch.cat([moving_image[:, i:i+1, ...], moving_seg[:, i:i+1, ...],
                           fixed_image, fixed_seg], dim=1)
            deformation_matrix = self.unet(x)
            registered_image = self.spatial_transform(
                moving_image[:, i:i+1, ...], deformation_matrix)
            R_I.append(registered_image)
            registered_segs = self.spatial_transform(
                moving_seg[:, i:i+1, ...], deformation_matrix)
            R_S.append(registered_segs)
            DM.append(deformation_matrix)
        R_I = torch.cat(R_I, 1)
        R_S = torch.cat(R_S, 1)
        DM = torch.cat(DM, 1)
        return R_I, R_S, DM
