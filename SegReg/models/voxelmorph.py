import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


class UNet(nn.Module):
    def __init__(self, in_channels=4, out_channel=2, dim=3, enc_nf=[16, 32, 32, 32], dec_nf=[32, 32, 32, 32, 8, 8],
                 bn=None, full_size=True, forseg=False):
        super(UNet, self).__init__()
        self.bn = bn
        self.dim = dim
        self.enc_nf = enc_nf
        self.full_size = full_size
        self.vm2 = len(dec_nf) == 7
        # Encoder functions
        self.enc = nn.ModuleList()
        for i in range(len(enc_nf)):
            prev_nf = in_channels if i == 0 else enc_nf[i - 1]
            self.enc.append(self.conv_block(
                dim, prev_nf, enc_nf[i], 3, 2, batchnorm=bn))
        # Decoder functions
        self.dec = nn.ModuleList()
        self.dec.append(self.conv_block(
            dim, enc_nf[3], dec_nf[0], batchnorm=bn))  # 1
        self.dec.append(self.conv_block(
            dim, dec_nf[0] + enc_nf[2], dec_nf[1], batchnorm=bn))  # 2
        self.dec.append(self.conv_block(
            dim, dec_nf[1] + enc_nf[1], dec_nf[2], batchnorm=bn))  # 3
        self.dec.append(self.conv_block(
            dim, dec_nf[2] + enc_nf[0], dec_nf[3], batchnorm=bn))  # 4
        self.dec.append(self.conv_block(
            dim, dec_nf[3], dec_nf[4], batchnorm=bn))  # 5
        self.forseg = forseg
        if self.full_size:
            self.dec.append(self.conv_block(
                dim, dec_nf[4] + in_channels, dec_nf[5], batchnorm=bn))
        if self.vm2:
            self.vm2_conv = self.conv_block(
                dim, dec_nf[5], dec_nf[6], batchnorm=bn)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # One conv to get the flow field
        conv_fn = getattr(nn, 'Conv%dd' % dim)
        if forseg:
            self.seg = conv_fn(dec_nf[-1], out_channel,
                               kernel_size=3, padding=1)
        else:
            self.flow = conv_fn(dec_nf[-1], dim, kernel_size=3, padding=1)
            # Make flow weights + bias small. Not sure this is necessary.
            nd = Normal(0, 1e-5)
            self.flow.weight = nn.Parameter(nd.sample(self.flow.weight.shape))
            self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))
            self.batch_norm = getattr(nn, "BatchNorm{0}d".format(dim))(3)

    def conv_block(self, dim, in_channels, out_channels, kernel_size=3, stride=1, padding=1, batchnorm=False):
        conv_fn = getattr(nn, "Conv{0}d".format(dim))
        bn_fn = getattr(nn, "BatchNorm{0}d".format(dim))
        if batchnorm:
            layer = nn.Sequential(
                conv_fn(in_channels, out_channels, kernel_size,
                        stride=stride, padding=padding),
                bn_fn(out_channels),
                nn.LeakyReLU(0.2))
        else:
            layer = nn.Sequential(
                conv_fn(in_channels, out_channels, kernel_size,
                        stride=stride, padding=padding),
                nn.LeakyReLU(0.2))
        return layer

    def forward(self, x):
        if isinstance(x, list):
            x = torch.stack(x, 1)
        x_enc = [x]
        for i, l in enumerate(self.enc):
            x = l(x_enc[-1])
            x_enc.append(x)
        # Three conv + upsample + concatenate series
        y = x_enc[-1]
        for i in range(3):
            y = self.dec[i](y)
            y = self.upsample(y)
            y = torch.cat([y, x_enc[-(i + 2)]], dim=1)
        # Two convs at full_size/2 res
        y = self.dec[3](y)
        y = self.dec[4](y)
        # Upsample to full res, concatenate and conv
        if self.full_size:
            y = self.upsample(y)
            y = torch.cat([y, x_enc[0]], dim=1)
            y = self.dec[5](y)
        # Extra conv for vm2
        if self.vm2:
            y = self.vm2_conv(y)
        if self.forseg:
            y = self.seg(y)
            return y
        else:
            flow = self.flow(y)
            if self.bn:
                flow = self.batch_norm(flow)
            return flow


class SpatialTransformation(nn.Module):
    def __init__(self, size=[64, 128, 128], mode='bilinear'):
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


class VoxelMorph3d(nn.Module):
    def __init__(self, in_channels=4):
        super(VoxelMorph3d, self).__init__()
        self.unet = UNet(in_channels, forseg=False)
        self.spatial_transform = SpatialTransformation()

    def forward(self, moving_image, moving_seg, fixed_image):
        x = torch.cat([moving_image, fixed_image], dim=1)
        deformation_matrix = self.unet(x)
        registered_image = self.spatial_transform(
            moving_image, deformation_matrix)
        registered_seg = self.spatial_transform(moving_seg, deformation_matrix)
        return registered_image, registered_seg, deformation_matrix


class ShapeMorph3d(nn.Module):
    def __init__(self, in_channels=4, num_mods=2):
        super(ShapeMorph3d, self).__init__()
        self.unet = UNet(in_channels, forseg=False)
        self.spatial_transform = SpatialTransformation()
        self.num_mods = num_mods

    def forward(self, moving_image, moving_seg, fixed_image, fixed_seg):
        if isinstance(moving_image, list):
            moving_image = torch.cat(moving_image, 1)
        # if isinstance(moving_seg,list):
        #    moving_seg=torch.stack(moving_seg,1)
        R_I = [fixed_image.unsqueeze(1)]
        R_S = [fixed_seg]
        DM = []
        for i in range(self.num_mods-1):
            x = torch.cat([moving_image[:, i:i+1, ...], moving_seg[i],
                           fixed_image.unsqueeze(1), fixed_seg], dim=1)
            deformation_matrix = self.unet(x)
            registered_image = self.spatial_transform(
                moving_image[:, i:i+1, ...], deformation_matrix)
            R_I.append(registered_image)
            registered_segs = self.spatial_transform(
                moving_seg[i], deformation_matrix)
            R_S.append(registered_segs)
            DM.append(deformation_matrix)
        return R_I, R_S, DM
