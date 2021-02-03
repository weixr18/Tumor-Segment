import torch
import torch.nn as nn

from .voxelmorph import UNet, ShapeMorph3d


class Segmenter(nn.Module):
    def __init__(self, modality=3, num_of_cls=1):
        super(Segmenter, self).__init__()
        ####################################################################
        # Segmenter UNet
        self.nets = UNet(in_channels=modality,
                         out_channels=modality*num_of_cls,
                         for_seg=True)
        ####################################################################
        self.modality = modality
        self.num_of_cls = num_of_cls

    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x, 1)
        pred = self.nets(x)
        Slist = []
        for i in range(self.modality):
            Slist.append(pred[:, i * self.num_of_cls:(i + 1) *
                              self.num_of_cls, :, :, :])
        Slist = torch.cat(Slist, 1)
        return Slist


class RSModel(nn.Module):
    def __init__(self, seg_loss, reg_loss, imp_loss, num_modality=3, num_cls=2):
        super(RSModel, self).__init__()
        self.segmenter = Segmenter(num_modality, 1)
        self.register = ShapeMorph3d(4, num_modality)
        self.num_modality = num_modality
        self.num_cls = num_cls
        self.seg_loss = seg_loss
        self.reg_loss = reg_loss
        self.imp_loss = imp_loss

    def forward(self, I: list, S: list, baseline=False):
        # S=S+1

        # Segment phase 1
        S0 = self.segmenter(I)  # [batch_size, channel, z, x, y]
        S0 = torch.tanh(S0)
        Ls1 = self.seg_loss(S0, S) * 10
        if baseline:
            return Ls1, Ls1, Ls1, Ls1, Ls1, S0, S0, S0, S0, S0, S0

        # Regression phase 1
        I1, S1, DM = self.register(
            I[:, 1:, ...], S0[:, 1:, ...], I[:, 0:1, ...], S0[:, 0:1, ...]
        )
        Lr1 = (self.reg_loss(S1[:, 0:1, ...], S1[:, 1:2, ...], I1[:, 0:1, ...], I1[:, 1:2, ...], DM[:, 0:1, ...]) +
               self.reg_loss(S1[:, 0:1, ...], S1[:, 2:3, ...], I1[:, 0:1, ...], I1[:, 2:3, ...], DM[:, 1:2, ...]))

        # Segment phase 2
        S2 = self.segmenter(I1)
        S2 = torch.tanh(S2)
        Ls2 = self.seg_loss(S2, S) * 3

        # Regression phase 2
        I2, S3, DM2 = self.register(
            I1[:, 1:, ...], S2[:, 1:, ...], I1[:, 0:1, ...], S2[:, 0:1, ...]
        )
        Lr2 = (self.reg_loss(S3[:, 0:1, ...], S3[:, 1:2, ...], I2[:, 0:1, ...], I2[:, 1:2, ...], DM2[:, 0:1, ...]) +
               self.reg_loss(S3[:, 0:1, ...], S3[:, 2:3, ...], I2[:, 0:1, ...], I2[:, 2:3, ...], DM2[:, 1:2, ...])) * 0.5

        # Improvement loss
        Li = (self.imp_loss(Lr1 - Lr2,
                            torch.ones_like(Lr1 - Lr2).cuda()) +
              self.imp_loss(Ls1 - Ls2,
                            torch.ones_like(Ls1 - Ls2).cuda())) * 0.2
        return Ls1, Ls2, Lr1, Lr2, Li, S0, I1, S1, I2, S2, S3
