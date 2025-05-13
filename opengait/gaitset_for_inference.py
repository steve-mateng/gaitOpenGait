import torch
import torch.nn as nn
import copy

# 以下是从原项目中提取的关键模块
from modeling.modules import BasicConv2d, SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper
from modeling.modules import SeparateFCs


class GaitSetForInference(nn.Module):
    def __init__(self):
        super(GaitSetForInference, self).__init__()

        in_c = [1, 32, 64, 128]

        # Set Block 1
        self.set_block1 = nn.Sequential(
            BasicConv2d(in_c[0], in_c[1], kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(inplace=True),
            BasicConv2d(in_c[1], in_c[1], kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.set_block1 = SetBlockWrapper(self.set_block1)

        # Set Block 2
        self.set_block2 = nn.Sequential(
            BasicConv2d(in_c[1], in_c[2], kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            BasicConv2d(in_c[2], in_c[2], kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.set_block2 = SetBlockWrapper(self.set_block2)

        # Set Block 3
        self.set_block3 = nn.Sequential(
            BasicConv2d(in_c[2], in_c[3], kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            BasicConv2d(in_c[3], in_c[3], kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True)
        )
        self.set_block3 = SetBlockWrapper(self.set_block3)

        # Global Branch Blocks (deep copies)
        self.gl_block2 = copy.deepcopy(self.set_block2)
        self.gl_block3 = copy.deepcopy(self.set_block3)

        # Pooling
        self.set_pooling = PackSequenceWrapper(torch.max)

        # Head
        self.Head = SeparateFCs(
            in_channels=128,
            out_channels=256,
            parts_num=62
        )

        # Horizontal Pooling Pyramid
        self.HPP = HorizontalPoolingPyramid(bin_num=[16, 8, 4, 2, 1])

    # def forward(self, sils):
    #     """
    #     sils: [N, C, S, H, W]
    #     """
    #     if len(sils.size()) == 4:
    #         sils = sils.unsqueeze(1)  # [N, S, H, W] -> [N, C=1, S, H, W]
    #
    #     # Main Path
    #     outs = self.set_block1(sils)
    #     gl = self.set_pooling(outs, seqL=None, options={"dim": 2})[0]
    #     gl = self.gl_block2(gl)
    #
    #     outs = self.set_block2(outs)
    #     gl = gl + self.set_pooling(outs, seqL=None, options={"dim": 2})[0]
    #     gl = self.gl_block3(gl)
    #
    #     outs = self.set_block3(outs)
    #     outs = self.set_pooling(outs, seqL=None, options={"dim": 2})[0]
    #     gl = gl + outs
    #
    #     # Horizontal Pooling Matching
    #     feature1 = self.HPP(outs)  # [n, c, p]
    #     feature2 = self.HPP(gl)    # [n, c, p]
    #     feature = torch.cat([feature1, feature2], dim=-1)  # [n, c, p]
    #
    #     embs = self.Head(feature)  # [n, c, p] -> [n, emb_dim]
    #
    #     return embs

    def forward(self, sils):
        if len(sils.size()) == 4:
            sils = sils.unsqueeze(1)  # [N, S, H, W] -> [N, C=1, S, H, W]

        outs = self.set_block1(sils)

        gl = self.set_pooling(outs, seqL=None, options={"dim": 2})[0].unsqueeze(2)
        gl = self.gl_block2(gl)

        outs = self.set_block2(outs)
        gl_ = self.set_pooling(outs, seqL=None, options={"dim": 2})[0].unsqueeze(2)
        gl = gl + gl_
        gl = self.gl_block3(gl)

        outs = self.set_block3(outs)
        outs = self.set_pooling(outs, seqL=None, options={"dim": 2})[0].unsqueeze(2)
        gl = gl + outs

        feature1 = self.HPP(outs)
        feature2 = self.HPP(gl)
        feature = torch.cat([feature1, feature2], dim=-1)
        embs = self.Head(feature)

        return embs

