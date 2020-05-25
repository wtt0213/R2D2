import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.sampler import FullSampler


class CosimLoss(nn.Module):
    def __init__(self, N=16):
        nn.Module.__init__(self)
        self.name = f'cosim{N}'
        # 定义一个取patch的容器，在输入[B,D,H,W]的图片后，会根据N的大小
        # 以卷积的移动方式进行，输出向量是[B,(D*N*N),L)这里L是移动的次数(L=(H-N+1)*(W-N+1))
        self.patches = nn.Unfold(N, padding=0, stride=N // 2)

    def extract_patches(self, sal):
        # 将输出的向量变为[B,L,(D*N*N)]
        patches = self.patches(sal).transpose(1, 2)
        # 将第三维的向量归一化
        patches = F.normalize(patches, p=2, dim=2)
        return patches

    def forward(self, repeatability, aflow, **kw):
        B, two, H, W = aflow.shape
        assert two == 2

        # normalize
        sali1, sali2 = repeatability
        # 将aflow转换为grid[B,H,W,2],并进行一定的规范化
        grid = FullSampler._aflow_to_grid(aflow)
        # 根据光流变换矩阵u(grid)得到repeatable'对应的repeatable_u',即通过光流变化前的图像的repeatable，与
        # 输入的repeatable进行余弦相似度的计算，这里的u是根据两张原始图像得到的
        sali2 = F.grid_sample(sali2, grid, mode='bilinear', padding_mode='border')

        patches1 = self.extract_patches(sali1)
        patches2 = self.extract_patches(sali2)
        cosim = (patches1 * patches2).sum(dim=2)
        return 1 - cosim.mean()


class PeakyLoss(nn.Module):
    def __init__(self, N=16):
        nn.Module.__init__(self)
        self.name = f'peaky{N}'
        assert N % 2 == 0, 'N must be pair'
        # 通过padding进行原尺寸池化，输入输出尺寸相同
        self.preproc = nn.AvgPool2d(3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(N + 1, stride=1, padding=N // 2)
        self.avgpool = nn.AvgPool2d(N + 1, stride=1, padding=N // 2)

    def forward_one(self, sali):
        sali = self.preproc(sali)  # remove super high frequency
        return 1 - (self.maxpool(sali) - self.avgpool(sali)).mean()

    def forward(self, repeatability, **kw):
        sali1, sali2 = repeatability
        return (self.forward_one(sali1) + self.forward_one(sali2)) / 2
