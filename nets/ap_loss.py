import pdb
import numpy as np
import torch
import torch.nn as nn


class APLoss(nn.Module):
    """ differentiable AP loss, through quantization.

        Input: (N, M)   values in [min, max]
        label: (N, M)   values in {0, 1}

        Returns: list of query AP (for each n in {1..N})
                 Note: typically, you want to minimize 1 - mean(AP)
    """

    def __init__(self, nq=25, min=0, max=1, euc=False):
        nn.Module.__init__(self)
        assert isinstance(nq, int) and 2 <= nq <= 100
        # nq代表q的个数
        self.nq = nq
        self.min = min
        self.max = max
        self.euc = euc
        gap = max - min
        assert gap > 0

        # init quantizer = non-learnable (fixed) convolution
        # 这个一维卷积就是让所有的值先除以gap进行归一化，然后
        self.quantizer = q = nn.Conv1d(1, 2 * nq, kernel_size=1, bias=True)
        a = (nq - 1) / gap
        # 1st half = lines passing to (min+x,1) and (min+x+1/a,0) with x = {nq-1..0}*gap/(nq-1)
        # 前nq个
        q.weight.data[:nq] = -a
        q.bias.data[:nq] = torch.from_numpy(a * min + np.arange(nq, 0, -1))  # b = 1 + a*(min+x)

        # 2nd half = lines passing to (min+x,1) and (min+x-1/a,0) with x = {nq-1..0}*gap/(nq-1)
        # 后nq个
        q.weight.data[nq:] = a
        q.bias.data[nq:] = torch.from_numpy(np.arange(2 - nq, 2, 1) - a * min)  # b = 1 - a*(min+x)
        # first and last one are special: just horizontal straight line
        q.weight.data[0] = q.weight.data[-1] = 0
        q.bias.data[0] = q.bias.data[-1] = 1

    def compute_AP(self, x, label):
        N, M = x.shape
        if self.euc:  # euclidean distance in same range than similarities
            x = 1 - torch.sqrt(2.001 - 2 * x)

        # quantize all predictions
        # 加入定义的一维卷积的特征数1，即[N,1,M]
        q = self.quantizer(x.unsqueeze(1))
        # 经过一维卷积后的结果 q [N,2*nq,M],然后对应的第二维的值比较选择最小值，若小于0就等于0
        q = torch.min(q[:, :self.nq], q[:, self.nq:]).clamp(min=0)  # [N,nq,M]

        nbs = q.sum(dim=-1)  # number of samples  N x Q = c
        rec = (q * label.view(N, 1, M).float()).sum(dim=-1)  # nb of correct samples = c+ N x Q
        # 这里用cumsum将行加在一起，现在每一个点都是对应的该nq的从第一行到该行的区域的累加
        # [N,Q(nq)]
        prec = rec.cumsum(dim=-1) / (1e-16 + nbs.cumsum(dim=-1))  # precision
        # 每一行的所有nq加在一起，到时候一处得到归一化的数据
        rec /= rec.sum(dim=-1).unsqueeze(1)  # norm in [0,1]

        # 按nq加到一起，得到一张图片的AP
        ap = (prec * rec).sum(dim=-1)  # per-image AP

        print(ap.shape)
        return ap

    def forward(self, x, label):
        assert x.shape == label.shape  # N x M
        return self.compute_AP(x, label)

