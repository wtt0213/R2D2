import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FullSampler(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.mode = 'bilinear'
        self.padding = 'zeros'

    # 转换为后面grid_sample能调用的格式,即uniform的grid，归一化到[-1,1]
    @staticmethod
    def _aflow_to_grid(aflow):
        # 传入的aflow的大小是[B,2,h,w]
        H, W = aflow.shape[2:]
        grid = aflow.permute(0, 2, 3, 1).clone()
        # 归一化[0,2]
        grid[:, :, :, 0] *= 2 / (W - 1)
        grid[:, :, :, 1] *= 2 / (H - 1)
        # 归一化到[-1,1]
        grid -= 1
        grid[torch.isnan(grid)] = 9e9  # invalids
        return grid

    def _warp(self, feats, confs, aflow):
        if isinstance(aflow, tuple):
            return aflow  # result was precomputed
        feat1, feat2 = feats
        conf1, conf2 = confs if confs else (None, None)
        B, two, H, W = aflow.shape
        D = feat1.shape[1]
        assert feat1.shape == feat2.shape == (B, D, H, W)  # D = 128, B = batch
        assert conf1.shape == conf2.shape == (B, 1, H, W) if confs else True

        # 通过aflow将img_2 再变回 img_1
        grid = self._aflow_to_grid(aflow)
        ones2 = feat2.new_ones(feat2[:, 0:1].shape)
        feat2to1 = F.grid_sample(feat2, grid, mode=self.mode, padding_mode=self.padding)
        mask2to1 = F.grid_sample(ones2, grid, mode='nearest', padding_mode='zeros')
        conf2to1 = F.grid_sample(conf2, grid, mode=self.mode, padding_mode=self.padding) if confs else None
        return feat2to1, mask2to1.byte(), conf2to1

    def _warp_positions(self, aflow):
        B, two, H, W = aflow.shape
        assert two == 2

        Y = torch.arange(H, device=aflow.device)
        X = torch.arange(W, device=aflow.device)
        XY = torch.stack(torch.meshgrid(Y, X)[::-1], dim=0)  # [2,H,W]
        XY = XY[None].expand(B, 2, H, W).float()

        grid = self._aflow_to_grid(aflow)
        XY2 = F.grid_sample(XY, grid, mode='bilinear', padding_mode='zeros')
        return XY, XY2


class SubSampler(FullSampler):
    def __init__(self, border, subq, subd, perimage=False):
        """
        降采样
        :param border:
        :param subq:
        :param subd:
        :param perimage: 这里说明一下，如果是False，那么将batch里面所有图片全都混在一起，最后输出batch维度没有了
                         否则只计算一张图片，然后根据 batch扩展一下
        """
        FullSampler.__init__(self)
        assert subq % subd == 0, 'subq必须是subd的整数倍'
        self.sub_q = subq
        self.sub_d = subd
        self.border = border
        self.perimage = perimage

    def __repr__(self):
        return "SubSampler(border=%d, subq=%d, subd=%d, perimage=%d)" % (
            self.border, self.sub_q, self.sub_d, self.perimage)

    def __call__(self, feats, confs, aflow):
        feat1, conf1 = feats[0], (confs[0] if confs else None)
        # 将feat2，conf2，mask2用aflow转换回再img1中的位置
        feat2, mask2, conf2 = self._warp(feats, confs, aflow)

        # 这里需要注意一下，虽然subq是subd的整数倍，但是这是间隔取样的间隔数，间隔越大降采样后的图片更小
        # 降采样img1
        slq = slice(self.border, -self.border or None, self.sub_q)
        feat1 = feat1[:, :, slq, slq]
        conf1 = conf1[:, :, slq, slq] if confs else None
        # 降采样img2
        sld = slice(self.border, -self.border or None, self.sub_d)
        feat2 = feat2[:, :, sld, sld]
        mask2 = mask2[:, :, sld, sld]
        conf2 = conf2[:, :, sld, sld] if confs else None

        B, D, Hq, Wq = feat1.shape
        B, D, Hd, Wd = feat2.shape

        # 计算ground_true
        if self.perimage or self.sub_q != self.sub_d:
            # f拿到feature1的整尺寸，这里只需要取D中的一维就行，因为只要计算一个gt(位置关系)，不会看里面内容
            f = feats[0][0:1, 0] if self.perimage else feats[0][:, 0]
            # 将f中的各个像素的索引作成一个数组
            # 比如f =[3,4] 那么idxs = [ 0,1,2,3;4,5,6,7;8,9,10,11 ]
            idxs = torch.arange(f.numel(), dtype=torch.int64, device=feat1.device).view(f.shape)
            # 在索引数组中裁剪出降采样的相应区域
            idxs1 = idxs[:, slq, slq].reshape(-1, Hq * Wq)
            idxs2 = idxs[:, sld, sld].reshape(-1, Hd * Wd)

            if self.perimage:
                gt = (idxs1[0].view(-1, 1) == idxs2[0].view(1, -1))
                gt = gt[None, :, :].expand(B, Hq * Wq, Hd * Wd)
            else:
                gt = (idxs1.view(-1, 1) == idxs2.view(1, -1))
        else:
            # 这里如果sub_q和sub_d相同，那么直接是一个对角矩阵，尺寸是[B*Hq*Wq,B*Hq*Wq]
            gt = torch.eye(feat1[:, 0].numel(), dtype=torch.uint8, device=feat1.device)

        queries = feat1.reshape(B, D, -1)  # B x D x (Hq x Wq)
        database = feat2.reshape(B, D, -1)  # B x D x (Hd x Wd)

        if self.perimage:
            queries = queries.transpose(1, 2)  # B x (Hd x Wd) x D
            # 这里利用相同的数乘法最大计算score（对应的D相同那么加在一起的位置的和更大，score就更大）
            scores = torch.bmm(queries, database)  # B x (Hq x Wq) x (Hd x Wd)
        else:
            queries = queries.transpose(1, 2).reshape(-1, D)  # (B x Hq x Wq) x D
            database = database.transpose(1, 0).reshape(D, -1)  # D x (B x Hd x Wd)
            scores = torch.matmul(queries, database)  # (B x Hq x Wq) x (B x Hd x Wd)

        # 计算 reliability
        # TODO 这里有点不明白，对于conf1和conf2的shape不一样无法计算
        qconf = (conf1 + conf2) / 2 if confs else None

        assert gt.shape == scores.shape
        # scores [(B x Hq x Wq) , (B x Hd x Wd)]或 [B , (Hq x Wq) , (Hd x Wd)]
        # gt [B, Hq * Wq, Hd * Wd] 或 [B * Hq * Wq, B * Hd * Wd]
        # mask2 [B, D, sld.length, sld.length]
        # qconf 未知 如果尺寸一样 就是 [B, D, sld.length, sld.length]
        return scores, gt, mask2, qconf


class NghSampler (FullSampler):
    """
    neighborhood取样
    """
    def __init__(self, ngh, subq=1, subd=1, ignore=1, border=None):
        FullSampler.__init__(self)
        assert 0 <= ignore < ngh
        self.ngh = ngh
        self.ignore = ignore
        assert subd <= ngh
        self.sub_q = subq
        self.sub_d = subd
        if border is None: border = ngh
        assert border >= ngh, 'border has to be larger than ngh'
        self.border = border

    def __repr__(self):
        return "NghSampler(ngh=%d, subq=%d, subd=%d, ignore=%d, border=%d)" % (
            self.ngh, self.sub_q, self.sub_d, self.ignore, self.border)

    # 这里把subsample直接做成一个lambda直接调用
    def trans(self, arr, i, j):
        s = lambda i: slice(self.border + i, i - self.border or None, self.sub_q)
        return arr[:, :, s(j), s(i)]

    def __call__(self, feats, confs, aflow):

        feat1, conf1 = feats[0], (confs[0] if confs else None)
        feat2, mask2, conf2 = self._warp(feats, confs, aflow)
        qfeat = self.trans(feat1, 0, 0)
        qconf = (self.trans(conf1, 0, 0) + self.trans(conf2, 0, 0)) / 2 if confs else None
        mask2 = self.trans(mask2, 0, 0)
        # 将D的那个维度加在一起
        scores_at = lambda i, j: (qfeat * self.trans(feat2, i, j)).sum(dim=1)

        # 计算所有neighborhood的score
        B, D = feat1.shape[:2]
        min_d = self.ignore ** 2
        max_d = self.ngh ** 2
        # 这里计算可以走的步数
        rad = (self.ngh // self.sub_d) * self.ngh
        negs = []
        offsets = []
        # 这里模拟的是0，0为中心点，然后取0+i,0+j的一系列点为中心的neighborhood
        for j in range(-rad, rad + 1, self.sub_d):
            for i in range(-rad, rad + 1, self.sub_d):
                if not (min_d < i * i + j * j <= max_d):
                    # 不在范围里不行
                    continue
                offsets.append((i, j))
                negs.append(scores_at(i, j))

        scores = torch.stack([scores_at(0, 0)] + negs, dim=-1)
        gt = scores.new_zeros(scores.shape, dtype=torch.uint8)
        # 这里只有中心点对应的neighborhood为1
        gt[..., 0] = 1
        return scores, gt, mask2, qconf


class NghSampler2 (nn.Module):
    def __init__(self, ngh, subq=1, subd=1, pos_d=0, neg_d=2, border=None,
                 maxpool_pos=True, subd_neg=0):
        """
        :param ngh: neighborhood大小，即patch能够移动的范围，由于不能移动出去，所以一定小于border
        :param subq: query patch的取样间隔数
        :param subd: database patch的取样间隔数
        :param pos_d:
        :param neg_d:
        对于上面两个参数，都是对于距离GT的距离的一个标定值，距离在0~pos_d之内算正，距离在pos_d~neg_d算为0，距离>neg_d的算负
        :param border: 一开始选取的中心patch距离边界的距离
        :param maxpool_pos:
        :param subd_neg:
        """
        nn.Module.__init__(self)
        assert 0 <= pos_d < neg_d <= (ngh if ngh else 99)
        self.ngh = ngh
        self.pos_d = pos_d
        self.neg_d = neg_d
        assert subd <= ngh or ngh == 0
        assert subq != 0
        self.sub_q = subq
        self.sub_d = subd
        self.sub_d_neg = subd_neg
        if border is None: border = ngh
        assert border >= ngh, 'border has to be larger than ngh'
        self.border = border
        self.maxpool_pos = maxpool_pos
        self.precompute_offsets()

    def precompute_offsets(self):
        pos_d2 = self.pos_d ** 2
        neg_d2 = self.neg_d ** 2
        rad2 = self.ngh ** 2
        rad = (self.ngh // self.sub_d) * self.ngh
        pos = []
        neg = []
        for j in range(-rad, rad + 1, self.sub_d):
            for i in range(-rad, rad + 1, self.sub_d):
                d2 = i * i + j * j
                if d2 <= pos_d2:
                    pos.append((i, j))
                elif neg_d2 <= d2 <= rad2:
                    neg.append((i, j))
        self.register_buffer('pos_offsets', torch.LongTensor(pos).view(-1, 2).t())
        self.register_buffer('neg_offsets', torch.LongTensor(neg).view(-1, 2).t())

    def gen_grid(self, step, aflow):
        B, two, H, W = aflow.shape
        dev = aflow.device
        b1 = torch.arange(B, device=dev)
        if step > 0:
            # 返回的是从begin到end的range数列
            x1 = torch.arange(self.border, W - self.border, step, device=dev)
            y1 = torch.arange(self.border, H - self.border, step, device=dev)
            H1, W1 = len(y1), len(x1)
            x1 = x1[None, None, :].expand(B, H1, W1).reshape(-1)
            y1 = y1[None, :, None].expand(B, H1, W1).reshape(-1)
            b1 = b1[:, None, None].expand(B, H1, W1).reshape(-1)
            shape = (B, H1, W1)
        else:
            # 返回的是范围内的随机位置
            n = (H - 2 * self.border) * (W - 2 * self.border) // step ** 2
            x1 = torch.randint(self.border, W - self.border, (n,), device=dev)
            y1 = torch.randint(self.border, H - self.border, (n,), device=dev)
            x1 = x1[None, :].expand(B, n).reshape(-1)
            y1 = y1[None, :].expand(B, n).reshape(-1)
            b1 = b1[:, None].expand(B, n).reshape(-1)
            shape = (B, n)
        return b1, y1, x1, shape

    def forward(self, feats, confs, aflow, **kw):
        B, two, H, W = aflow.shape
        assert two == 2
        feat1, conf1 = feats[0], (confs[0] if confs else None)
        feat2, conf2 = feats[1], (confs[1] if confs else None)

        b1, y1, x1, shape = self.gen_grid(self.sub_q, aflow)

        feat1 = feat1[b1, :, y1, x1]
        qconf = conf1[b1, :, y1, x1].view(shape) if confs else None

        b2 = b1
        xy2 = (aflow[b1, :, y1, x1] + 0.5).long().t()
        mask = (0 <= xy2[0]) * (0 <= xy2[1]) * (xy2[0] < W) * (xy2[1] < H)
        mask = mask.view(shape)

        def clamp(xy):
            torch.clamp(xy[0], 0, W - 1, out=xy[0])
            torch.clamp(xy[1], 0, H - 1, out=xy[1])
            return xy

        # 首先计算在正范围内每一个点的最大得分以及对应的那个偏移

        # 这里利用广播机制，假设pos_offset有n个位置偏移可以用，即为[2:n_p]
        # 假设xy2中经过截取aflow中的点后得到的为[2:N]，即有N个对应信息(N=b1*y1*x1)
        # 这里相加之后为[2:n_p:N]，即N中的每一个点的n种offset种一种下的x，y坐标
        xy2p = clamp(xy2[:, None, :] + self.pos_offsets[:, :, None])
        # 这里再看一下维度关系 feat1[None, :, :]为[1,N,128]
        # feat2[b2, :, xy2p[1], xy2p[0]]为[n_p,N,128]
        # 最后结果是[N,n_p]
        pscores = (feat1[None, :, :] * feat2[b2, :, xy2p[1], xy2p[0]]).sum(dim=-1).t()

        if self.maxpool_pos:
            # 这里返回了n个位置里面的最大值以及对应的位置
            pscores, pos = pscores.max(dim=1, keepdim=True)
            if confs:
                # xy2->[2:N]
                # self.pos_offsets[:, pos.view(-1)] ->[2:N]
                sel = clamp(xy2 + self.pos_offsets[:, pos.view(-1)])
                qconf = (qconf + conf2[b2, :, sel[1], sel[0]].view(shape)) / 2

        # 接下来计算在范围外的点，和上面类似，这里不解释了
        xy2n = clamp(xy2[:, None, :] + self.neg_offsets[:, :, None])
        nscores = (feat1[None, :, :] * feat2[b2, :, xy2n[1], xy2n[0]]).sum(dim=-1).t()

        if self.sub_d_neg:
            # add distractors from a grid
            b3, y3, x3, _ = self.gen_grid(self.sub_d_neg, aflow)
            distractors = feat2[b3, :, y3, x3]
            dscores = torch.matmul(feat1, distractors.t())
            del distractors

            # remove scores that corresponds to positives or nulls
            dis2 = (x3 - xy2[0][:, None]) ** 2 + (y3 - xy2[1][:, None]) ** 2
            dis2 += (b3 != b2[:, None]).long() * self.neg_d ** 2
            dscores[dis2 < self.neg_d ** 2] = 0

            scores = torch.cat((pscores, nscores, dscores), dim=1)
        else:
            # concat everything
            # [N,n_p+n_n]
            scores = torch.cat((pscores, nscores), dim=1)

        gt = scores.new_zeros(scores.shape, dtype=torch.uint8)
        gt[:, :pscores.shape[1]] = 1

        return scores, gt, mask, qconf


