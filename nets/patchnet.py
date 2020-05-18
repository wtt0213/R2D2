import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseNet(nn.Module):
    """
    input:[n,d,h,w] n张图片，d维度
    输入一系列的图片，为每一个图片返回两个东西：
    ① 像素级别的描述符
    ② 像素级别的heatMap，即每一个点的置信度
    """
    def softmax(self, ux):
        """
        :param ux: 可能是一列或两列
        :return: 根据列数 将输入的数组 用 softplus平滑并 归一至[0,1] 或 直接softmax
        """
        # 这里对应论文中的输出R和S的倒数第二步，然后经过一个normalize后调用本函数
        # 这里本来应该都是2维的，但是可能是作者出现了一些失误，这里的repeatability classifier是1维的，所以只能
        # 用softplus来模拟
        if ux.shape[1] == 1:
            x = F.softplus(ux)
            # 保证在区间[0,1]
            return x / (1 + x)
        elif ux.shape[1] == 2:
            return F.softmax(ux, dim=1)[:, 1:2]

    def normalize(self, x, uReliability, uRepeatability):
        """
        :param x: 输入的像素级别的描述符 [n,128,h,w]
        :param uReliability:  pixel-wise reliable 置信度 [n,1,h,w]
        :param uRepeatability: pixel-wise repeatable 置信度 [n,1,h,w]
        :return:
        """
        return dict(descriptors=F.normalize(x, p=2, dim=1),
                    repeatability=self.softmax(uRepeatability),
                    reliability=self.softmax(uReliability))

    def forward_one(self, x):
        raise NotImplementedError()

    def forward(self, imgs, **kw):
        res = [self.forward_one(img) for img in imgs]
        # 把list中的所有dict全部融合为一个,一个k对应着把每个的图片输出的结果中的k项放在一个列表中
        res = {k: [r[k] for r in res if k in r] for k in {k for r in res for k in r}}
        return dict(res, imgs=imgs, **kw)


class PatchNet(BaseNet):
    """
    Helper class: 构造一个完全卷积的网络，该网络提取l2规范化的 patch descriptor。
    """

    def __init__(self, inchan=3, dilated=True, dilation=1, bn=True, bn_affine=False):
        """
        :param inchan: 输入的通道数，默认是3
        :param dilated: 是否采用卷积核膨胀
        :param dilation: 卷积核膨胀系数
        :param bn: 是否 batch_normalize
        :param bn_affine:
        """
        BaseNet.__init__(self)
        self.inchan = inchan
        # 当前处理的通道数
        self.curchan = inchan
        self.dilated = dilated
        self.dilation = dilation
        self.bn = bn
        self.bn_affine = bn_affine
        # 操作序列
        self.ops = nn.ModuleList([])

    def _make_bn(self, outd):
        return nn.BatchNorm2d(outd, affine=self.bn_affine)

    def _add_conv(self, outd, k=3, stride=1, dilation=1, bn=True, relu=True):
        d = self.dilation * dilation
        if self.dilated:
            conv_params = dict(padding=((k - 1) * d) // 2, dilation=d, stride=1)
            self.dilation *= stride
        else:
            conv_params = dict(padding=((k - 1) * d) // 2, dilation=d, stride=stride)
        self.ops.append(nn.Conv2d(self.curchan, outd, kernel_size=k, **conv_params))
        if bn and self.bn:
            self.ops.append(self._make_bn(outd))
        if relu:
            self.ops.append(nn.ReLU(inplace=True))
        self.curchan = outd

    def forward_one(self, x):
        assert self.ops, "You need to add convolutions first"
        for n, op in enumerate(self.ops):
            x = op(x)
        # 这里只输出descriptor X，源码中此处有问题，调用的是self.normalize，这个需要输入三个参数
        # 即也会包括R和S
        return F.normalize(x, p=2, dim=1)


class L2_Net(PatchNet):
    def __init__(self, dim=128, **kw):
        PatchNet.__init__(self, **kw)
        self._add_conv((32 * dim) // 128, **kw)
        self._add_conv((32 * dim) // 128, **kw)
        self._add_conv((64 * dim) // 128, stride=2, **kw)
        self._add_conv((64 * dim) // 128, **kw)
        self._add_conv((128 * dim) // 128, stride=2, **kw)
        self._add_conv((128 * dim) // 128, **kw)
        self._add_conv((128 * dim) // 128, k=7, stride=8, bn=False, relu=False, **kw)
        self.out_dim = dim


class Quad_L2Net(PatchNet):
    """
    将L2net的最后的8*8卷积换为连续的三个2*2
    """

    def __init__(self, dim=128, mchan=4, relu22=False, **kw):
        PatchNet.__init__(self, **kw)
        self._add_conv(8 * mchan)
        self._add_conv(8 * mchan)
        self._add_conv(16 * mchan, stride=2)
        self._add_conv(16 * mchan)
        self._add_conv(32 * mchan, stride=2)
        self._add_conv(32 * mchan)
        # replace last 8x8 convolution with 3 2x2 convolutions
        self._add_conv(32 * mchan, k=2, stride=2, relu=relu22)
        self._add_conv(32 * mchan, k=2, stride=2, relu=relu22)
        self._add_conv(dim, k=2, stride=2, bn=False, relu=False)
        self.out_dim = dim


class Quad_L2Net_ConfCFS (Quad_L2Net):
    """
    将三个输出直接输出,包括 X,R,S
    """
    def __init__(self, **kw):
        Quad_L2Net.__init__(self, **kw)
        # reliability classifier
        self.clf = nn.Conv2d(self.out_dim, 2, kernel_size=1)
        # repeatability classifier 这里是1就有点尴尬了……本来应该是2的，作者写错了
        self.sal = nn.Conv2d(self.out_dim, 1, kernel_size=1)

    def forward_one(self, x):
        assert self.ops, "You need to add convolutions first"
        for op in self.ops:
            x = op(x)
        uReliability = self.clf(x**2)
        uRepeatability = self.sal(x**2)
        return self.normalize(x, uReliability, uRepeatability)
