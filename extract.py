import os
import pdb
from PIL import Image
import numpy as np
import torch

from tools import common
from tools.dataloader import *
from nets.patchnet import *


class NonMaxSuppression(torch.nn.Module):
    """
    对输入的 reliability, repeatability, 进行局部极值提取
    """

    def __init__(self, rel_thr=0.7, rep_thr=0.7):
        nn.Module.__init__(self)
        self.max_filter = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.relthr = rel_thr
        self.repthr = rep_thr

    def forward(self, reliability, repeatability, **kw):
        assert len(reliability) == len(repeatability) == 1, "shape of reliability and repeatability are not the same"
        reliability, repeatability = reliability[0], repeatability[0]
        # 计算局部极值
        maxima = (repeatability == self.max_filter(repeatability))
        # 将低于阈值部分的值抛弃
        maxima *= (repeatability >= self.relthr)
        maxima *= (reliability >= self.relthr)
        # 现在的maxima大下是[1,1,H,W] ∈ {0，1}，这里Batch默认是1
        # 这里nonzero返回的是4个array中的后两个，
        return maxima.nonzero().t()[2:4]


def load_network(model_fn):
    checkpoint = torch.load(model_fn)
    print("\n>> Creating net = " + checkpoint['net'])
    # checkpoint['net'] = "Quad_L2Net_ConfCFS()" 字符串
    # checkpoint['state_dict'] = Dict 存储网络的参数值
    net = eval(checkpoint['net'])
    # 这里就是执行函数 Quad_L2Net_ConfCFS() 返回一个网络
    nb_of_weights = common.model_size(net)
    print(f" ( Model size: {nb_of_weights / 1000:.0f}K parameters )")
    # 初始化
    weights = checkpoint['state_dict']
    # 这里将k(key，即层的名称)修改个名字后和k对应的一个数组重新组成一个新的dict
    net.load_state_dict({k.replace('module.', ''): v for k, v in weights.items()})
    return net.eval()


def extract_multiscale(net, img, detector, scale_f=2 ** 0.25,
                       min_scale=0.0, max_scale=1,
                       min_size=256, max_size=1024,
                       verbose=False):
    """
    :param net: 输入的网络模型
    :param img: 输入的测试图片
    :param detector: non-maxima detector
    :param scale_f: 缩放因子，即论文中说的 2^(1/4)
    :param min_scale: 0
    :param max_scale: 1
    :param min_size: 最小的尺寸：256(根据论文中所说的，就是最长边的大小)
    :param max_size: 最大尺寸：1024
    :param verbose: 是否打印当前的尺寸信息
    :return:
    """
    # speed up
    old_bm = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False

    # 多尺寸上提取，注意这里的H和W是原图片尺寸
    B, three, H, W = img.shape
    assert B == 1 and three == 3, "should be a batch with a single RGB image"
    assert max_scale <= 1
    # cur_scale_f
    s = 1.0

    X, Y, S, C, Q, D = [], [], [], [], [], []

    # 这里要保证s的缩放不会使图片小于min_size
    while s + 0.001 >= max(min_scale, min_size / max(H, W)):
        if s - 0.001 <= min(max_scale, max_size / max(H, W)):
            # 这里记录的是缩放后的图片大小
            nh, nw = img.shape[2:]
            if verbose:
                print(f"extracting at scale x{s:.02f} = {nw:4d}x{nh:3d}")
            # 开始提取
            with torch.no_grad():
                res = net(imgs=[img])
            # 将得到的三个结果保存，但是注意输出是list[[1,n,H,W]]，baseNet中提及过
            # 所以要的是第一项,取出[1,n,H,W](n=1)
            descriptors = res['descriptors'][0]
            reliability = res['reliability'][0]
            repeatability = res['repeatability'][0]
            # 得到局部极大值的位置
            y, x = detector(**res)
            # 将对应位置的置信度和提取出来，假设点有n个，c和q都是torch.size([n])
            c = reliability[0, 0, y, x]
            q = repeatability[0, 0, y, x]
            # d 的维度是[128,n],转置一下为[n,128]
            d = descriptors[0, :, y, x].t()
            # n = 点的个数
            n = d.shape[0]

            # 将结果记录下来，这里要将位置还原
            X.append(x.float() * W / nw)
            Y.append(y.float() * H / nh)
            # 将各个信息记录
            C.append(c)
            Q.append(q)
            D.append(d)
            # 将缩放因子记录，这里是取32/s
            S.append((32 / s) * torch.ones(n, dtype=torch.float32, device=d.device))

        s /= scale_f
        # 每一次循环将图片缩小
        nh, nw = round(H * s), round(W * s)
        img = F.interpolate(img, (nh, nw), mode='bilinear', align_corners=False)

    # 还原bm
    torch.backends.cudnn.benchmark = old_bm

    # 假设在所有尺度下一共检测出来的点是N个
    # Y = [N,1] N个点的纵坐标
    Y = torch.cat(Y)
    # X = [N,1] N个点的横坐标
    X = torch.cat(X)
    # S = [N,1] N个点的缩放因子（32/s）
    S = torch.cat(S)
    # D = [N,128] N个点的描述子
    D = torch.cat(D)
    # 这里直接将repeatability和reliability乘积求出来（也是还原论文中的同时考虑两个因素下的高置信度）
    scores = torch.cat(C) * torch.cat(Q)
    # 进一步将位置信息和缩放信息整合
    XYS = torch.stack([X, Y, S], dim=-1)
    return XYS, D, scores


def extract_keypoints(args):
    iscuda = common.torch_set_gpu(args.gpu)
    # load net
    net = load_network(args.model)
    if iscuda:
        net = net.cuda()
    detector = NonMaxSuppression(
        rel_thr=args.reliability_thr,
        rep_thr=args.repeatability_thr)
    # 将输入的图片一张张的提取
    while args.images:
        img_path = args.images.pop(0)
        # 这个就是输入的是图片名称数组
        if img_path.endswith('.txt'):
            args.images = open(img_path).read().splitlines() + args.images
            continue
        print(f"\nExtracting features for {img_path}")
        img = Image.open(img_path).convert('RGB')
        W, H = img.size
        img = norm_RGB(img)[None]
        if iscuda: img = img.cuda()
        xys, desc, scores = extract_multiscale(net, img, detector,
                                               scale_f=args.scale_f,
                                               min_scale=args.min_scale,
                                               max_scale=args.max_scale,
                                               min_size=args.min_size,
                                               max_size=args.max_size,
                                               verbose=True)
        xys = xys.cpu().numpy()
        desc = desc.cpu().numpy()
        scores = scores.cpu().numpy()
        # 这里返回前n名的位置
        idxs = scores.argsort()[-args.top_k or None:]
        outpath = img_path + '.' + args.tag
        print(f"Saving {len(idxs)} keypoints to {outpath}")
        np.savez(open(outpath, 'wb'),
                 imsize=(W, H),
                 keypoints=xys[idxs],
                 descriptors=desc[idxs],
                 scores=scores[idxs])


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser("Extract keypoints for a given image")
    parser.add_argument("--model", type=str, required=True, help='model path')

    parser.add_argument("--images", type=str, required=True, nargs='+', help='images / list')
    parser.add_argument("--tag", type=str, default='r2d2', help='output file tag')

    parser.add_argument("--top-k", type=int, default=5000, help='number of keypoints')

    parser.add_argument("--scale-f", type=float, default=2 ** 0.25)
    parser.add_argument("--min-size", type=int, default=256)
    parser.add_argument("--max-size", type=int, default=1024)
    parser.add_argument("--min-scale", type=float, default=0)
    parser.add_argument("--max-scale", type=float, default=1)

    parser.add_argument("--reliability-thr", type=float, default=0.7)
    parser.add_argument("--repeatability-thr", type=float, default=0.7)

    parser.add_argument("--gpu", type=int, nargs='+', default=[0], help='use -1 for CPU')
    args = parser.parse_args()

    extract_keypoints(args)