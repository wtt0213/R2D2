import pdb
from PIL import Image
import numpy as np

import torch
import torchvision.transforms as tvf

from tools.transforms import instanciate_transformation
from tools.transforms_tools import persp_apply

RGB_mean = [0.485, 0.456, 0.406]
RGB_std = [0.229, 0.224, 0.225]

norm_RGB = tvf.Compose([tvf.ToTensor(), tvf.Normalize(mean=RGB_mean, std=RGB_std)])


class PairLoader:
    """PairLoader(CatPairDataset(`data`),
        scale   = 'RandomScale(256,1024,can_upscale=True)',
        distort = 'ColorJitter(0.2,0.2,0.2,0.1)',
        crop    = 'RandomCrop(192)')"""

    def __init__(self, dataset, crop='', scale='', distort='', norm=norm_RGB,
                 what='aflow mask', idx_as_rng_seed=False):
        assert hasattr(dataset, 'npairs')
        assert hasattr(dataset, 'get_pair')
        self.dataset = dataset
        self.distort = instanciate_transformation(distort)
        self.crop = instanciate_transformation(crop)
        self.norm = instanciate_transformation(norm)
        self.scale = instanciate_transformation(scale)
        self.idx_as_rng_seed = idx_as_rng_seed
        self.what = what.split() if isinstance(what, str) else what
        self.n_samples = 5

    def __len__(self):
        assert len(self.dataset) == self.dataset.npairs, pdb.set_trace()  # and not nimg
        return len(self.dataset)

    def __repr__(self):
        fmt_str = 'PairLoader\n'
        fmt_str += repr(self.dataset)
        fmt_str += '  npairs: %d\n' % self.dataset.npairs
        short_repr = lambda s: repr(s).strip().replace('\n', ', ')[14:-1].replace('    ', ' ')
        fmt_str += '  Distort: %s\n' % short_repr(self.distort)
        fmt_str += '  Crop: %s\n' % short_repr(self.crop)
        fmt_str += '  Norm: %s\n' % short_repr(self.norm)
        return fmt_str

    def __getitem__(self, i):
        # 每次都生成新的随机数
        if self.idx_as_rng_seed:
            import random
            random.seed(i)
            np.random.seed(i)

        img_a, img_b, metadata = self.dataset.get_pair(i, self.what)
        # aflow 大小是[H, W, 2]
        aflow = np.float32(metadata['aflow'])
        # mask 大小是[H, W],若没有默认值是全1
        mask = metadata.get('mask', np.ones(aflow.shape[:2], np.uint8))

        # 将变换放到第二张图片
        img_b = {'img': img_b, 'persp': (1, 0, 0, 0, 1, 0, 0, 0)}
        if self.scale:
            img_b = self.scale(img_b)
        if self.distort:
            img_b = self.distort(img_b)

        # 将对应的aflow和flow改变
        aflow[:] = persp_apply(img_b['persp'], aflow.reshape(-1, 2)).reshape(aflow.shape)

        # 将对应的corres改变
        corres = None
        if 'corres' in metadata:
            corres = np.float32(metadata['corres'])
            corres[:, 1] = persp_apply(img_b['persp'], corres[:, 1])

        # apply the same transformation to the homography
        homography = None
        if 'homography' in metadata:
            homography = np.float32(metadata['homography'])
            # p_b = homography * p_a
            persp = np.float32(img_b['persp'] + (1,)).reshape(3, 3)
            homography = persp @ homography

        img_b = img_b['img']
        crop_size = self.crop({'imsize': (10000, 10000)})['imsize']
        output_size_a = min(img_a.size, crop_size)
        output_size_b = min(img_b.size, crop_size)
        img_a = np.array(img_a)
        img_b = np.array(img_b)

        ah, aw, p1 = img_a.shape
        bh, bw, p2 = img_b.shape
        assert p1 == 3
        assert p2 == 3
        assert aflow.shape == (ah, aw, 2)
        assert mask.shape == (ah, aw)

        # 计算光流的变化尺度，这里用来考虑一个点所对应的窗口大小
        # 输出是x坐标在w方向(dx[0])和h方向(dx[1])的梯度以及y坐标在w方向(dy[0])和h方向(dy[1])的梯度
        dx = np.gradient(aflow[:, :, 0])
        dy = np.gradient(aflow[:, :, 1])
        # scale在一定的范围内,[H*W]
        scale = np.sqrt(np.clip(np.abs(dx[1] * dy[0] - dx[0] * dy[1]), 1e-16, 1e16))

        accu2 = np.zeros((16, 16), bool)
        Q = lambda x, w: \
            np.int32(16 * (x - w.start) / (w.stop - w.start))

        # 生成一个窗口的一个维度上的起始位置
        def window1(x, size, w):
            l = x - int(0.5 + size / 2)
            r = l + int(0.5 + size)
            if l < 0:
                l, r = (0, r - l)
            if r > w:
                l, r = (l + w - r, w)
            if l < 0:
                l, r = 0, w  # larger than width
            return slice(l, r)

        # 根据两个维度信息生成一个窗口,返回窗口内的所有点，只不过表示成n:m的形式，到时候应用在矩阵上就ok了
        def window(cx, cy, win_size, scale, img_shape):
            return (window1(cy, win_size[1] * scale, img_shape[0]),
                    window1(cx, win_size[0] * scale, img_shape[1]))

        n_valid_pixel = mask.sum()
        # 防止除0
        sample_w = mask / (1e-16 + n_valid_pixel)

        def sample_valid_pixel():
            # 从mask中等概率的取出一个点
            n = np.random.choice(sample_w.size, p=sample_w.ravel())
            y, x = np.unravel_index(n, sample_w.shape)
            # 还原点的坐标
            return x, y

        # 找到一个合适的窗口

        # 找到最合适的窗口数量
        trials = 0
        best = -np.inf, None
        for _ in range(50 * self.n_samples):
            # 如果对应的sample找够了或者在原图中没有valid的点就直接break
            if trials >= self.n_samples or n_valid_pixel == 0:
                break
            # 这是在原图的位置
            c1x, c1y = sample_valid_pixel()
            # 通过flow获得在变化后的图片中的位置
            c2x, c2y = (aflow[c1y, c1x] + 0.5).astype(np.int32)
            # 判断位置的可行性质，超过范围了就舍弃
            if not (0 <= c2x < bw and 0 <= c2y < bh):
                continue

            # Get the flow scale
            sigma = scale[c1y, c1x]

            # Determine sampling windows
            if 0.2 < sigma < 1:
                win1 = window(c1x, c1y, output_size_a, 1 / sigma, img_a.shape)
                win2 = window(c2x, c2y, output_size_b, 1, img_b.shape)
            elif 1 <= sigma < 5:
                win1 = window(c1x, c1y, output_size_a, 1, img_a.shape)
                win2 = window(c2x, c2y, output_size_b, sigma, img_b.shape)
            else:
                # bad scale
                continue

            # 将原图的窗口的所有点用aflow变成变换后图片的点,x2是所有横坐标，y2是所有纵坐标
            x2, y2 = aflow[win1].reshape(-1, 2).T.astype(np.int32)
            # 带回到win2中，找到窗口中的valid点，将在范围中点记录下来
            valid = (win2[1].start <= x2) & (x2 < win2[1].stop) & (win2[0].start <= y2) & (y2 < win2[0].stop)

            # 将valid点和mask中的有效点相乘得到最终的平均有效点score1 ∈[0，1]
            score1 = (valid * mask[win1].ravel()).mean()

            # win2中的valid点的分数
            # 每次迭代重新初始化accu2
            accu2[:] = False
            # y2[valid]会拿到y2中满足范围的纵坐标值
            accu2[Q(y2[valid], win2[0]), Q(x2[valid], win2[1])] = True
            score2 = accu2.mean()

            # Check how many hits we got
            score = min(score1, score2)

            trials += 1
            if score > best[0]:
                best = score, win1, win2

        # TODO


def collate(batch, _use_shared_memory=True):
    """Puts each data field into a tensor with outer dimension batch size.
    Copied from https://github.com/pytorch in torch/utils/data/_utils/collate.py
    """
    import re
    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        out = None
        if _use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        assert elem_type.__name__ == 'ndarray'
        # array of string classes and object
        if re.search('[SaUO]', elem.dtype.str) is not None:
            raise TypeError(error_msg.format(elem.dtype))
        batch = [torch.from_numpy(b) for b in batch]
        try:
            return torch.stack(batch, 0)
        except RuntimeError:
            return batch
    elif batch[0] is None:
        return list(batch)
    elif isinstance(batch[0], int):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], str):
        return batch
    elif isinstance(batch[0], dict):
        return {key: collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [collate(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))


def tensor2img(tensor, model=None):
    """ convert back a torch/numpy tensor to a PIL Image
        by undoing the ToTensor() and Normalize() transforms.
    """
    mean = norm_RGB.transforms[1].mean
    std = norm_RGB.transforms[1].std
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()

    res = np.uint8(np.clip(255 * ((tensor.transpose(1, 2, 0) * std) + mean), 0, 255))
    from PIL import Image
    return Image.fromarray(res)

