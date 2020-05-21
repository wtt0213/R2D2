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

