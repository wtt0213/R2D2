import os
import json
import pdb
import numpy as np


class Dataset(object):
    root = ''
    img_dir = ''
    nimg = 0

    def __len__(self):
        return self.nimg

    def get_key(self, img_idx):
        raise NotImplementedError()

    def get_filename(self, img_idx, root=None):
        return os.path.join(root or self.root, self.img_dir, self.get_key(img_idx))

    def get_image(self, img_idx):
        from PIL import Image
        fname = self.get_filename(img_idx)
        try:
            return Image.open(fname).convert('RGB')
        except Exception as e:
            raise IOError("Could not load image %s (reason: %s)" % (fname, str(e)))

    def __repr__(self):
        res = 'Dataset: %s\n' % self.__class__.__name__
        res += '  %d images' % self.nimg
        res += '\n  root: %s...\n' % self.root
        return res


class CatDataset (Dataset):
    def __init__(self, *datasets):
        assert len(datasets) >= 1
        self.datasets = datasets
        offsets = [0]
        for db in datasets:
            offsets.append(db.nimg)
        # 具化每一个dataset的第一个图片在总set中的偏移
        self.offsets = np.cumsum(offsets)
        self.nimg = self.offsets[-1]
        self.root = None

    def which(self, i):
        # pos 是i在offset中的最后一个小于等于i的位置
        # 这里就是确定是在那个子dataset中
        pos = np.searchsorted(self.offsets, i, side='right') - 1
        assert pos < self.nimg, 'Bad image index %d >= %d' % (i, self.nimg)
        return pos, i - self.offsets[pos]

    def get_key(self, i):
        b, i = self.which(i)
        return self.datasets[b].get_key(i)

    def get_filename(self, i, root=None):
        b, i = self.which(i)
        return self.datasets[b].get_filename(i)

    def __repr__(self):
        fmt_str = "CatDataset("
        for db in self.datasets:
            fmt_str += str(db).replace("\n", " ") + ', '
        return fmt_str[:-2] + ')'
