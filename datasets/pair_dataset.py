import os, pdb
import numpy as np
from PIL import Image

from .dataset import Dataset, CatDataset
from tools.transforms import instanciate_transformation
from tools.transforms_tools import persp_apply


def is_pair(x):
    if isinstance(x, (tuple, list)) and len(x) == 2:
        return True
    if isinstance(x, np.ndarray) and x.ndim == 1 and x.shape[0] == 2:
        return True
    return False


class PairDataset(Dataset):
    def __init__(self):
        Dataset.__init__(self)
        self.npairs = 0

    def get_filename(self, img_idx, root=None):
        if is_pair(img_idx):  # if img_idx is a pair of indices, we return a pair of filenames
            return tuple(Dataset.get_filename(self, i, root) for i in img_idx)
        return Dataset.get_filename(self, img_idx, root)

    def get_image(self, img_idx):
        if is_pair(img_idx):  # if img_idx is a pair of indices, we return a pair of images
            return tuple(Dataset.get_image(self, i) for i in img_idx)
        return Dataset.get_image(self, img_idx)

    def get_corres_filename(self, pair_idx):
        raise NotImplementedError()

    def get_homography_filename(self, pair_idx):
        raise NotImplementedError()

    def get_flow_filename(self, pair_idx):
        raise NotImplementedError()

    def get_mask_filename(self, pair_idx):
        raise NotImplementedError()

    def get_pair(self, idx, output=()):
        """ returns (img1, img2, `metadata`)

        `metadata` is a dict() that can contain:
            flow: optical flow
            aflow: absolute flow
            corres: list of 2d-2d correspondences
            mask: boolean image of flow validity (in the first image)
            ...
        """
        raise NotImplementedError()

    def get_paired_images(self):
        fns = set()
        for i in range(self.npairs):
            a, b = self.image_pairs[i]
            fns.add(self.get_filename(a))
            fns.add(self.get_filename(b))
        return fns

    def __len__(self):
        # 这里是pair的数目，不是图片
        return self.npairs

    def __repr__(self):
        res = 'Dataset: %s\n' % self.__class__.__name__
        res += '  %d images,' % self.nimg
        res += ' %d image pairs' % self.npairs
        res += '\n  root: %s...\n' % self.root
        return res

    @staticmethod
    def _flow2png(flow, path):
        flow = np.clip(np.around(16 * flow), -2 ** 15, 2 ** 15 - 1)
        bytes = np.int16(flow).view(np.uint8)
        Image.fromarray(bytes).save(path)
        return flow / 16

    @staticmethod
    def _png2flow(path):
        try:
            flow = np.asarray(Image.open(path)).view(np.int16)
            return np.float32(flow) / 16
        except:
            raise IOError("Error loading flow for %s" % path)


class SyntheticPairDataset (PairDataset):
    def __init__(self, dataset, scale='', distort=''):
        PairDataset.__init__(self)
        assert isinstance(dataset, Dataset) and not isinstance(dataset, PairDataset)
        self.dataset = dataset
        self.npairs = dataset.nimg
        self.get_image = dataset.get_image
        self.get_key = dataset.get_key
        self.get_filename = dataset.get_filename
        self.root = None
        self.distort = instanciate_transformation(distort)
        self.scale = instanciate_transformation(scale)

    def attach_dataset(self, dataset):
        assert isinstance(dataset, Dataset) and not isinstance(dataset, PairDataset)
        self.dataset = dataset
        self.npairs = dataset.nimg
        self.get_image = dataset.get_image
        self.get_key = dataset.get_key
        self.get_filename = dataset.get_filename
        self.root = None

    @staticmethod
    def make_pair(img):
        return img, img

    def get_pair(self, i, output='aflow'):
        if isinstance(output, str):
            output = output.split()
        # 这里做出图片对，即原图片和改变后的图片，tvf是变换矩阵
        original_img = self.dataset.get_image(i)
        scaled_image = self.scale(original_img)
        scaled_image, scaled_image2 = self.make_pair(scaled_image)
        scaled_and_distorted_image = self.distort(
            dict(img=scaled_image2, persp=(1, 0, 0, 0, 1, 0, 0, 0)))
        W, H = scaled_image.size
        # 这里是变换后的
        trf = scaled_and_distorted_image['persp']
        meta = dict()
        if 'aflow' in output or 'flow' in output:
            # compute optical flow
            # 这里xy生成了位置对，以列进行遍历,这里一定要先2*n再转置，否则会乱掉
            xy = np.mgrid[0:H, 0:W][::-1].reshape(2, H * W).T
            # 输出的还是(n,2)，只不过是原始点的对应点
            aflow = np.float32(persp_apply(trf, xy).reshape(H, W, 2))
            meta['flow'] = aflow - xy.reshape(H, W, 2)
            meta['aflow'] = aflow
        if 'homography' in output:
            meta['homography'] = np.float32(trf + (1,)).reshape(3, 3)

        return scaled_image, scaled_and_distorted_image['img'], meta

    def __repr__(self):
        res = 'Dataset: %s\n' % self.__class__.__name__
        res += '  %d images and pairs' % self.npairs
        res += '\n  root: %s...' % self.dataset.root
        res += '\n  Scale: %s' % (repr(self.scale).replace('\n', ''))
        res += '\n  Distort: %s' % (repr(self.distort).replace('\n', ''))
        return res + '\n'
