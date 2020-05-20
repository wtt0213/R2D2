import pdb
import numpy as np
from PIL import Image, ImageOps, ImageEnhance


class DummyImg:
    """
    该类根据输入的图片尺寸创造出一个虚拟映像（不会有图像真实的值，知识返回一个尺寸信息）
    """

    def __init__(self, size):
        self.size = size

    @staticmethod
    def resize(size, *args, **kwargs):
        return DummyImg(size)

    def expand(self, border):
        """
        :param border: 如果是int就是两边长和宽扩展相同的长度（注意给定的是单边扩展长度）
                       否则就是给出四个方向需要扩展的长度，顺序为w+ h+ w- h-
        :return:
        """
        w, h = self.size
        if isinstance(border, int):
            size = (w + 2 * border, h + 2 * border)
        else:
            l, t, r, b = border
            size = (w + l + r, h + t + b)
        return DummyImg(size)

    def crop(self, border):
        """
        :param border: 这里的border必须是一个给定四个方向新的长度（不是扩展）
        :return:
        """
        w, h = self.size
        l, t, r, b = border
        assert 0 <= l <= r <= w, "裁剪大小不能超过原图像大小，应满足0 <= l <= r <= w"
        assert 0 <= t <= b <= h, "裁剪大小不能超过原图像大小，应满足0 <= t <= b <= h"
        size = (r - l, b - t)
        return DummyImg(size)

    def rotate(self, angle):
        raise NotImplementedError

    @staticmethod
    def transform(size, *args, **kwargs):
        """
        :param size: 转换到特定的变换，尺寸是不变的
        :param args:
        :param kwargs:
        :return:
        """
        return DummyImg(size)


def grab_img(img_and_label):
    """
    :param img_and_label: 处理有可能带有尺寸信息的图片，把其中的图片返回
    :return:
    """
    if isinstance(img_and_label, dict):
        # if input is a dictionary, then
        # it must contains the img or its size.
        try:
            return img_and_label['img']
        except KeyError:
            return DummyImg(img_and_label['imsize'])

    else:
        # or it must be the img directly
        return img_and_label


def persp_mul(mat, mat2):
    """
    homography multiplication,将多个变换融合
    :param mat: 8-tuple (homography transform)
    :param mat2: 8-tuple (homography transform)
    :return:
    """
    assert isinstance(mat, tuple)
    assert isinstance(mat2, tuple)

    # 将最后一维度（一定是1）加上，即将原来的8*1 变换为3*3
    mat = np.float32(mat + (1,)).reshape(3, 3)
    mat2 = np.array(mat2 + (1,)).reshape(3, 3)
    res = np.dot(mat, mat2)
    # 返回的时候再变换8*1(满秩形式)
    return tuple((res / res[2, 2]).ravel()[:8])


def update_img_and_labels(img_and_label, img, persp=None):
    """
    记录着从原图到现在的变换矩阵
    :param img_and_label:
    :param img:
    :param persp: perspective 变换矩阵
    :return:
    """
    if isinstance(img_and_label, dict):
        # 先进行更新。整体思路是更新的时候img_and_label中是旧的img和persp，而后面是新的
        img_and_label['img'] = img
        img_and_label['imsize'] = img.size
        if persp:
            if 'persp' not in img_and_label:
                img_and_label['persp'] = (1, 0, 0, 0, 1, 0, 0, 0)
            # 更新
            img_and_label['persp'] = persp_mul(persp, img_and_label['persp'])

        return img_and_label
    else:
        return img


def rand_log_uniform(a, b):
    return np.exp(np.random.uniform(np.log(a), np.log(b)))


def translate(tx, ty):
    return (1, 0, tx,
            0, 1, ty,
            0, 0)


def rotate(angle):
    return (np.cos(angle), -np.sin(angle), 0,
            np.sin(angle), np.cos(angle), 0,
            0, 0)


def persp_apply(mat, pts):
    """
    :param mat: homograph 8-tuple
    :param pts: 需要被应用的矩阵 numpy array
    :return:
    """
    assert isinstance(mat, tuple)
    assert isinstance(pts, np.ndarray)
    assert pts.shape[-1] == 2
    mat = np.float32(mat + (1,)).reshape(3, 3)
    # 如果只有一个点 即pts = 1*2
    if pts.ndim == 1:
        # 这里要保证pt也是一维的，用ravel处理一下
        pt = np.dot(pts, mat[:, :2].T).ravel() + mat[:, 2]
        pt /= pt[2]  # homogeneous coordinates
        # 返回二维坐标
        return tuple(pt[:2])
    # 有n个点 即pts = n*2
    else:
        pt = np.dot(pts, mat[:, :2].T) + mat[:, 2]
        pt[:, :2] /= pt[:, 2]
        # 返回二维坐标
        return tuple(pt[:, :2])


def is_pil_image(img):
    return isinstance(img, Image.Image)


# 调整亮度
def adjust_brightness(img, brightness_factor):
    """
    调整图片亮度
    :param img:
    :param brightness_factor: 0 gives a black image, 1 gives the
    original image while 2 increases the brightness by a factor of 2.
    :return:
    """
    # Copied from https://github.com/pytorch in torchvision/transforms/functional.py
    if not is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness_factor)
    return img


# 调整对比度
def adjust_contrast(img, contrast_factor):
    """Adjust contrast of an Image.
    Args:
    img (PIL Image): PIL Image to be adjusted.
    contrast_factor (float): How much to adjust the contrast. Can be any
    non negative number. 0 gives a solid gray image, 1 gives the
    original image while 2 increases the contrast by a factor of 2.
    Returns:
    PIL Image: Contrast adjusted image.
    Copied from https://github.com/pytorch in torchvision/transforms/functional.py
    """
    if not is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)
    return img


# 调整饱和度
def adjust_saturation(img, saturation_factor):
    """Adjust color saturation of an image.
    Args:
    img (PIL Image): PIL Image to be adjusted.
    saturation_factor (float):  How much to adjust the saturation. 0 will
    give a black and white image, 1 will give the original image while
    2 will enhance the saturation by a factor of 2.
    Returns:
    PIL Image: Saturation adjusted image.
    Copied from https://github.com/pytorch in torchvision/transforms/functional.py
    """
    if not is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(saturation_factor)
    return img


# 调整色调
def adjust_hue(img, hue_factor):
    """Adjust hue of an image.
    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.
    `hue_factor` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`.
    See https://en.wikipedia.org/wiki/Hue for more details on Hue.
    Args:
    img (PIL Image): PIL Image to be adjusted.
    hue_factor (float):  How much to shift the hue channel. Should be in
    [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
    HSV space in positive and negative direction respectively.
    0 means no shift. Therefore, both -0.5 and 0.5 will give an image
    with complementary colors while 0 gives the original image.
    Returns:
    PIL Image: Hue adjusted image.
    Copied from https://github.com/pytorch in torchvision/transforms/functional.py
    """
    if not(-0.5 <= hue_factor <= 0.5):
        raise ValueError('hue_factor is not in [-0.5, 0.5].'.format(hue_factor))

    if not is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    input_mode = img.mode
    if input_mode in {'L', '1', 'I', 'F'}:
        return img

    h, s, v = img.convert('HSV').split()

    np_h = np.array(h, dtype=np.uint8)
    # uint8 addition take cares of rotation across boundaries
    with np.errstate(over='ignore'):
        np_h += np.uint8(hue_factor * 255)
        h = Image.fromarray(np_h, 'L')

    img = Image.merge('HSV', (h, s, v)).convert(input_mode)
    return img