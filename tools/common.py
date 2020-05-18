import numpy as np
import torch
import os


def model_size(model):
    """ 计算模型的参数数量 """
    size = 0
    for weights in model.state_dict().values():
        # np.prod 计算参数数组内的所有元素乘积
        size += np.prod(weights.shape)
    return size


def mkdir_for(file_path):
    """ 创建目录 """
    os.makedirs(os.path.split(file_path)[0], exist_ok=True)


def torch_set_gpu(gpus):
    if type(gpus) is int:
        gpus = [gpus]

    cuda = all(gpu >= 0 for gpu in gpus)
    if cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(gpu) for gpu in gpus])
        assert cuda and torch.cuda.is_available(), "%s has GPUs %s unavailable" % (
            os.environ['HOSTNAME'], os.environ['CUDA_VISIBLE_DEVICES'])
        torch.backends.cudnn.benchmark = True  # speed-up cudnn
        torch.backends.cudnn.fastest = True  # even more speed-up?
        print('Launching on GPUs ' + os.environ['CUDA_VISIBLE_DEVICES'])

    else:
        print('Launching on CPU')

    return cuda
