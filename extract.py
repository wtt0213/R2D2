import os
import pdb
from PIL import Image
import numpy as np
import torch

from tools import common
from tools.dataloader import *
from nets.patchnet import *


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

