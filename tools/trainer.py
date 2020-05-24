
import pdb
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn


class Trainer (nn.Module):
    def __init__(self, net, loader, loss, optimizer):
        nn.Module.__init__(self)
        self.net = net
        self.loader = loader
        self.loss_func = loss
        self.optimizer = optimizer

    def iscuda(self):
        return next(self.net.parameters()).device != torch.device('cpu')

    def todevice(self, x):
        if isinstance(x, dict):
            return {k: self.todevice(v) for k, v in x.items()}
        if isinstance(x, (tuple, list)):
            return [self.todevice(v) for v in x]

        if self.iscuda():
            return x.contiguous().cuda(non_blocking=True)
        else:
            return x.cpu()

    def __call__(self):

        self.net.train()

        stats = defaultdict(list)

        for it, inputs in enumerate(tqdm(self.loader)):

            inputs = self.todevice(inputs)
            # compute gradient and do model update
            self.optimizer.zero_grad()
            loss, details = self.forward_backward(inputs)
            if torch.isnan(loss):
                raise RuntimeError('Loss is NaN')
            self.optimizer.step()
            for key, val in details.items():
                stats[key].append(val)
        print(" Summary of losses during this epoch:")
        mean = lambda lis: sum(lis) / len(lis)
        for loss_name, vals in stats.items():
            N = 1 + len(vals) // 10
            print(f"  - {loss_name:20}:", end='')
            print(f" {mean(vals[:N]):.3f} --> {mean(vals[-N:]):.3f} (avg: {mean(vals):.3f})")
        return mean(stats['loss'])  # return average loss

    def forward_backward(self, inputs):
        raise NotImplementedError()
