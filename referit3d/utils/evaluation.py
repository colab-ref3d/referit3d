import numpy as np
import torch


class _AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class DistAverageMeter(_AverageMeter):
    def __init__(self, dist_mgr):
        super().__init__()
        self.dist_mgr = dist_mgr

    def update(self, val, n=1):
        is_tensor = True
        if not isinstance(val, torch.Tensor):
            val = torch.tensor(val)
            is_tensor = False
        self.dist_mgr.allreduce_mean(val)
        if not is_tensor:
            val = val.item()
        return super().update(val, n)
