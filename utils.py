from typing import Iterable

import numpy as np
import torch
from timeit import default_timer as timer
from scipy.stats import norm

def seed_everything(seed):
    import numpy as np
    import random
    import torch
    import os

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def build_logger(log_dir, file_name):
    import logging
    import os.path as osp

    logger = logging.getLogger(log_dir)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s -- %(message)s")
    hdlr = logging.FileHandler(osp.join(log_dir, file_name))
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.propagate = False
    return logger


class Timer(object):
    def __init__(self):
        self._cur = timer()
    
    def tiktok(self):
        cur = timer()
        elapsed = cur - self._cur
        self._cur = cur
        return elapsed
    
    def reset(self):
        self._cur = timer()


class AverageMeter(object):
    def __init__(self) -> None:
        self._total = 0
        self._count = 0
    
    @property
    def value(self):
        return self._total / self._count
    
    def add(self, values, count=1):
        if isinstance(values, np.ndarray) or isinstance(values, torch.Tensor):
            values = values.flatten().tolist()

        if not isinstance(values, Iterable):
            values = [values] * count

        self._count += len(values)
        self._total += sum(values)
    
    def reset(self):
        self._total = 0
        self._count = 0


class GaussDiscreter(object):
    def __init__(self, min, max, width, sigma=1.0) -> None:
        assert (max - min) % width == 0
        self.sigma = sigma
        self.width = width
        self.centers = np.arange(min, max, width) + width / 2
        self.left = np.arange(min, max, width)
        self.right = self.left + width
    
    def contin2dis(self, x):
        if isinstance(x, torch.Tensor):
            x = x.flatten().tolist()
        x = np.array(x).reshape(-1, 1)
        cdf1 = norm.cdf(self.left, loc=x, scale=self.sigma)
        cdf2 = norm.cdf(self.right, loc=x, scale=self.sigma)
        return (cdf2 - cdf1).astype(np.float32)
    
    def dis2contin(self, x):
        if isinstance(x, torch.Tensor):
            centers = torch.tensor(self.centers, device=x.device, dtype=x.dtype)
            return x.detach() @ centers
        return x @ self.centers


if __name__ == '__main__':
    discreter = GaussDiscreter(5, 105, 2, 1)
    labels = discreter.contin2dis([42.3, 9])
    print(labels.max(axis=1))
    print(discreter.dis2contin(labels))
