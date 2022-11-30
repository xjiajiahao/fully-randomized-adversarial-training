import numpy as np
from scipy.special import softmax, logsumexp
from torch import nn
import torch


def projection_simplex_sort(v, z=1):
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)

    def forward(self, x):
        return (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[None, :, None, None]


class NormalizedModel(nn.Module):
    def __init__(self, model, mean, std):
        super(NormalizedModel, self).__init__()
        self.model = model
        self.normalize = Normalize(mean, std)

    def forward(self, x):
        return self.model(self.normalize(x))
