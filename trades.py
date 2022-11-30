import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np


def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()


class TRADESAttack:
    def __init__(self, predict, eps=0.3, nb_iter=40, eps_iter=0.01, clip_min=0., clip_max=1.):
        """
        Create an instance of the PGDAttack.
        """
        self.predict = predict
        self.eps = eps
        self.nb_iter = nb_iter
        self.eps_iter = eps_iter
        self.clip_min = clip_min
        self.clip_max = clip_max

    def perturb(self, x_natural, y, lbda,  n_restarts=1, restart_all=False):
        # define KL-loss
        criterion_kl = nn.KLDivLoss(reduction='sum')
        batch_size = len(x_natural)
        # generate adversarial example
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()

        for _ in range(self.nb_iter):
            x_adv.requires_grad_()
            loss_kl = 0
            with torch.enable_grad():
                for k, l in enumerate(lbda):
                    loss_kl = loss_kl + l * criterion_kl(F.log_softmax(self.predict[k](x_adv), dim=1),
                                                         F.softmax(self.predict[k](x_natural), dim=1))

            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + self.eps_iter * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - self.eps), x_natural + self.eps)
            x_adv = torch.clamp(x_adv, self.clip_min, self.clip_max)
        return np.array([0]), x_adv, 0

    def lbr_perturb(self, x_natural, y, classifier_buffer, lbda_buffer,  n_restarts=1, restart_all=False, noise_level=0.0001):
        # define KL-loss
        criterion_kl = nn.KLDivLoss(reduction='sum')
        batch_size = len(x_natural)
        # generate adversarial example
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()

        size_buffer = len(lbda_buffer)
        grad = torch.zeros_like(x_natural)
        for _ in range(self.nb_iter):
            x_adv.requires_grad_()
            loss_kl = 0
            grad.zero_()
            for j in range(size_buffer):
                lbda = lbda_buffer[j]
                models = classifier_buffer[j]
                models.eval()
                with torch.enable_grad():
                    for k, l in enumerate(lbda):
                            loss_kl = loss_kl + l * criterion_kl(F.log_softmax(models[k](x_adv), dim=1),
                                                                 F.softmax(self.predict[k](x_natural), dim=1))    
                grad += l / size_buffer * torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + self.eps_iter * torch.sign(grad.detach())  + self.eps_iter**0.5 * noise_level * torch.randn_like(x_natural.data)
            x_adv = torch.min(torch.max(x_adv, x_natural - self.eps), x_natural + self.eps)
            x_adv = torch.clamp(x_adv, self.clip_min, self.clip_max)
        return x_adv



class TradesLoss(nn.Module):

    def __init__(self, models, beta):
        super(TradesLoss, self).__init__()
        self.models = models
        self.beta = beta

    def forward(self, inputs, inputs_adv, i):
        return self.models[i](inputs), self.models[i](inputs_adv)
