import torch
import numpy as np
from torch import nn


def perturb_iterative(xvar, yvar, predict, lbda, nb_iter, eps, eps_iter, loss_fn,
                      delta_init=None, minimize=False, ord=np.inf,
                      clip_min=0.0, clip_max=1.0):
    """
    Iteratively maximize the loss over the input. It is a shared method for
    iterative attacks including IterativeGradientSign, LinfPGD, etc.
    :param xvar: input data.
    :param yvar: input labels.
    :param predict: forward pass function.
    :param nb_iter: number of iterations.
    :param eps: maximum distortion.
    :param eps_iter: attack step size.
    :param loss_fn: loss function.
    :param delta_init: (optional) tensor contains the random initialization.
    :param minimize: (optional bool) whether to minimize or maximize the loss.
    :param ord: (optional) the order of maximum distortion (inf or 2).
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param l1_sparsity: sparsity value for L1 projection.
                  - if None, then perform regular L1 projection.
                  - if float value, then perform sparse L1 descent from
                    Algorithm 1 in https://arxiv.org/pdf/1904.13000v1.pdf
    :return: tensor containing the perturbed input.
    """
    if delta_init is not None:
        delta = delta_init
    else:
        delta = torch.zeros_like(xvar)

    delta.requires_grad_()
    for ii in range(nb_iter):
        loss = 0
        with torch.enable_grad():
            for k, l in enumerate(lbda):
                outputs = predict[k](xvar + delta)
                loss = loss + l * loss_fn(outputs, yvar)  # expected loss value
            if minimize:
                loss = -loss
            loss.backward()
            grad = delta.grad
        if ord == np.inf:
            grad_sign = grad.data.sign()
            delta.data = delta.data + eps_iter * grad_sign
            delta.data = torch.clamp(delta.data, min=-eps, max=eps)
            delta.data = torch.clamp(xvar.data + delta.data, min=clip_min, max=clip_max) - xvar.data  # the perturbed input lies in [-1, 1]

        delta.grad.data.zero_()

    x_adv = torch.clamp(xvar + delta, clip_min, clip_max)
    return x_adv


def lbr_perturb_iterative(xvar, yvar, classifier_buffer, lbda_buffer, nb_iter, 
                          eps, eps_iter, loss_fn, delta_init=None,
                          minimize=False, ord=np.inf,
                          clip_min=0.0, clip_max=1.0,
                          noise_level=0.0001):
    """
    Iteratively maximize the loss over the input. It is a shared method for
    iterative attacks including IterativeGradientSign, LinfPGD, etc.
    :param xvar: input data.
    :param yvar: input labels.
    :param classifier_buffer: the buffer of mxiture models
    :param lbda_buffer: the buffer of weights of the mxiture models
    :param nb_iter: number of iterations.
    :param eps: maximum distortion.
    :param eps_iter: attack step size.
    :param loss_fn: loss function.
    :param delta_init: (optional) tensor contains the random initialization.
    :param minimize: (optional bool) whether to minimize or maximize the loss.
    :param ord: (optional) the order of maximum distortion (inf or 2).
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param l1_sparsity: sparsity value for L1 projection.
                  - if None, then perform regular L1 projection.
                  - if float value, then perform sparse L1 descent from
                    Algorithm 1 in https://arxiv.org/pdf/1904.13000v1.pdf
    :param noise_level: the Gaussian noise level
    :return: tensor containing the perturbed input.
    """
    if delta_init is not None:
        delta = delta_init
    else:
        delta = torch.zeros_like(xvar)
    size_buffer = len(lbda_buffer)

    delta.requires_grad_()
    grad = torch.zeros_like(xvar)
    for ii in range(nb_iter):
        grad.zero_()
        for j in range(size_buffer):
            lbda = lbda_buffer[j]
            models = classifier_buffer[j]
            models.eval()
            for k, l in enumerate(lbda):
                with torch.enable_grad():
                    models[k].cuda()
                    outputs = models[k](xvar + delta)
                    loss = loss_fn(outputs, yvar)  # expected loss value
                    if minimize:
                        loss = -loss
                grad += l / size_buffer * torch.autograd.grad(loss, [delta])[0].detach()  # 1 backward pass (eot_iter = 1)
            # for k, _ in enumerate(lbda):
            #     models[k].to('cpu')
        if ord == np.inf:
            grad_sign = grad.data.sign()
            delta.data = delta.data + eps_iter * grad_sign + eps_iter**0.5 * noise_level * torch.randn_like(delta.data)
            delta.data = torch.clamp(delta.data, min=-eps, max=eps)
            delta.data = torch.clamp(xvar.data + delta.data, min=clip_min, max=clip_max) - xvar.data  # the perturbed input lies in [-1, 1]

        # delta.grad.data.zero_()

    x_adv = torch.clamp(xvar + delta, clip_min, clip_max)
    return x_adv



class PGDAttack:
    """
    The projected gradient descent attack (Madry et al, 2017).
    The attack performs nb_iter steps of size eps_iter, while always staying
    within eps from the initial point.
    Paper: https://arxiv.org/pdf/1706.06083.pdf
    :param predict: forward pass function.
    :param loss_fn: loss function.
    :param eps: maximum distortion.
    :param nb_iter: number of iterations.
    :param eps_iter: attack step size.
    :param rand_init: (optional bool) random initialization.
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param ord: (optional) the order of maximum distortion (inf or 2).
    :param targeted: if the attack is targeted.
    """

    def __init__(self, predict, loss_fn=None, eps=0.3, nb_iter=40,
                 eps_iter=0.01, rand_init=True, clip_min=0., clip_max=1.,
                 ord=np.inf, targeted=False):
        """
        Create an instance of the PGDAttack.
        """
        self.predict = predict
        self.eps = eps
        self.nb_iter = nb_iter
        self.eps_iter = eps_iter
        self.rand_init = rand_init
        self.ord = ord
        self.targeted = targeted
        self.loss_fn = nn.CrossEntropyLoss(reduction="sum")
        self.clip_min = clip_min
        self.clip_max = clip_max

        # assert is_float_or_torch_tensor(self.eps_iter)
        # assert is_float_or_torch_tensor(self.eps)

    def perturb(self, x, y, lbda, n_restarts=1, restart_all=None):
        """
        Given examples (x, y), returns their adversarial counterparts with
        an attack length of eps.
        :param x: input tensor.
        :param y: label tensor.
                  - if None and self.targeted=False, compute y as predicted
                    labels.
                  - if self.targeted=True, then y must be the targeted labels.
        :return: tensor containing perturbed inputs.
        """
        # x, y = self._verify_and_process_inputs(x, y)


        if n_restarts == 1:
            delta = torch.zeros_like(x)
            delta = nn.Parameter(delta)  # the perturbed input as a variable
            if self.rand_init:
                delta.uniform_(-1, 1)
                delta = self.eps * delta.data
                delta.data = torch.clamp(x + delta.data, min=self.clip_min, max=self.clip_max) - x
                # linf norm ball constraint
            rval = perturb_iterative(
                x, y, self.predict, lbda, nb_iter=self.nb_iter,
                eps=self.eps, eps_iter=self.eps_iter,
                loss_fn=self.loss_fn, minimize=self.targeted,
                clip_max=self.clip_max, delta_init=delta,
            )
            # rval is the final perturbed input
            lbda_torch = torch.FloatTensor(lbda).cpu()
            acc_per_model = []
            for k, l in enumerate(lbda):
                acc_per_model.append(self.predict[k](rval).max(1)[1] == y)
            acc_per_model = torch.stack(acc_per_model).t().cpu() * 1.
            acc = torch.matmul(acc_per_model, lbda_torch)

            return acc, rval.data, acc_per_model
        else:
            acc_per_model = []
            adv_data = []
            acc = []
            for _ in range(n_restarts):
                delta = torch.zeros_like(x)
                delta = nn.Parameter(delta)
                delta.uniform_(-1, 1)
                delta = self.eps * delta.data
                delta.data = torch.clamp(x + delta.data, min=self.clip_min, max=self.clip_max) - x
                rval = perturb_iterative(
                    x, y, self.predict, lbda, nb_iter=self.nb_iter,
                    eps=self.eps, eps_iter=self.eps_iter,
                    loss_fn=self.loss_fn, minimize=self.targeted,
                    clip_max=self.clip_max, delta_init=delta,
                )
                lbda_torch = torch.FloatTensor(lbda).cpu()
                acc_curr_per_model = []
                for k, l in enumerate(lbda):
                    acc_curr_per_model.append(self.predict[k](rval).max(1)[1] == y)
                acc_curr_per_model = torch.stack(acc_curr_per_model).t().cpu() * 1.

                acc_curr = torch.matmul(acc_curr_per_model, lbda_torch)
                acc_per_model.append(acc_curr_per_model)
                adv_data.append(rval)
                acc.append(acc_curr)
            acc_per_model = torch.stack(acc_per_model)
            acc = torch.stack(acc)
            adv_data = torch.stack(adv_data)

            return acc, adv_data, acc_per_model

    def lbr_perturb(self, x, y, classifier_buffer, lbda_buffer, n_restarts=1, restart_all=None, noise_level=0.0001):
        """
        Given examples (x, y), returns their adversarial counterparts with
        an attack length of eps.
        :param x: input tensor.
        :param y: label tensor.
                  - if None and self.targeted=False, compute y as predicted
                    labels.
                  - if self.targeted=True, then y must be the targeted labels.
        :classifier_buffer: the buffer contanining previous mixture models
        :lbda_buffer: the buffer contanining the mixing rates in previous mixture models
        :return: tensor containing perturbed inputs.
        """

        if n_restarts == 1:
            delta = torch.zeros_like(x)
            delta = nn.Parameter(delta)  # the perturbed input as a variable
            if self.rand_init:
                delta.uniform_(-1, 1)
                delta = self.eps * delta.data
                delta.data = torch.clamp(x + delta.data, min=self.clip_min, max=self.clip_max) - x
                # linf norm ball constraint
            rval = lbr_perturb_iterative(
                x, y, classifier_buffer, lbda_buffer,
                nb_iter=self.nb_iter,
                eps=self.eps, eps_iter=self.eps_iter,
                loss_fn=self.loss_fn, minimize=self.targeted,
                clip_max=self.clip_max, delta_init=delta,
                noise_level=noise_level
            )
            # rval is the final perturbed input

            return rval.data
        else:
            adv_data = []
            for _ in range(n_restarts):
                delta = torch.zeros_like(x)
                delta = nn.Parameter(delta)
                delta.uniform_(-1, 1)
                delta = self.eps * delta.data
                delta.data = torch.clamp(x + delta.data, min=self.clip_min, max=self.clip_max) - x
                rval = lbr_perturb_iterative(
                    x, y, classifier_buffer, lbda_buffer,
                    nb_iter=self.nb_iter,
                    eps=self.eps, eps_iter=self.eps_iter,
                    loss_fn=self.loss_fn, minimize=self.targeted,
                    clip_max=self.clip_max, delta_init=delta,
                    noise_level=noise_level
                )
                adv_data.append(rval)

            # adv_data = torch.stack(adv_data)

            return adv_data



class LinfPGDAttack(PGDAttack):  # not used
    """
    PGD Attack with order=Linf
    :param predict: forward pass function.
    :param loss_fn: loss function.
    :param eps: maximum distortion.
    :param nb_iter: number of iterations.
    :param eps_iter: attack step size.
    :param rand_init: (optional bool) random initialization.
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param targeted: if the attack is targeted.
    """

    def __init__(
            self, predict, loss_fn=None, eps=0.3, nb_iter=40,
            eps_iter=0.01, rand_init=True, clip_min=0., clip_max=1.,
            targeted=False):
        ord = np.inf
        super(LinfPGDAttack, self).__init__(
            predict=predict, loss_fn=loss_fn, eps=eps, nb_iter=nb_iter,
            eps_iter=eps_iter, rand_init=rand_init, clip_min=clip_min,
            clip_max=clip_max, targeted=targeted,
            ord=ord)
