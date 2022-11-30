import numpy as np
import time
import torch
import torch
from utils import *
from torchvision import *
from torch import nn
import numpy as np
from models import GoogLeNet, ResNet18

class APGDAttack():
    def __init__(self, models, n_iter=100, norm='Linf', eps=None,
                 seed=0, loss='ce', rho=.75, verbose=False,
                 device='cuda'):
        self.models = models
        self.n_iter = n_iter
        self.eps = eps
        self.norm = norm
        self.seed = seed
        self.loss = loss
        self.thr_decr = rho
        self.verbose = verbose
        self.device = device

    def check_oscillation(self, x, j, k, y5, k3=0.75):
        t = np.zeros(x.shape[1])
        for counter5 in range(k):
            t += x[j - counter5] > x[j - counter5 - 1]

        return t <= k * k3 * np.ones(t.shape)

    def check_shape(self, x):
        return x if len(x.shape) > 0 else np.expand_dims(x, 0)

    def dlr_loss(self, x, y):
        x_sorted, ind_sorted = x.sort(dim=1)
        ind = (ind_sorted[:, -1] == y).float()

        return -(x[np.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind)) / (
                x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)

    def attack_single_run(self, x_in, y_in, lbda):
        x = x_in.clone() if len(x_in.shape) == 4 else x_in.clone().unsqueeze(0)
        y = y_in.clone() if len(y_in.shape) == 1 else y_in.clone().unsqueeze(0)
        lbda= torch.FloatTensor(lbda).cuda()
        self.n_iter_2, self.n_iter_min, self.size_decr = max(int(0.22 * self.n_iter), 1), max(int(0.06 * self.n_iter),
                                                                                              1), max(
            int(0.03 * self.n_iter), 1)
        if self.verbose:
            print('parameters: ', self.n_iter, self.n_iter_2, self.n_iter_min, self.size_decr)

        if self.norm == 'Linf':
            t = 2 * torch.rand(x.shape).to(self.device).detach() - 1
            x_adv = x.detach() + self.eps * torch.ones([x.shape[0], 1, 1, 1]).to(self.device).detach() * t / (
                t.reshape([t.shape[0], -1]).abs().max(dim=1, keepdim=True)[0].reshape([-1, 1, 1, 1]))
        elif self.norm == 'L2':
            t = torch.randn(x.shape).to(self.device).detach()
            x_adv = x.detach() + self.eps * torch.ones([x.shape[0], 1, 1, 1]).to(self.device).detach() * t / (
                    (t ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)
        x_adv = x_adv.clamp(0., 1.)
        x_best = x_adv.clone()
        x_best_adv = x_adv.clone()
        loss_steps = torch.zeros([self.n_iter, x.shape[0]])
        loss_best_steps = torch.zeros([self.n_iter + 1, x.shape[0]])
        acc_steps = torch.zeros_like(loss_best_steps)
        if self.loss == 'ce':
            criterion_indiv = nn.CrossEntropyLoss(reduction='none')
        elif self.loss == 'dlr':
            criterion_indiv = self.dlr_loss
        else:
            raise ValueError('unknowkn loss')

        x_adv.requires_grad_()
        grad = torch.zeros_like(x)
        acc_per_model = []
        loss_indiv_lbda = 0
        for k, l in enumerate(lbda):
            with torch.enable_grad():
                logits = self.models[k](x_adv)  # 1 forward pass (eot_iter = 1)
                loss_indiv = criterion_indiv(logits, y)
                loss = loss_indiv.sum()
                acc_per_model.append(logits.detach().max(1)[1] == y)
                loss_indiv_lbda += l * loss_indiv
            grad += l * torch.autograd.grad(loss, [x_adv])[0].detach()  # 1 backward pass (eot_iter = 1)
        grad_best = grad.clone()
        acc_per_model = torch.stack(acc_per_model).t()
        acc = torch.matmul(acc_per_model * 1., lbda)
        acc_steps[0] = acc
        loss_best = loss_indiv_lbda.detach().clone()

        step_size = self.eps * torch.ones([x.shape[0], 1, 1, 1]).to(self.device).detach() * torch.Tensor([2.0]).to(
            self.device).detach().reshape([1, 1, 1, 1])
        x_adv_old = x_adv.clone()
        counter = 0
        k = self.n_iter_2 + 0
        u = np.arange(x.shape[0])
        counter3 = 0

        loss_best_last_check = loss_best.clone()
        reduced_last_check = np.zeros(loss_best.shape) == np.zeros(loss_best.shape)
        n_reduced = 0

        for i in range(self.n_iter):
            ### gradient step
            with torch.no_grad():
                x_adv = x_adv.detach()
                grad2 = x_adv - x_adv_old
                x_adv_old = x_adv.clone()

                a = 0.75 if i > 0 else 1.0

                if self.norm == 'Linf':
                    x_adv_1 = x_adv + step_size * torch.sign(grad)
                    x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1, x - self.eps), x + self.eps), 0.0, 1.0)
                    x_adv_1 = torch.clamp(
                        torch.min(torch.max(x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a), x - self.eps),
                                  x + self.eps), 0.0, 1.0)

                elif self.norm == 'L2':
                    x_adv_1 = x_adv + step_size * grad / ((grad ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)
                    x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (
                            ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                        self.eps * torch.ones(x.shape).to(self.device).detach(),
                        ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()), 0.0, 1.0)
                    x_adv_1 = x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a)
                    x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (
                            ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                        self.eps * torch.ones(x.shape).to(self.device).detach(),
                        ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12), 0.0, 1.0)

                x_adv = x_adv_1 + 0.

            ### get gradient

            x_adv.requires_grad_()
            grad = torch.zeros_like(x)
            pred_per_model = []
            loss_indiv_lbda = 0
            for k, l in enumerate(lbda):
                with torch.enable_grad():
                    logits = self.models[k](x_adv)  # 1 forward pass (eot_iter = 1)
                    loss_indiv = criterion_indiv(logits, y)
                    loss = loss_indiv.sum()
                    pred_per_model.append(logits.detach().max(1)[1] == y)
                    loss_indiv_lbda += l * loss_indiv
                grad += l * torch.autograd.grad(loss, [x_adv])[0].detach()  # 1 backward pass (eot_iter = 1)
            grad_best = grad.clone()
            pred_per_model = torch.stack(pred_per_model).t()
            pred = torch.matmul(pred_per_model * 1., lbda)
            x_best_adv[(pred < acc).nonzero().squeeze()] = x_adv[(pred < acc).nonzero().squeeze()] + 0.
            acc_per_model[(pred < acc).nonzero().squeeze()] = pred_per_model[(pred < acc).nonzero().squeeze()]
            acc = torch.min(acc, pred)

            acc_steps[i + 1] = acc + 0
            if self.verbose:
                print('iteration: {} - Best loss: {:.6f}'.format(i, loss_best.sum()))

            ### check step size
            with torch.no_grad():
                y1 = loss_indiv_lbda.detach().clone()
                loss_steps[i] = y1.cpu() + 0
                ind = (y1 > loss_best).nonzero().squeeze()
                x_best[ind] = x_adv[ind].clone()
                grad_best[ind] = grad[ind].clone()
                loss_best[ind] = y1[ind] + 0
                loss_best_steps[i + 1] = loss_best + 0

                counter3 += 1

                if counter3 == k:
                    fl_oscillation = self.check_oscillation(loss_steps.detach().cpu().numpy(), i, k,
                                                            loss_best.detach().cpu().numpy(), k3=self.thr_decr)
                    fl_reduce_no_impr = (~reduced_last_check) * (
                            loss_best_last_check.cpu().numpy() >= loss_best.cpu().numpy())
                    fl_oscillation = ~(~fl_oscillation * ~fl_reduce_no_impr)
                    reduced_last_check = np.copy(fl_oscillation)
                    loss_best_last_check = loss_best.clone()

                    if np.sum(fl_oscillation) > 0:
                        step_size[u[fl_oscillation]] /= 2.0
                        n_reduced = fl_oscillation.astype(float).sum()

                        fl_oscillation = np.where(fl_oscillation)

                        x_adv[fl_oscillation] = x_best[fl_oscillation].clone()
                        grad[fl_oscillation] = grad_best[fl_oscillation].clone()

                    counter3 = 0
                    k = np.maximum(k - self.size_decr, self.n_iter_min)
        return x_best, acc.cpu() * 1., loss_best, x_best_adv, acc_per_model.cpu() * 1.

    def lbr_attack_single_run(self, x_in, y_in, classifier_buffer, lbda_buffer, noise_level=0.0001):
        x = x_in.clone() if len(x_in.shape) == 4 else x_in.clone().unsqueeze(0)
        y = y_in.clone() if len(y_in.shape) == 1 else y_in.clone().unsqueeze(0)
        self.n_iter_2, self.n_iter_min, self.size_decr = max(int(0.22 * self.n_iter), 1), max(int(0.06 * self.n_iter),
                                                                                              1), max(
            int(0.03 * self.n_iter), 1)
        if self.verbose:
            print('parameters: ', self.n_iter, self.n_iter_2, self.n_iter_min, self.size_decr)

        if self.norm == 'Linf':
            t = 2 * torch.rand(x.shape).to(self.device).detach() - 1
            x_adv = x.detach() + self.eps * torch.ones([x.shape[0], 1, 1, 1]).to(self.device).detach() * t / (
                t.reshape([t.shape[0], -1]).abs().max(dim=1, keepdim=True)[0].reshape([-1, 1, 1, 1]))
        elif self.norm == 'L2':
            t = torch.randn(x.shape).to(self.device).detach()
            x_adv = x.detach() + self.eps * torch.ones([x.shape[0], 1, 1, 1]).to(self.device).detach() * t / (
                    (t ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)
        x_adv = x_adv.clamp(0., 1.)
        x_best = x_adv.clone()
        x_best_adv = x_adv.clone()
        loss_steps = torch.zeros([self.n_iter, x.shape[0]])
        loss_best_steps = torch.zeros([self.n_iter + 1, x.shape[0]])
        acc_steps = torch.zeros_like(loss_best_steps)
        if self.loss == 'ce':
            criterion_indiv = nn.CrossEntropyLoss(reduction='none')
        elif self.loss == 'dlr':
            criterion_indiv = self.dlr_loss
        else:
            raise ValueError('unknowkn loss')

        x_adv.requires_grad_()
        grad = torch.zeros_like(x)
        loss_indiv_lbda = 0
        size_buffer = len(lbda_buffer)
        acc = torch.zeros(x.shape[0]).cuda()
        acc_new = torch.zeros(x.shape[0]).cuda()
        for j in range(size_buffer):
            lbda = lbda_buffer[j]
            models = classifier_buffer[j]
            models.eval()
            for k, l in enumerate(lbda):
                with torch.enable_grad():
                    models[k].cuda()
                    logits = models[k](x_adv)  # 1 forward pass (eot_iter = 1)
                    loss_indiv = criterion_indiv(logits, y)
                    loss = loss_indiv.sum()
                    acc.add_(l / size_buffer * (logits.detach().max(1)[1] == y))
                    loss_indiv_lbda += l / size_buffer * loss_indiv
                grad += l / size_buffer * torch.autograd.grad(loss, [x_adv])[0].detach()  # 1 backward pass (eot_iter = 1)
            # for k, _ in enumerate(lbda):
            #     models[k].to('cpu')
        grad_best = grad.clone()
        acc_steps[0] = acc
        loss_best = loss_indiv_lbda.detach().clone()

        step_size = self.eps * torch.ones([x.shape[0], 1, 1, 1]).to(self.device).detach() * torch.Tensor([2.0]).to(
            self.device).detach().reshape([1, 1, 1, 1])
        x_adv_old = x_adv.clone()
        k = self.n_iter_2 + 0
        u = np.arange(x.shape[0])
        counter3 = 0

        loss_best_last_check = loss_best.clone()
        reduced_last_check = np.zeros(loss_best.shape) == np.zeros(loss_best.shape)

        for i in range(self.n_iter):
            ### gradient step
            with torch.no_grad():
                x_adv = x_adv.detach()
                grad2 = x_adv - x_adv_old
                x_adv_old = x_adv.clone()

                a = 0.75 if i > 0 else 1.0

                if self.norm == 'Linf':
                    x_adv_1 = x_adv + step_size * torch.sign(grad) + step_size**0.5 * noise_level * torch.randn_like(x_adv.data)
                    x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1, x - self.eps), x + self.eps), 0.0, 1.0)
                    x_adv_1 = torch.clamp(
                        torch.min(torch.max(x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a), x - self.eps),
                                  x + self.eps), 0.0, 1.0)

                elif self.norm == 'L2':
                    x_adv_1 = x_adv + step_size * grad / ((grad ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) + step_size**0.5 * noise_level * torch.randn_like(x_adv.data)
                    x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (
                            ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                        self.eps * torch.ones(x.shape).to(self.device).detach(),
                        ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()), 0.0, 1.0)
                    x_adv_1 = x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a)
                    x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (
                            ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                        self.eps * torch.ones(x.shape).to(self.device).detach(),
                        ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12), 0.0, 1.0)

                x_adv = x_adv_1 + 0.

            ### get gradient

            x_adv.requires_grad_()
            grad = torch.zeros_like(x)
            loss_indiv_lbda = 0
            acc_new.zero_()
            for j in range(size_buffer):
                lbda = lbda_buffer[j]
                models = classifier_buffer[j]
                models.eval()
                for k, l in enumerate(lbda):
                    with torch.enable_grad():
                        models[k].cuda()
                        logits = models[k](x_adv)  # 1 forward pass (eot_iter = 1)
                        loss_indiv = criterion_indiv(logits, y)
                        loss = loss_indiv.sum()
                        acc_new.add_(l / size_buffer * (logits.detach().max(1)[1] == y))
                        loss_indiv_lbda += l / size_buffer * loss_indiv
                    grad += l / size_buffer * torch.autograd.grad(loss, [x_adv])[0].detach()  # 1 backward pass (eot_iter = 1)
                # for k, _ in enumerate(lbda):
                #     models[k].to('cpu')
            grad_best = grad.clone()
            x_best_adv[(acc_new < acc).nonzero().squeeze()] = x_adv[(acc_new < acc).nonzero().squeeze()] + 0.
            acc = torch.min(acc, acc_new)

            acc_steps[i + 1] = acc + 0
            if self.verbose:
                print('iteration: {} - Best loss: {:.6f}'.format(i, loss_best.sum()))

            ### check step size
            with torch.no_grad():
                # y1 = loss_indiv.detach().clone()
                y1 = loss_indiv_lbda.detach().clone()
                loss_steps[i] = y1.cpu() + 0
                ind = (y1 > loss_best).nonzero().squeeze()
                x_best[ind] = x_adv[ind].clone()
                grad_best[ind] = grad[ind].clone()
                loss_best[ind] = y1[ind] + 0
                loss_best_steps[i + 1] = loss_best + 0

                counter3 += 1

                if counter3 == k:
                    fl_oscillation = self.check_oscillation(loss_steps.detach().cpu().numpy(), i, k,
                                                            loss_best.detach().cpu().numpy(), k3=self.thr_decr)
                    fl_reduce_no_impr = (~reduced_last_check) * (
                            loss_best_last_check.cpu().numpy() >= loss_best.cpu().numpy())
                    fl_oscillation = ~(~fl_oscillation * ~fl_reduce_no_impr)
                    reduced_last_check = np.copy(fl_oscillation)
                    loss_best_last_check = loss_best.clone()

                    if np.sum(fl_oscillation) > 0:
                        step_size[u[fl_oscillation]] /= 2.0

                        fl_oscillation = np.where(fl_oscillation)

                        x_adv[fl_oscillation] = x_best[fl_oscillation].clone()
                        grad[fl_oscillation] = grad_best[fl_oscillation].clone()

                    counter3 = 0
                    k = np.maximum(k - self.size_decr, self.n_iter_min)
        return x_best, acc.cpu() * 1., loss_best, x_best_adv


    def perturb(self, x_in, y_in, lbda, best_loss=False, cheap=True, n_restarts=1, restart_all=False):
        assert self.norm in ['Linf', 'L2']
        x = x_in.clone() if len(x_in.shape) == 4 else x_in.clone().unsqueeze(0)
        y = y_in.clone() if len(y_in.shape) == 1 else y_in.clone().unsqueeze(0)
        lbda_torch = torch.FloatTensor(lbda).cpu()

        adv = x.clone()
        acc_per_model = []
        for k, l in enumerate(lbda):
            acc_per_model.append(self.models[k](x).max(1)[1] == y)
        acc_per_model = torch.stack(acc_per_model).t().cpu() * 1.
        acc = torch.matmul(acc_per_model, lbda_torch)
        loss = -1e10 * torch.ones_like(acc).float()
        if self.verbose:
            print('-------------------------- running {}-attack with epsilon {:.4f} --------------------------'.format(
                self.norm, self.eps))
            print('initial accuracy: {:.2%}'.format(acc.float().mean()))
        startt = time.time()

        if not best_loss:
            torch.random.manual_seed(self.seed)
            torch.cuda.random.manual_seed(self.seed)

            if not cheap:
                raise ValueError('not implemented yet')

            else:
                all_advs = []
                all_accs = []
                all_accs_per_model = []
                for counter in range(n_restarts):
                    if restart_all:
                        best_curr, acc_curr, loss_curr, adv_curr, acc_curr_per_model = self.attack_single_run(x, y,
                                                                                                              lbda)
                        all_accs.append(acc_curr)
                        all_advs.append(adv)
                        all_accs_per_model.append(acc_curr_per_model)

                    else:
                        ind_to_fool = (acc != 0.).nonzero().squeeze()
                        if len(ind_to_fool.shape) == 0: ind_to_fool = ind_to_fool.unsqueeze(0)
                        if ind_to_fool.numel() != 0:
                            x_to_fool, y_to_fool = x[ind_to_fool].clone(), y[ind_to_fool].clone()
                            best_curr, acc_curr, loss_curr, adv_curr, acc_curr_per_model = self.attack_single_run(
                                x_to_fool, y_to_fool, lbda)
                            # ind_curr = (acc_curr == 0.).nonzero().squeeze()
                            ind_better = (acc_curr < acc[ind_to_fool]).nonzero().squeeze()
                            acc[ind_to_fool[ind_better]] = acc_curr[ind_better]
                            acc_per_model[ind_to_fool[ind_better]] = acc_curr_per_model[ind_better]

                            adv[ind_to_fool[ind_better]] = adv_curr[ind_better].clone()
                            if self.verbose:
                                print('restart {} - robust accuracy: {:.2%} - cum. time: {:.1f} s'.format(
                                    counter, acc.float().mean(), time.time() - startt))
            if restart_all:
                acc = torch.stack(all_accs)
                adv = torch.stack(all_advs)
                acc_per_model = torch.stack(all_accs_per_model)
            return acc, adv, acc_per_model

        else:
            adv_best = x.detach().clone()
            loss_best = torch.ones([x.shape[0]]).to(self.device) * (-float('inf'))
            for counter in range(n_restarts):
                best_curr, _, loss_curr, _ = self.attack_single_run(x, y, lbda)
                ind_curr = (loss_curr > loss_best).nonzero().squeeze()
                adv_best[ind_curr] = best_curr[ind_curr] + 0.
                loss_best[ind_curr] = loss_curr[ind_curr] + 0.

                if self.verbose:
                    print('restart {} - loss: {:.5f}'.format(counter, loss_best.sum()))

            return loss_best, adv_best, 0.

    def lbr_perturb(self, x_in, y_in, classifier_buffer, lbda_buffer, best_loss=False, cheap=True, n_restarts=1, restart_all=False, noise_level=0.0001):
        assert self.norm in ['Linf', 'L2']
        x = x_in.clone() if len(x_in.shape) == 4 else x_in.clone().unsqueeze(0)
        y = y_in.clone() if len(y_in.shape) == 1 else y_in.clone().unsqueeze(0)

        adv = x.clone()
        size_buffer = len(lbda_buffer)
        acc = torch.zeros(x.shape[0])
        for j in range(size_buffer):
            lbda = lbda_buffer[j]
            models = classifier_buffer[j]
            models.eval()
            for k, l in enumerate(lbda):
                with torch.enable_grad():
                    models[k].cuda()
                    logits = models[k](x)  # 1 forward pass (eot_iter = 1)
                    acc.add_(l / size_buffer * (logits.detach().max(1)[1] == y).cpu())
            # for k, _ in enumerate(lbda):
            #     models[k].to('cpu')
        if self.verbose:
            print('-------------------------- running {}-attack with epsilon {:.4f} --------------------------'.format(
                self.norm, self.eps))
            print('initial accuracy: {:.2%}'.format(acc.float().mean()))
        startt = time.time()

        if not best_loss:
            torch.random.manual_seed(self.seed)
            torch.cuda.random.manual_seed(self.seed)

            if not cheap:
                raise ValueError('not implemented yet')

            else:
                all_advs = []
                all_accs = []
                for counter in range(n_restarts):
                    if restart_all:
                        best_curr, acc_curr, loss_curr, adv_curr = self.lbr_attack_single_run(x, y, classifier_buffer, lbda_buffer, noise_level=noise_level)
                        all_accs.append(acc_curr)
                        all_advs.append(adv)

                    else:
                        ind_to_fool = (acc != 0.).nonzero().squeeze()
                        if len(ind_to_fool.shape) == 0: ind_to_fool = ind_to_fool.unsqueeze(0)
                        if ind_to_fool.numel() != 0:
                            x_to_fool, y_to_fool = x[ind_to_fool].clone(), y[ind_to_fool].clone()
                            best_curr, acc_curr, loss_curr, adv_curr = self.lbr_attack_single_run(
                                x_to_fool, y_to_fool, classifier_buffer, lbda_buffer, noise_level=noise_level)
                            # ind_curr = (acc_curr == 0.).nonzero().squeeze()
                            ind_better = (acc_curr < acc[ind_to_fool]).nonzero().squeeze()
                            acc[ind_to_fool[ind_better]] = acc_curr[ind_better]

                            adv[ind_to_fool[ind_better]] = adv_curr[ind_better].clone()
                            if self.verbose:
                                print('restart {} - robust accuracy: {:.2%} - cum. time: {:.1f} s'.format(
                                    counter, acc.float().mean(), time.time() - startt))
            if restart_all:
                acc = torch.stack(all_accs)
                adv = torch.stack(all_advs)
            return adv

        else:
            adv_best = x.detach().clone()
            loss_best = torch.ones([x.shape[0]]).to(self.device) * (-float('inf'))
            for counter in range(n_restarts):
                best_curr, _, loss_curr, _ = self.lbr_attack_single_run(x, y, classifier_buffer, lbda_buffer, noise_level=noise_level)
                ind_curr = (loss_curr > loss_best).nonzero().squeeze()
                adv_best[ind_curr] = best_curr[ind_curr] + 0.
                loss_best[ind_curr] = loss_curr[ind_curr] + 0.

                if self.verbose:
                    print('restart {} - loss: {:.5f}'.format(counter, loss_best.sum()))

            return adv_best


if __name__ == "__main__":
    data = torch.load("model_GoogLeNet_adv_False/34999977/checkpoint.pth")
    model1 = GoogLeNet()
    model1 = NormalizedModel(model1, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    model1 = torch.nn.DataParallel(model1, device_ids=[0, 1])
    model1.load_state_dict(data["model"])
    model1.cuda()
    model1.eval()

    data = torch.load("model_ResNet18_adv_False/34999976/checkpoint.pth")
    model2 = ResNet18()
    model2 = NormalizedModel(model2, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    model2 = torch.nn.DataParallel(model2, device_ids=[0, 1])
    model2.load_state_dict(data["model"])
    model2.cuda()
    model2.eval()


    class MixedModel(nn.Module):
        def __init__(self, models):
            super(MixedModel, self).__init__()
            self.models = torch.nn.ModuleList(models)
            self.lbda = np.ones(len(self.models)) / len(self.models)

        def forward(self, x):
            ii = np.random.choice(np.arange(len(self.lbda)), p=self.lbda)
            return self.models[ii](x)


    mixedmodel = MixedModel([model1, model2])
    transform = transforms.Compose([transforms.ToTensor()])
    test_set = datasets.CIFAR10(
        "data/cifar10", train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=100, shuffle=True, num_workers=2)

    adversary_perso = APGDAttack([model1, model2], n_iter=10, norm='Linf', eps=0.03,
                                 seed=0, loss='ce', rho=.75, verbose=True)

    i = 0
    for x, y in test_loader:
        x, y = x.cuda(), y.cuda()
        if i == 1:
            break
        # u = adversary.perturb(x, y)
        u = adversary_perso.perturb(x, y, np.array([1 / 2, 1 / 2]), n_restarts=1, restart_all=True)
        print(u[0])
        print(u[2])
        i += 1
