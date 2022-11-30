import os
from config import *
from torch import optim, nn
import attr
import torch.backends.cudnn as cudnn
from typing import List as List
from typing import Tuple as Tuple
from typing import Any as Any
import time
from auto_attack import *
from scipy.special import softmax
from torch import distributed
from models import *
from pgd_attack import *
from trades import *
from datetime import datetime

dict_models = {
    "ResNet18": ResNet18,
    "PreActResNet18": PreActResNet18,
    "GoogLeNet": GoogLeNet,
    "SimpleDLA": SimpleDLA,
    "DenseNet121": DenseNet121,
    "WideResNet28x10": WideResNet28x10,
    "WideResNet34x20": WideResNet34x20
}


@attr.s(auto_attribs=True)  # auto attribute
class TrainerState:
    """
    Contains the state of the Trainer.
    It can be saved to checkpoint the training and loaded to resume it.
    """

    epoch: int
    models: nn.ModuleList
    optimizer: List[optim.Optimizer]
    lr_scheduler: List[torch.optim.lr_scheduler._LRScheduler]
    lbda: np.array  # the mixture weight vector
    best_acc_adv: float

    def save(self, filename: str) -> None:
        data = attr.asdict(self)
        # store only the state dict
        data[f"models"] = self.models.state_dict()
        data[f"optimizer"] = []
        data[f"lr_scheduler"] = []

        for opt in self.optimizer:
            data[f"optimizer"].append(opt.state_dict())

        for lr_s in self.lr_scheduler:
            data[f"lr_scheduler"].append(lr_s.state_dict())
        data[f"lbda"] = self.lbda
        data[f"epoch"] = self.epoch
        data["best_acc_adv"] = self.best_acc_adv
        torch.save(data, filename)

    @classmethod
    def load(cls, filename: str, default: "TrainerState", gpu: int) -> "TrainerState":
        data = torch.load(filename, map_location=lambda storage, loc: storage.cuda(gpu))
        # We need this default to load the state dict
        models = default.models
        models.load_state_dict(data[f"models"])
        data[f"models"] = models
        for k, opt in enumerate(default.optimizer):
            opt.load_state_dict(data[f"optimizer"][k])
            data[f"optimizer"][k] = opt

        for k, lr_s in enumerate(default.lr_scheduler):
            lr_s.load_state_dict(data[f"lr_scheduler"][k])
            data[f"lr_scheduler"][k] = lr_s
        return cls(**data)  # dict as arguments


class Trainer:
    def __init__(self, train_cfg: TrainerConfig, cluster_cfg: ClusterConfig, job_env: JobEnvironment):
        self._train_cfg = train_cfg
        self._cluster_cfg = cluster_cfg
        self.n_classifiers = len(self._train_cfg.models)
        self.job_env = job_env

    def __call__(self):
        cudnn.benchmark = True  # accelerates the CNN
        # setup process group
        job_env = self.job_env
        torch.cuda.set_device(job_env.local_rank)  # local_rank: gpu id in one machine
        distributed.init_process_group(
            backend=self._cluster_cfg.dist_backend,
            init_method=self._cluster_cfg.dist_url,
            world_size=job_env.num_tasks,  # num_tasks: the number of processes in total
            rank=job_env.global_rank,  # global_rank: the process id
        )
        # initialize torch.distributed env

        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")

        print("Create data loaders", flush=True)

        transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor()])
        if self._train_cfg.dataset == "CIFAR10":
            # the CIFAR-10 training data set
            train_set = datasets.CIFAR10(self._train_cfg.data_root, train=True,
                                         download=True, transform=transform)
            num_classes = 10
        else:
            # the CIFAR-100 training data set
            train_set = datasets.CIFAR100(self._train_cfg.data_root, train=True,
                                          download=True, transform=transform)
            num_classes = 100

        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_set, num_replicas=job_env.num_tasks, rank=job_env.global_rank
        )  # num_replicas: Number of processes participating in distributed training. By default
        # distribute a minibatch to the #num_tasks GPUs

        self._train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=int(self._train_cfg.batch_per_gpu),
            num_workers=int(80 * self._cluster_cfg.num_nodes / self._cluster_cfg.num_gpus_per_node),
            sampler=train_sampler
        )

        transform = transforms.Compose([transforms.ToTensor()])
        # transform for the test loader (w/o data augmentation)

        if self._train_cfg.dataset == "CIFAR10":
            test_set = datasets.CIFAR10(self._train_cfg.data_root, train=False,
                                        download=True, transform=transform)
        else:
            test_set = datasets.CIFAR100(self._train_cfg.data_root, train=False,
                                         download=True, transform=transform)

        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_set, num_replicas=job_env.num_tasks, rank=job_env.global_rank
        )

        self._test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=int(self._train_cfg.batch_per_gpu),
            num_workers=int(80 * self._cluster_cfg.num_nodes / self._cluster_cfg.num_gpus_per_node),
            sampler=test_sampler,
        )
        print(f"Total batch_size: {self._train_cfg.batch_per_gpu * job_env.num_tasks}", flush=True)  # the large batch

        print("Create distributed model", flush=True)

        models = nn.ModuleList([NormalizedModel(dict_models[model](num_classes=num_classes),
                                                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) for model in
                                self._train_cfg.models])
        # normalize each image to the range [-1, 1], not normalizing in transforms because the attacker wants the original image
        # @NOTE The output of torchvision datasets are PILImage images of range [0, 1]
        for i in range(self.n_classifiers):
            models[i].cuda(job_env.local_rank)
            models[i].train()  # training mode
            models[i] = torch.nn.parallel.DistributedDataParallel(models[i], device_ids=[job_env.local_rank],
                                                                  output_device=job_env.local_rank, broadcast_buffers=False)
            # parallelizing over machines and devices
        models = nn.ModuleList(models)

        optimizer = [optim.SGD(models[k].parameters(),
                            #    lr=max(0.1 * self._train_cfg.batch_per_gpu *
                            #           job_env.num_tasks / 256, 0.1),  # lr=0.4
                               lr=0.4,
                               momentum=0.9, weight_decay=5e-4) for k in range(self.n_classifiers)]

        lr_scheduler = [
            torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer[k], 'max', factor=0.1, threshold=0.001, threshold_mode='abs', min_lr=0.0001)
            for k in range(self.n_classifiers)]

        lbda = np.ones(self.n_classifiers) / self.n_classifiers
        # initialize as equal weights

        self._state = TrainerState(epoch=0, models=models, optimizer=optimizer,
                                   lr_scheduler=lr_scheduler, lbda=lbda, best_acc_adv=-1)
        # the initial state
        self.ckpt_run = self._train_cfg.ckpt_run
        checkpoint_fn = os.path.join(self._train_cfg.data_root, self.ckpt_run, "checkpoint.pth")
        # load checkpoint
        if os.path.isfile(checkpoint_fn):
            print(f"Load existing checkpoint from {checkpoint_fn}", flush=True)
            self._state = TrainerState.load(
                checkpoint_fn, default=self._state, gpu=job_env.local_rank)
            # restore state
        criterion = nn.CrossEntropyLoss(reduction='mean').cuda(job_env.local_rank)
        criterion_kl = nn.KLDivLoss(reduction='sum').cuda(job_env.local_rank)
        # KL divergence loss

        # Start from the loaded epoch
        start_epoch = self._state.epoch
        if self._train_cfg.attack == "APGD":
            adversary = APGDAttack(self._state.models, n_iter=self._train_cfg.iter_attack, norm='Linf',
                                   eps=self._train_cfg.eps_attack, loss=self._train_cfg.loss, rho=.75, verbose=False)
            adversary_ent = APGDAttack(self._state.models, n_iter=3, norm='Linf',
                                       eps=self._train_cfg.eps_attack, loss=self._train_cfg.loss, rho=.75,
                                       verbose=False)
            # adversary is the normal attack, adversary_ent is the regularized one

        elif self._train_cfg.attack == "PGD":
            eps_iter = 2. * self._train_cfg.eps_attack / self._train_cfg.iter_attack
            adversary = PGDAttack(self._state.models, eps=self._train_cfg.eps_attack,
                                  nb_iter=self._train_cfg.iter_attack, eps_iter=eps_iter, rand_init=True,
                                  clip_min=0., clip_max=1., ord=np.inf, targeted=False)
            eps_iter = 2. * self._train_cfg.eps_attack / 3.
            adversary_ent = PGDAttack(self._state.models, eps=self._train_cfg.eps_attack,
                                      nb_iter=3, eps_iter=eps_iter, rand_init=True,
                                      clip_min=0., clip_max=1., ord=np.inf, targeted=False)
        elif self._train_cfg.attack == "TRADES":
            eps_iter = 2. * self._train_cfg.eps_attack / self._train_cfg.iter_attack
            adversary = TRADESAttack(self._state.models, eps=self._train_cfg.eps_attack,
                                     nb_iter=self._train_cfg.iter_attack, eps_iter=eps_iter,
                                     clip_min=0., clip_max=1.)
            eps_iter = 2. * self._train_cfg.eps_attack / 3.
            adversary_ent = PGDAttack(self._state.models, eps=self._train_cfg.eps_attack,
                                      nb_iter=3, eps_iter=eps_iter, rand_init=True,
                                      clip_min=0., clip_max=1., ord=np.inf, targeted=False)
            trades_loss = TradesLoss(self._state.models, 6.)

        if job_env.global_rank == 0:
            print(f"Attack characteristics : epsilon = {self._train_cfg.eps_attack}, iters = {self._train_cfg.iter_attack},"
                  f"n_restarts = {self._train_cfg.n_restarts}")
            print(f"Regularization : {self._train_cfg.reg}")
            print(f"Models: {self._train_cfg.models}")
            print(f"Number of epochs per model:{int(self._train_cfg.epochs / self.n_classifiers)}")
        print_freq = 10

        n_update_models = int(50 * 1024 / (self._train_cfg.batch_per_gpu * job_env.num_tasks))  # number of updates of model weights: 50
        if self.n_classifiers > 1:
            m = 25  # update lbda with m GD steps
            period = n_update_models * self.n_classifiers + 1  # update lbda once in each period

        else:
            m = 0
            period = n_update_models

        t = 0
        acc_adv = -1
        order = torch.IntTensor([0] * self.n_classifiers).cuda(job_env.local_rank,
                                                               non_blocking=True)
        delay_lr_scheduler = [0] * self.n_classifiers
        for epoch in range(start_epoch, self._train_cfg.epochs):
            # from start_epoch to the final epoch
            lrs = []
            for k in range(self.n_classifiers):
                for param_group in self._state.optimizer[k].param_groups:
                    lrs.append(param_group['lr'])
            t_epoch_start = time.time()

            if job_env.global_rank == 0:
                print('[{:s}] Start epoch {:d}, learning rates = {}'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), epoch, lrs), flush=True)

            for i, data in enumerate(self._train_loader):
                inputs, labels = data
                inputs = inputs.cuda(job_env.local_rank, non_blocking=True)
                labels = labels.cuda(job_env.local_rank, non_blocking=True)
                batch_size = len(labels)

                if (t % period == 0) and (job_env.global_rank == 0):
                    order = np.arange(self.n_classifiers)
                    np.random.shuffle(order)
                    order = torch.IntTensor(order).cuda(job_env.local_rank, non_blocking=True)
                    # let the root node determines the order

                if (t % period == 0):
                    distributed.broadcast(order, src=0)
                    order_np = order.cpu().numpy()
                    if  (job_env.global_rank == 0):
                        print(f"Order = {order_np}")
                # order: the order of models to update in this period
                torch.autograd.set_detect_anomaly(True)
                if (t % period) < n_update_models * self.n_classifiers:
                    k = int((t % period) / n_update_models) # update one model with #n_update_models steps
                    i = order_np[k]

                    if delay_lr_scheduler[i] == 0:
                        delay_lr_scheduler[i] = -1
                    elif delay_lr_scheduler[i] > 0:
                        delay_lr_scheduler[i] = - delay_lr_scheduler[i] - 1

                    self._state.models.eval()
                    with torch.no_grad():
                        robust_acc_ex, inputs_adv, _ = adversary.perturb(inputs, labels, self._state.lbda,
                                                                         restart_all=False)
                    # attack the data minibatch
                    inputs_adv = torch.autograd.Variable(inputs_adv, requires_grad=False)  # adv input
                    inputs = torch.autograd.Variable(inputs, requires_grad=False)  # original inputs
                    self._state.models.train()
                    self._state.optimizer[i].zero_grad()
                    if self._train_cfg.attack == "TRADES":
                        logits = self._state.models[i](inputs)
                        loss_natural = criterion(logits, labels)

                        loss_robust = (1.0 / batch_size)*criterion_kl(F.log_softmax(self._state.models[i](inputs_adv), dim=1),
                                                                      F.softmax(self._state.models[i](inputs), dim=1))

                        loss = loss_natural+6.*loss_robust
                    else:
                        outputs = self._state.models[i](inputs_adv)
                        loss = criterion(outputs, labels)
                    # compute gradient on the adversarial inputs
                    loss.backward()
                    self._state.optimizer[i].step()
                    # gradient descent step
                else:
                    # update lbda
                    self._state.lbda = np.ones(self.n_classifiers) / self.n_classifiers
                    # initialize as equal weights
                    self._state.models.eval()
                    if self._train_cfg.reg < 0.:  # no regularization
                        for _ in range(m):  # update lbda with multiple GD steps
                            with torch.no_grad():
                                robust_acc, _, acc_per_model = adversary.perturb(inputs, labels, self._state.lbda,
                                                                                 n_restarts=self._train_cfg.n_restarts,
                                                                                 restart_all=False)

                            placeholder = torch.zeros_like(acc_per_model).cuda(
                                job_env.local_rank, non_blocking=True)
                            acc_per_model_list = [placeholder] * job_env.num_tasks
                            distributed.all_gather(
                                acc_per_model_list, acc_per_model.cuda(job_env.local_rank))
                            acc_per_model = torch.cat(acc_per_model_list, dim=0)

                            grad_lbda = 1. - acc_per_model.mean(dim=0).cpu().numpy()
                            self._state.lbda = self._state.lbda - \
                                np.sqrt(4. / m) * grad_lbda
                            self._state.lbda = projection_simplex_sort(self._state.lbda)
                    else:
                        for _ in range(m):
                            # estimate the gradient w.r.t. lbda
                            with torch.no_grad():
                                robust_acc, _, acc_per_model = adversary_ent.perturb(inputs, labels, self._state.lbda,
                                                                                     n_restarts=5, restart_all=True)

                            placeholder = torch.zeros_like(acc_per_model).cuda(
                                job_env.local_rank, non_blocking=True)
                            acc_per_model_list = [placeholder] * job_env.num_tasks
                            distributed.all_gather(
                                acc_per_model_list, acc_per_model.cuda(job_env.local_rank))
                            acc_per_model = torch.cat(acc_per_model_list, dim=1)
                            placeholder = torch.zeros_like(robust_acc).cuda(
                                job_env.local_rank, non_blocking=True)
                            robust_acc_list = [placeholder] * job_env.num_tasks
                            distributed.all_gather(
                                robust_acc_list, robust_acc.cuda(job_env.local_rank))
                            robust_acc = torch.cat(robust_acc_list, dim=1)

                            acc_per_model = 1. - acc_per_model.cpu().numpy()
                            acc_per_model = acc_per_model.transpose(2, 0, 1)
                            robust_acc = (1. - robust_acc.cpu().numpy()) / self._train_cfg.reg
                            ss = softmax(robust_acc, axis=0)
                            grad_lbda = (ss * acc_per_model)
                            grad_lbda = grad_lbda.sum(axis=1)
                            grad_lbda = grad_lbda.mean(axis=1)

                            # update lbda via PGD
                            self._state.lbda = self._state.lbda - self._train_cfg.lr_reg * grad_lbda
                            self._state.lbda = projection_simplex_sort(self._state.lbda)
                    if job_env.global_rank == 0:
                        print(f"grad_lbda = {grad_lbda}")
                        print(f"lbda = {self._state.lbda}")
                if i % print_freq == print_freq - 1 and job_env.global_rank == 0:
                    print(
                        f"[{epoch}, {i}] last loss: {loss.item()}, last robust acc = {robust_acc_ex.mean()},"
                        f"last lambda = {self._state.lbda}",
                        flush=True)

                t += 1

            t_epoch_finish = time.time()

            # Checkpoint only on the master
            self._state.epoch = epoch + 1
            if (job_env.global_rank == 0):
                self.checkpoint()  # save current model in each epoch
            self._state.epoch = epoch  # correct the epoch index
            # if ((epoch % self.n_classifiers) == 0):
            acc, acc_adv = self._eval(t_epoch_finish - t_epoch_start)  # evaluate the current performance

            # lr_scheduler update
            for k in range(self.n_classifiers):
                self._state.lr_scheduler[k].step(acc_adv)

            if (job_env.global_rank == 0) and (acc_adv > self._state.best_acc_adv):
                # save the current best result
                save_dir = os.path.join(self._train_cfg.data_root, self.ckpt_run)
                self._state.best_acc_adv = acc_adv

                self._state.save(os.path.join(save_dir, "best.pth"))

        if job_env.global_rank == 0:
            # save the final models after training
            save_dir = os.path.join(self._train_cfg.data_root, self.ckpt_run)
            self._state.save(os.path.join(save_dir, "last.pth"))
        # finish training

    def checkpoint(self, name="checkpoint.pth"):
        # will be called by submitit in case of preemption
        save_dir = os.path.join(self._train_cfg.data_root, self.ckpt_run)
        os.makedirs(save_dir, exist_ok=True)
        self._state.save(os.path.join(save_dir, name))
        return

    def _eval(self, epoch_time=0) -> Tuple[Any, Any]:

        job_env = self.job_env
        if job_env.global_rank == 0:
            print("Start evaluation of the model", flush=True)

        total = 0

        self._state.models.eval()  # eval mode

        eps_iter = 2. * self._train_cfg.eps_attack / 20  # attack step size
        adversary = PGDAttack(self._state.models, eps=self._train_cfg.eps_attack,
                              nb_iter=20, eps_iter=eps_iter, rand_init=True,
                              clip_min=0., clip_max=1., ord=np.inf, targeted=False)
        # compute natural accuracy
        correct = torch.zeros(self.n_classifiers).cuda(job_env.local_rank)
        correct_adv = torch.zeros(self.n_classifiers).cuda(job_env.local_rank)

        with torch.no_grad():
            for data in self._test_loader:
                images, labels = data
                images = images.cuda(job_env.local_rank, non_blocking=True)
                labels = labels.cuda(job_env.local_rank, non_blocking=True)
                total += labels.size(0)
                for k, l in enumerate(self._state.lbda):
                    outputs = self._state.models[k](images)
                    _, predicted = torch.max(outputs.data, 1)
                    correct[k] += (predicted == labels).cuda(job_env.local_rank).sum().item()

                robust_acc, _, acc_per_model = adversary.perturb(images, labels, self._state.lbda,
                                                                 n_restarts=1)
                correct_adv += acc_per_model.cuda(job_env.local_rank).sum(dim=0)

        distributed.all_reduce(correct, op=torch.distributed.ReduceOp.SUM)
        distributed.all_reduce(correct_adv, op=torch.distributed.ReduceOp.SUM)

        total = torch.Tensor([total]).cuda(job_env.local_rank)  # number of samples
        distributed.all_reduce(total, op=torch.distributed.ReduceOp.SUM)
        acc = (correct / total).cpu().numpy()  # natural accuracy
        acc_adv = (correct_adv / total).cpu().numpy()  # robust accuracy

        if job_env.global_rank == 0:
            print(f"Accuracy per model of the network on the 10000 test images: {acc}", flush=True)
            print(
                f"Adversarial accuracy per model of the network on the 10000 test images: {acc_adv}", flush=True)
        acc = acc.dot(self._state.lbda)  # the weigted average natural accuracy
        acc_adv = acc_adv.dot(self._state.lbda)  # the weigted average robust accuracy

        if job_env.global_rank == 0:
            print(f"Accuracy of the network on the 10000 test images: {acc:.1%}", flush=True)
            print(
                f"Adversarial accuracy of the network on the 10000 test images: {acc_adv:.1%}", flush=True)
            print(f"Epoch time: {epoch_time}", flush=True)
        if (job_env.global_rank == 0):
            with open(self._train_cfg.result_file, 'a') as f:
                f.write(f"{self._train_cfg.eps_attack} {self.n_classifiers} {self._state.epoch} "
                        f"{acc_adv} {acc} {epoch_time}\n")

        return acc, acc_adv

    def _big_eval(self, model_path: str, eval_file: str, lbda_default: np.array = None) -> None:
        print("Start evaluation of the model", flush=True)
        job_env = self.job_env
        torch.cuda.set_device(job_env.local_rank)
        distributed.init_process_group(
            backend=self._cluster_cfg.dist_backend,
            init_method=self._cluster_cfg.dist_url,
            world_size=job_env.num_tasks,
            rank=job_env.global_rank,
        )

        transform = transforms.Compose([transforms.ToTensor()])
        if self._train_cfg.dataset == "CIFAR10":
            test_set = datasets.CIFAR10(self._train_cfg.data_root, train=False,
                                        download=True, transform=transform)
            num_classes = 10
        else:
            test_set = datasets.CIFAR100(self._train_cfg.data_root, train=False,
                                         download=True, transform=transform)
            num_classes = 100

        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_set, num_replicas=job_env.num_tasks, rank=job_env.global_rank
        )
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=int(self._train_cfg.batch_per_gpu),
            num_workers=int(80 * self._cluster_cfg.num_nodes / self._cluster_cfg.num_gpus_per_node),
            sampler=test_sampler,
        )

        models_best = nn.ModuleList(
            [NormalizedModel(dict_models[model](num_classes=num_classes), mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) for model in
             self._train_cfg.models])

        for i in range(self.n_classifiers):
            models_best[i].cuda(job_env.local_rank)
            models_best[i] = torch.nn.parallel.DistributedDataParallel(models_best[i], device_ids=[job_env.local_rank],
                                                                       output_device=job_env.local_rank)
        models_best = nn.ModuleList(models_best)

        data = torch.load(model_path, map_location=lambda storage,
                          loc: storage.cuda(job_env.local_rank))
        models_best.load_state_dict(data[f"models"])
        models_best.eval()

        lbda = data["lbda"]
        if lbda_default is not None:
            np.copyto(lbda, lbda_default)
        epoch = data[f"epoch"]
        print(torch.cuda.get_device_name(0))
        print(f"lambda ={lbda}")
        print(f"epoch = {epoch}")
        adversary_ce = APGDAttack(models_best, n_iter=100,
                                  norm='Linf', eps=self._train_cfg.eps_attack,
                                  loss="ce", rho=.75, verbose=False)

        adversary_dlr = APGDAttack(models_best, n_iter=100,
                                   norm='Linf', eps=self._train_cfg.eps_attack,
                                   loss="dlr", rho=.75, verbose=False)

        total = 0
        correct = torch.zeros(self.n_classifiers).cuda(job_env.local_rank)
        correct_ce = torch.zeros(self.n_classifiers).cuda(job_env.local_rank)
        correct_dlr = torch.zeros(self.n_classifiers).cuda(job_env.local_rank)
        correct_all = torch.zeros(self.n_classifiers).cuda(job_env.local_rank)
        t0 = time.time()

        with torch.no_grad():
            for i, data in enumerate(test_loader):
                images, labels = data
                images = images.cuda(job_env.local_rank, non_blocking=True)
                labels = labels.cuda(job_env.local_rank, non_blocking=True)
                total += labels.size(0)
                for k, l in enumerate(lbda):
                    outputs = models_best[k](images)
                    _, predicted = torch.max(outputs.data, 1)
                    correct[k] += (predicted == labels).cuda(job_env.local_rank).sum().item()

                _, _, acc_per_model_ce = adversary_ce.perturb(images, labels, lbda, n_restarts=5)
                _, _, acc_per_model_dlr = adversary_dlr.perturb(images, labels, lbda, n_restarts=5)

                acc_per_model_all = acc_per_model_ce * acc_per_model_dlr
                correct_ce += acc_per_model_ce.cuda(job_env.local_rank).sum(dim=0)
                correct_dlr += acc_per_model_dlr.cuda(job_env.local_rank).sum(dim=0)
                correct_all += acc_per_model_all.cuda(job_env.local_rank).sum(dim=0)
                print(f"batch {i} done")

        distributed.all_reduce(correct, op=torch.distributed.ReduceOp.SUM)
        distributed.all_reduce(correct_ce, op=torch.distributed.ReduceOp.SUM)
        distributed.all_reduce(correct_dlr, op=torch.distributed.ReduceOp.SUM)
        distributed.all_reduce(correct_all, op=torch.distributed.ReduceOp.SUM)

        total = torch.Tensor([total]).cuda(job_env.local_rank)
        distributed.all_reduce(total, op=torch.distributed.ReduceOp.SUM)

        acc = (correct / total).cpu().numpy()
        acc_ce = (correct_ce / total).cpu().numpy()
        acc_dlr = (correct_dlr / total).cpu().numpy()
        acc_all = (correct_all / total).cpu().numpy()

        acc = acc.dot(lbda)
        acc_ce = acc_ce.dot(lbda)
        acc_dlr = acc_dlr.dot(lbda)
        acc_all = acc_all.dot(lbda)

        if (job_env.global_rank == 0):
            print(f"Accuracy of the network on the 10000 test images: {acc:.1%}", flush=True)
            print(
                f"Adversarial CE accuracy of the network on the 10000 test images: {acc_ce:.1%}", flush=True)
            print(
                f"Adversarial DLR accuracy of the network on the 10000 test images: {acc_dlr:.1%}", flush=True)
            print(
                f"Adversarial ALL accuracy of the network on the 10000 test images: {acc_all:.1%}", flush=True)
            print(f"time: {time.time()-t0}")

            with open(eval_file, 'a') as f:
                f.write(f"{model_path} {epoch} {acc} {acc_ce} {acc_dlr} {acc_all}\n")
