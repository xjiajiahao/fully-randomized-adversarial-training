from codecs import replace_errors
import os
from struct import iter_unpack

from matplotlib.pyplot import clabel
from config import ClusterConfig, TrainerConfig, JobEnvironment
from torch import optim, nn
import attr
import torch.backends.cudnn as cudnn
from typing import List as List
from typing import Tuple as Tuple
from typing import Any as Any
import time
from auto_attack import APGDAttack
from torch import distributed
from models import ResNet18, PreActResNet18, GoogLeNet, SimpleDLA, DenseNet121, WideResNet28x10, WideResNet34x20
from pgd_attack import PGDAttack
from trades import TRADESAttack
from datetime import datetime
import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from utils import NormalizedModel
import torch.nn.functional as F
import collections

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
class MyTrainerState:
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
    if __debug__:
        classifier_buffer: List[nn.ModuleList]
        lbda_buffer: List[np.array]
    iter_idx: int

    def save(self, filename: str) -> None:
        data = attr.asdict(self)
        # store only the state dict
        data[f"models"] = self.models.state_dict()
        data[f"optimizer"] = []
        data[f"lr_scheduler"] = []
        if __debug__:
            data[f"classifier_buffer"] = []
            data[f"lbda_buffer"] = []

        for opt in self.optimizer:
            data[f"optimizer"].append(opt.state_dict())

        for lr_s in self.lr_scheduler:
            data[f"lr_scheduler"].append(lr_s.state_dict())

        data[f"lbda"] = self.lbda
        data[f"epoch"] = self.epoch
        data["best_acc_adv"] = self.best_acc_adv
        if __debug__:
            for ms in self.classifier_buffer:
                data[f"classifier_buffer"].append(ms.state_dict())
            data[f"lbda_buffer"] = self.lbda_buffer
        data[f"iter_idx"] = self.iter_idx
        torch.save(data, filename)

    @classmethod
    def load(cls, filename: str, default: "MyTrainerState", gpu: int) -> "MyTrainerState":
        data = torch.load(filename, map_location=lambda storage, loc: storage.cuda(gpu))  # load tensor to gpu
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
        if __debug__:
            for k, ms in enumerate(default.classifier_buffer):
                ms.load_state_dict(data[f"classifier_buffer"][k])
                # ms.to('cpu')
                data[f"classifier_buffer"][k] = ms
        return cls(**data)  # dict as arguments


class MyTrainer:
    def __init__(self, train_cfg: TrainerConfig, cluster_cfg: ClusterConfig, job_env: JobEnvironment):
        self._train_cfg = train_cfg
        self._cluster_cfg = cluster_cfg
        self.n_classifiers = len(train_cfg.models)
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
        # distribute a minibatch to all GPUs

        self._train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=int(self._train_cfg.batch_per_gpu),
            num_workers=2,
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
            num_workers=2,
            sampler=test_sampler,
        )
        print(f"Total batch_size: {self._train_cfg.batch_per_gpu * job_env.num_tasks}", flush=True)  # the large batch

        print("Create distributed model", flush=True)

        models = nn.ModuleList([NormalizedModel(dict_models[model](num_classes=num_classes),
                                                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) for model in
                                self._train_cfg.models])
        # normalize each image to the range [-1, 1], not normalizing in transforms because the attacker wants the original image
        if __debug__:
            classifier_buffer = [nn.ModuleList([NormalizedModel(dict_models[model](num_classes=num_classes),
                                                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) for model in
                                    self._train_cfg.models]) for _ in range(self._train_cfg.size_classifier_buffer)]
            # copy models to each classifier_buffer
            with torch.no_grad():
                for j in range(self._train_cfg.size_classifier_buffer):
                    classifier_buffer[j].load_state_dict(models.state_dict())
                    classifier_buffer[j].cuda(job_env.local_rank)


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

        if __debug__:
            lbda_buffer = [ np.copy(lbda) for _ in range(self._train_cfg.size_classifier_buffer)]
        # initialize as equal weights

        if __debug__:
            self._state = MyTrainerState(epoch=0, models=models, optimizer=optimizer,
                                       lr_scheduler=lr_scheduler, lbda=lbda, best_acc_adv=-1,
                                       classifier_buffer=classifier_buffer,
                                       lbda_buffer=lbda_buffer,
                                       iter_idx=0)
        if not __debug__:
            self._state = MyTrainerState(epoch=0, models=models, optimizer=optimizer,
                                       lr_scheduler=lr_scheduler, lbda=lbda, best_acc_adv=-1,
                                       iter_idx=0)

        # the initial state
        self.ckpt_run = self._train_cfg.ckpt_run
        checkpoint_fn = os.path.join(self._train_cfg.data_root, self.ckpt_run, "checkpoint.pth")
        # load checkpoint
        if os.path.isfile(checkpoint_fn):
            print(f"Load existing checkpoint from {checkpoint_fn}", flush=True)
            self._state = MyTrainerState.load(
                checkpoint_fn, default=self._state, gpu=job_env.local_rank)
            # restore state
        criterion = nn.CrossEntropyLoss(reduction='mean').cuda(job_env.local_rank)
        criterion_kl = nn.KLDivLoss(reduction='sum').cuda(job_env.local_rank)
        # KL divergence loss

        # Start from the loaded epoch
        start_epoch = self._state.epoch
        start_t = self._state.iter_idx
        if self._train_cfg.attack == "APGD":
            adversary = APGDAttack(self._state.models, n_iter=self._train_cfg.iter_attack, norm='Linf',
                                   eps=self._train_cfg.eps_attack, loss=self._train_cfg.loss, rho=.75, verbose=False)

        elif self._train_cfg.attack == "PGD":
            eps_iter = 2. * self._train_cfg.eps_attack / self._train_cfg.iter_attack
            adversary = PGDAttack(self._state.models, eps=self._train_cfg.eps_attack,
                                  nb_iter=self._train_cfg.iter_attack, eps_iter=eps_iter, rand_init=True,
                                  clip_min=0., clip_max=1., ord=np.inf, targeted=False)
        elif self._train_cfg.attack == "TRADES":
            eps_iter = 2. * self._train_cfg.eps_attack / self._train_cfg.iter_attack
            adversary = TRADESAttack(self._state.models, eps=self._train_cfg.eps_attack,
                                     nb_iter=self._train_cfg.iter_attack, eps_iter=eps_iter,
                                     clip_min=0., clip_max=1.)

        if job_env.global_rank == 0:
            print(f"Attack characteristics : epsilon = {self._train_cfg.eps_attack}, iters = {self._train_cfg.iter_attack}, size_classifier_buffer = {self._train_cfg.size_classifier_buffer}")
            print(f"Regularization : {self._train_cfg.reg}")
            print(f"Models: {self._train_cfg.models}")
            print(f"Number of epochs: {int(self._train_cfg.epochs)}")
            print(f"Size of the classifier buffer: {int(self._train_cfg.size_classifier_buffer)}")


        t = start_t
        acc_adv = -1
        curr_losses = torch.zeros(self.n_classifiers).cuda(job_env.local_rank)
        # the main loop
        for epoch in range(start_epoch, self._train_cfg.epochs):
            # from start_epoch to the final epoch
            lrs = []
            for k in range(self.n_classifiers):
                for param_group in self._state.optimizer[k].param_groups:
                    lrs.append(param_group['lr'])
            t_epoch_start = time.time()

            if job_env.global_rank == 0:
                print('[{:s}] Start epoch {:d}, learning rates = {}'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), epoch, lrs), flush=True)
                print(f"mixing weights = {self._state.lbda}",
                      flush=True)

            self._state.iter_idx = t
            for i, data in enumerate(self._train_loader):
                inputs, labels = data
                inputs = inputs.cuda(job_env.local_rank, non_blocking=True)
                labels = labels.cuda(job_env.local_rank, non_blocking=True)
                batch_size = len(labels)

                torch.autograd.set_detect_anomaly(True)
                # generate perturbed data points
                self._state.models.eval()
                with torch.no_grad():
                    if __debug__:
                        if t < self._train_cfg.size_classifier_buffer - 1:
                            classifier_buffer = self._state.classifier_buffer[:t]
                            lbda_buffer = self._state.lbda_buffer[:t]
                        else:
                            classifier_buffer = self._state.classifier_buffer
                            lbda_buffer = self._state.lbda_buffer
                    if not __debug__:
                        classifier_buffer = [self._state.models]
                        lbda_buffer = [self._state.lbda]
                    inputs_adv = adversary.lbr_perturb(inputs, labels, classifier_buffer, lbda_buffer,
                                                        restart_all=False,
                                                        noise_level=self._train_cfg.noise_level_attack)

                self._state.models.train()
                curr_losses.zero_()
                for model_idx in range(self.n_classifiers):
                    self._state.optimizer[model_idx].zero_grad()
                    if self._train_cfg.attack == "TRADES":
                        logits = self._state.models[model_idx](inputs)
                        loss_natural = criterion(logits, labels)

                        loss_robust = (1.0 / batch_size)*criterion_kl(F.log_softmax(self._state.models[model_idx](inputs_adv), dim=1),
                                                                      F.softmax(self._state.models[model_idx](inputs), dim=1))


                        loss = loss_natural+6.*loss_robust
                    else:
                        outputs = self._state.models[model_idx](inputs_adv)
                        loss = criterion(outputs, labels)
                    # compute gradient on the adversarial inputs
                    loss.backward()
                    # gradient descent step
                    self._state.optimizer[model_idx].step()
                    curr_losses[model_idx] = loss.data.item()
                distributed.all_reduce(curr_losses, op=torch.distributed.ReduceOp.SUM)
                curr_losses.mul_(1.0 / job_env.num_tasks)

                # update the model weights lbda
                self._state.lbda = self._state.lbda * np.exp(- self._train_cfg.lr_reg * curr_losses.cpu().numpy())
                self._state.lbda = self._state.lbda / np.sum(self._state.lbda)

                t += 1

                # maintain classifier_buffer and lbda_buffer
                if __debug__:
                    buffer_idx = t % self._train_cfg.size_classifier_buffer
                    np.copyto(self._state.lbda_buffer[buffer_idx], self._state.lbda)

                    model_dic = self._state.models.state_dict()
                    new_state_dict = collections.OrderedDict() 
                    for k, v in model_dic.items(): 
                        name = k.replace('module.', '')  # remove `module.` 
                        new_state_dict[name] = v
                    self._state.classifier_buffer[buffer_idx].load_state_dict(new_state_dict)

            t_epoch_finish = time.time()

            # Checkpoint only on the master
            self._state.epoch = epoch + 1
            if (job_env.global_rank == 0):
                self.checkpoint()  # save current model in each epoch
            self._state.epoch = epoch  # correct the epoch index
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

    def _eval(self, epoch_time) -> Tuple[Any, Any]:

        # job_env = submitit.JobEnvironment()
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

    def _big_eval(self, model_path: str, eval_file: str) -> None:
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
            num_workers=2,
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
