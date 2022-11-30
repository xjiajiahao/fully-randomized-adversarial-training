import os
from config import *
from torch import optim, nn
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
from atm_cifar import Trainer, TrainerState
import collections
from torch.utils.data.dataset import TensorDataset

dict_models = {
    "ResNet18": ResNet18,
    "PreActResNet18": PreActResNet18,
    "GoogLeNet": GoogLeNet,
    "SimpleDLA": SimpleDLA,
    "DenseNet121": DenseNet121,
    "WideResNet28x10": WideResNet28x10,
    "WideResNet34x20": WideResNet34x20
}


class PinotTrainer(Trainer):
    def __init__(self, train_cfg: TrainerConfig, cluster_cfg: ClusterConfig, job_env: JobEnvironment):
        super(PinotTrainer, self).__init__(train_cfg, cluster_cfg, job_env)

    def gen_adv_dataset(self):
        cudnn.benchmark = True  # accelerates the CNN
        job_env = self.job_env  # so, how to set JobEnv?
        torch.cuda.set_device(job_env.local_rank)  # local_rank: gpu id in one machine

        print("Create data loaders", flush=True)

        transform = transforms.Compose([transforms.ToTensor()])
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

        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=int(self._train_cfg.batch_per_gpu),
            num_workers=2,
            shuffle=True)

        models_best = nn.ModuleList(
            [NormalizedModel(dict_models[model](num_classes=num_classes), mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) for model in
             self._train_cfg.models])

        for i in range(self.n_classifiers):
            models_best[i].cuda(job_env.local_rank)
        models_best = nn.ModuleList(models_best)

        self.ckpt_run = self._train_cfg.ckpt_run
        model_path = os.path.join(self._train_cfg.data_root, self.ckpt_run, "best.pth")

        data = torch.load(model_path, map_location=lambda storage,
                          loc: storage.cuda(job_env.local_rank))
        model_dict = data[f"models"]
        new_state_dict = collections.OrderedDict() 
        for k, v in model_dict.items(): 
            name = k.replace('module.', '')# remove `module.` 
            new_state_dict[name] = v
        models_best.load_state_dict(new_state_dict)
        models_best.eval()
        lbda = data["lbda"]
        lbda_tensor = torch.Tensor(lbda)

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

        perturbed_data = []
        targets = []
        with torch.no_grad():
            for i, data in enumerate(train_loader):
                images, labels = data
                images = images.cuda(job_env.local_rank, non_blocking=True)
                labels = labels.cuda(job_env.local_rank, non_blocking=True)
                total += labels.size(0)
                for k, l in enumerate(lbda):
                    outputs = models_best[k](images)
                    _, predicted = torch.max(outputs.data, 1)
                    correct[k] += (predicted == labels).cuda(job_env.local_rank).sum().item()

                _, adv_data_ce, acc_per_model_ce = adversary_ce.perturb(images, labels, lbda)
                _, adv_data_dlr, acc_per_model_dlr = adversary_dlr.perturb(images, labels, lbda)

                acc_ce = torch.matmul(acc_per_model_ce, lbda_tensor)
                acc_dlr = torch.matmul(acc_per_model_dlr, lbda_tensor)

                for sample_idx in range(labels.size(0)):
                    if acc_ce[sample_idx] > acc_dlr[sample_idx]:
                        adv_data_ce[sample_idx] = adv_data_dlr[sample_idx]
                adv_data_ce = adv_data_ce.to('cpu')
                perturbed_data.append(adv_data_ce)
                labels = labels.to('cpu')
                targets.append(labels)

                acc_per_model_all = acc_per_model_ce * acc_per_model_dlr
                correct_ce += acc_per_model_ce.cuda(job_env.local_rank).sum(dim=0)
                correct_dlr += acc_per_model_dlr.cuda(job_env.local_rank).sum(dim=0)
                correct_all += acc_per_model_all.cuda(job_env.local_rank).sum(dim=0)
                print(f"batch {i} done")

        perturbed_data = torch.cat(perturbed_data, dim=0)
        targets = torch.cat(targets, dim=0)

        perturbed_data_path = os.path.join(self._train_cfg.data_root, self.ckpt_run, "perturbed_data_for_at.pth")
        torch.save({'features': perturbed_data, 'labels': targets}, perturbed_data_path)

        acc = (correct / total).cpu().numpy()
        acc_ce = (correct_ce / total).cpu().numpy()
        acc_dlr = (correct_dlr / total).cpu().numpy()
        acc_all = (correct_all / total).cpu().numpy()

        acc = acc.dot(lbda)
        acc_ce = acc_ce.dot(lbda)
        acc_dlr = acc_dlr.dot(lbda)
        acc_all = acc_all.dot(lbda)
        print(f"Accuracy of the network on the 10000 test images: {acc:.1%}", flush=True)
        print(
            f"Adversarial CE accuracy of the network on the 10000 test images: {acc_ce:.1%}", flush=True)
        print(
            f"Adversarial DLR accuracy of the network on the 10000 test images: {acc_dlr:.1%}", flush=True)
        print(
            f"Adversarial ALL accuracy of the network on the 10000 test images: {acc_all:.1%}", flush=True)
        print(f"time: {time.time()-t0}")

    def natural_train_over_perturbed_data(self):
        # step 1. initialize the data loader
        self.ckpt_run = self._train_cfg.ckpt_run
        if self.n_classifiers != 1:
            raise ValueError("n_classifiers MUST BE 1.")
        cudnn.benchmark = True  # accelerates the CNN
        job_env = self.job_env  # so, how to set JobEnv?
        torch.cuda.set_device(job_env.local_rank)  # local_rank: gpu id in one machine
        distributed.init_process_group(
            backend=self._cluster_cfg.dist_backend,
            init_method=self._cluster_cfg.dist_url,
            world_size=job_env.num_tasks,
            rank=job_env.global_rank,
        )


        print("Create data loader", flush=True)

        transform = transforms.Compose([transforms.ToTensor()])
        if self._train_cfg.dataset == "CIFAR10":
            # the CIFAR-10 training data set
            num_classes = 10
        else:
            # the CIFAR-100 training data set
            num_classes = 100
        
        perturbed_data_path = os.path.join(self._train_cfg.data_root, self.ckpt_run, "perturbed_data_for_at.pth")
        data_dict = torch.load(perturbed_data_path, map_location='cpu')
        features = data_dict['features']
        labels = data_dict['labels']
        train_set = TensorDataset(features, labels)

        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_set, num_replicas=job_env.num_tasks, rank=job_env.global_rank
        )  # num_replicas: Number of processes participating in distributed training. By default
        # distribute a minibatch to the #num_tasks GPUs

        self._train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=int(self._train_cfg.batch_per_gpu),
            num_workers=0,  # seems a lot workers
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
            num_workers=0,
            sampler=test_sampler,
        )
        print(f"Total batch_size: {self._train_cfg.batch_per_gpu * job_env.num_tasks}", flush=True)  # the large batch

        # step 2. initialize the 2-mixture model
        models= nn.ModuleList(
            [NormalizedModel(dict_models[model](num_classes=num_classes), mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) for model in
             self._train_cfg.models])

        for i in range(self.n_classifiers):
            models[i].cuda(job_env.local_rank)
        models = nn.ModuleList(models)

        self.ckpt_run = self._train_cfg.ckpt_run
        model_path = os.path.join(self._train_cfg.data_root, self.ckpt_run, "best.pth")

        data = torch.load(model_path, map_location=lambda storage,
                          loc: storage.cuda(job_env.local_rank))
        model_dict = data[f"models"]
        new_state_dict = collections.OrderedDict() 
        for k, v in model_dict.items(): 
            name = k.replace('module.', '')# remove `module.` 
            new_state_dict[name] = v
        models.load_state_dict(new_state_dict)

        model_new = NormalizedModel(dict_models[self._train_cfg.models[0]](num_classes=num_classes), mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        model_new.cuda(job_env.local_rank)
        model_new.train()
        models.append(model_new)

        # set self.n_classifiers to 2
        self.n_classifiers = 2
        for i in range(self.n_classifiers):
            models[i] = torch.nn.parallel.DistributedDataParallel(models[i], device_ids=[job_env.local_rank],
                                                                  output_device=job_env.local_rank, broadcast_buffers=False)
        models= nn.ModuleList(models)

        lbda = np.ones(2) / 2

        # step 3. initialize the optimizer
        optimizer = [optim.SGD(models[k].parameters(),
                               lr=max(0.1 * self._train_cfg.batch_per_gpu *
                                      job_env.num_tasks / 256, 0.1),  # lr=0.4
                               momentum=0.9, weight_decay=5e-4) for k in range(2)]

        lr_scheduler = [torch.optim.lr_scheduler.MultiStepLR(optimizer[k],
                                                             [int(self._train_cfg.epochs / 2),
                                                              int(3 * self._train_cfg.epochs / 4)],
                                                             gamma=0.1) for k in
                        range(2)]

        self._state = TrainerState(epoch=0, models=models, optimizer=optimizer,
                                   lr_scheduler=lr_scheduler, lbda=lbda, best_acc_adv=-1)

        # step 4. training for multiple epochs
        criterion = nn.CrossEntropyLoss(reduction='mean').cuda(job_env.local_rank)

        for epoch in range(self._train_cfg.epochs):
            lrs = []
            for param_group in self._state.optimizer[1].param_groups:
                lrs.append(param_group['lr'])

            if job_env.global_rank == 0:
                print('[{:s}] Start epoch {:d}, learning rates = {}'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), epoch, lrs), flush=True)

            running_loss = 0
            count = 0
            for i, data in enumerate(self._train_loader):
                inputs, labels = data
                inputs = inputs.cuda(job_env.local_rank, non_blocking=True)
                labels = labels.cuda(job_env.local_rank, non_blocking=True)
                self._state.optimizer[1].zero_grad()

                outputs = self._state.models[1](inputs)
                loss = criterion(outputs, labels)
                # compute gradient on the adversarial inputs
                loss.backward()
                self._state.optimizer[1].step()
                running_loss += loss.detach().cpu()
                count += 1
            self._state.lr_scheduler[1].step()
            running_loss /= count
            print('[{:s}] running loss: {:f}'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), running_loss))

        # step 5. line search for lbda
        models.eval()
        best_lbda = None
        for i in range(0, 11):
            self._state.lbda[0] = i * 0.1
            self._state.lbda[1] = (10 - i) * 0.1
            _, acc_adv = self._eval()
            if (acc_adv > self._state.best_acc_adv):
                best_lbda = np.copy(self._state.lbda)
                self._state.best_acc_adv = acc_adv
        np.copyto(self._state.lbda, best_lbda)

        # step 6. save the model
        if job_env.global_rank == 0:
            print(f"best lbda = {self._state.lbda}, best acc_adv: {self._state.best_acc_adv}")
            # save the current best result
            save_dir = os.path.join(self._train_cfg.data_root, self.ckpt_run)
            self._state.save(os.path.join(save_dir, "bat_strategy.pth"))
       