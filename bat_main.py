import argparse
import torch
import os
from bat_cifar import PinotTrainer
from config import ClusterConfig, JobEnvironment, TrainerConfig
import numpy as np


# define input arguments
parser = argparse.ArgumentParser()
# parser.add_argument('--local_rank', type=int, default=0,
#                     help='the local device id determined by pytorch, no need to set')
# parser.add_argument('--master_addr', type=str, default='127.0.0.1',
#                     help='the ip of the central node')
parser.add_argument('--num_nodes', type=int, default=1,
                    help='number of machines')
parser.add_argument('--node_rank', type=int, default=0,
                    help='the global rank of the node')
parser.add_argument('--num_gpus_per_node', type=int, default=1,
                    help='number of gpus per node')
parser.add_argument('--epochs', type=int, default=200,
                    help='the total number of epochs')
parser.add_argument('--batch_per_gpu', type=int, default=128,
                    help='split a minibatch to multiple GPUs')
parser.add_argument('--data_root', type=str, default='./data/',
                    help='where the data and result files are stored')
parser.add_argument('--model_name', type=str, default='ResNet18',
                    const='ResNet18', nargs='?',
                    choices=['ResNet18', 'PreActResNet18', 'GoogLeNet', 'SimpleDLA',
                             'DenseNet121', 'WideResNet28x10', 'WideResNet34x20'],
                    help='name of the classification model')
parser.add_argument('--n_classifiers', type=int, default=2,
                    help='number of the classifiers')
parser.add_argument('--reg', type=float, default=0.001,
                    help='regularization parameter')
parser.add_argument('--eps_attack', type=float, default=8/255,
                    help='the adversarial level, default: 8/255')
parser.add_argument('--iter_attack', type=int, default=20,
                    help='number of iterations of the attack')
parser.add_argument('--n_restarts', type=int, default=3,
                    help='number of restarts, each run returns a attack point')
parser.add_argument('--loss', type=str, default='ce',
                    const='all', nargs='?', choices=['ce', 'dlr'],
                    help='ce loss or dlr loss for attack')
parser.add_argument('--result_file', type=str, default='./data/result.csv',
                    help='file to record the accuracy at each epoch')
parser.add_argument('--lr_reg', type=float, default=0.001,
                    help='learning rate of the mixing weights')
parser.add_argument('--attack', type=str, default='APGE',
                    const='APGD', nargs='?', choices=['APGD', 'PGD', 'TRADES'],
                    help='attack method to use')
parser.add_argument('--ckpt_run', type=str, default='result',
                    help='name of the folder that stores checkpoints, the checkpoint file is data_root/ckpt_run/checkpoint.pth')
parser.add_argument('--dataset', type=str, default='CIFAR10',
                    const='CIFAR10', nargs='?', choices=['CIFAR10', 'CIFAR100'],
                    help='dataset to use, which can be CIFAR10 or CIFAR100')
parser.add_argument('--lbda0', type=float, default=None,
                    help='the weight of the first component')
parser.add_argument('--phase', type=int, default=1,
                    help='must be executed for 3 times in the order 1, 2, 3')

args = parser.parse_args()

local_rank = int(os.environ["LOCAL_RANK"])
num_nodes = args.num_nodes
node_rank = args.node_rank
num_gpus_per_node = args.num_gpus_per_node
epochs = args.epochs
batch_per_gpu = args.batch_per_gpu
data_root = args.data_root
model_name = args.model_name
n_classifiers = args.n_classifiers
reg = args.reg
eps_attack = args.eps_attack
iter_attack = args.iter_attack
n_restarts = args.n_restarts
loss = args.loss
result_file = args.result_file
eps_attack = args.eps_attack
lr_reg = args.lr_reg
attack = args.attack
ckpt_run = args.ckpt_run
dataset = args.dataset
lbda0 = args.lbda0
phase = args.phase

models = [model_name] * n_classifiers

dist_backend = 'nccl'
dist_url = None
if not torch.cuda.is_available():
    raise ValueError("No cuda device available")

num_tasks = num_nodes * num_gpus_per_node
global_rank = local_rank + node_rank * num_gpus_per_node


cluster_cfg = ClusterConfig(dist_backend, dist_url,
                            num_nodes, num_gpus_per_node)
train_cfg = TrainerConfig(epochs, batch_per_gpu, data_root, models, reg, 
                          eps_attack, iter_attack, loss,
                          result_file, lr_reg, attack, ckpt_run, dataset,
                          n_restarts=n_restarts)
job_env = JobEnvironment(local_rank, num_tasks, global_rank)

trainer = PinotTrainer(train_cfg, cluster_cfg, job_env)

if phase == 1:
    trainer.gen_adv_dataset()
elif phase == 2:
    trainer.natural_train_over_perturbed_data()
else:
    save_dir = os.path.join(data_root, ckpt_run)
    model_path = os.path.join(save_dir, "bat_strategy.pth")
    eval_file = os.path.join(save_dir, "bat_final_eval_result.csv")
    models = [model_name] * 2
    train_cfg = TrainerConfig(epochs, batch_per_gpu, data_root, models, reg, 
                              eps_attack, iter_attack, loss,
                              result_file, lr_reg, attack, ckpt_run, dataset,
                              n_restarts=n_restarts)
    trainer = PinotTrainer(train_cfg, cluster_cfg, job_env)
    if lbda0 is not None:
        lbda = np.zeros(2)
        lbda[0] = lbda0
        lbda[1] = 1.0 - lbda0
        print(f"evaluating BAT algorithm with lbda = {lbda}")
        trainer._big_eval(model_path, eval_file, lbda_default=lbda)
    else:
        trainer._big_eval(model_path, eval_file)
    