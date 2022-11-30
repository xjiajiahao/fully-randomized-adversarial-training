import argparse
import torch
import os
from frat_cifar import MyTrainer
from config import ClusterConfig, JobEnvironment, TrainerConfig


# define input arguments
parser = argparse.ArgumentParser()
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
parser.add_argument('--loss', type=str, default='ce',
                    const='all', nargs='?', choices=['ce', 'dlr'],
                    help='ce loss or dlr loss for attack (not used)')
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
parser.add_argument('--size_classifier_buffer', type=int, default=20,
                    help='the size of the classifier buffer')
parser.add_argument('--noise_level_attack', type=float, default=0.001,
                    help='the noise level of the randomized attack')
parser.add_argument('--is_eval_mode', type=lambda x: (str(x).lower()
                                                    in ['true', '1', 'yes']), default=False)

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
loss = args.loss
result_file = args.result_file
eps_attack = args.eps_attack
lr_reg = args.lr_reg
attack = args.attack
ckpt_run = args.ckpt_run
dataset = args.dataset
size_classifier_buffer = args.size_classifier_buffer
noise_level_attack = args.noise_level_attack
is_eval_mode = args.is_eval_mode

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
                          size_classifier_buffer=size_classifier_buffer,
                          noise_level_attack=noise_level_attack)
job_env = JobEnvironment(local_rank, num_tasks, global_rank)

trainer = MyTrainer(train_cfg, cluster_cfg, job_env)
if not is_eval_mode:
    trainer()
else:
    save_dir = os.path.join(data_root, ckpt_run)
    model_path = os.path.join(save_dir, "best.pth")
    eval_file = os.path.join(save_dir, "final_eval_result.csv")
    trainer._big_eval(model_path, eval_file)
