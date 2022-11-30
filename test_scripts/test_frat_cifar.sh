#!/bin/bash
# input argument:
#   nic: the network interface card name (e.g., eno2)
#   machine_rank: the rank of the node (only applicable when using multiple nodes)



################### modify the setting here ###########################
dataset='CIFAR10'     # CIFAR-10 dataset
# dataset='CIFAR100'     # CIFAR-100 dataset

is_eval_mode='False'   # training mode
# is_eval_mode='True'  # evaluation mode

n_classifiers=2
batch_per_gpu=256
epochs=200

# n_classifiers=3
# batch_per_gpu=256
# epochs=200

# n_classifiers=4
# batch_per_gpu=128
# epochs=200
#########################################################################











if [[ $# -ne 0 ]] ; then
    export GLOO_SOCKET_IFNAME=$1
    export NCCL_SOCKET_IFNAME=$1
    # export NCCL_DEBUG=INFO
fi
export PYTHONOPTIMIZE=1  # remove the __debug__ code
# export OMP_NUM_THREADS=1

master_addr='0.0.0.0'  # the central node IP address

iter_attack=10

num_nodes=1
node_rank=$2
num_gpus_per_node=4
data_root='./data/'
model_name='ResNet18'
reg=0.001
eps_attack=0.03137254901960784  # 8/255
# lr_reg=0.001
lr_reg=0.0003162
# attack='APGD'
# attack='PGD'
attack='TRADES'
# size_classifier_buffer=20
# size_classifier_buffer=2
size_classifier_buffer=1
noise_level_attack=0.0001
# noise_level_attack=0.000
ckpt_run="results_aeg_${dataset}_my_${attack}_${n_classifiers}_models_${iter_attack}_attacks_${size_classifier_buffer}_buffers"
result_file="${data_root}/${ckpt_run}/result.csv"

# python3 -m torch.distributed.launch --nproc_per_node=${num_gpus_per_node} \
torchrun --nproc_per_node=${num_gpus_per_node} \
    --nnodes=${num_nodes} \
    --node_rank=${node_rank} \
    --master_addr=${master_addr} \
    --master_port=2500 \
    ./my_main.py \
    --num_nodes=${num_nodes} \
    --node_rank=${node_rank} \
    --num_gpus_per_node=${num_gpus_per_node} \
    --epochs=${epochs} \
    --batch_per_gpu=${batch_per_gpu} \
    --data_root=${data_root} \
    --model_name=${model_name} \
    --n_classifiers=${n_classifiers} \
    --reg=${reg} \
    --eps_attack=${eps_attack} \
    --iter_attack=${iter_attack} \
    --result_file=${result_file} \
    --lr_reg=${lr_reg} \
    --attack=${attack} \
    --ckpt_run=${ckpt_run} \
    --dataset=${dataset} \
    --size_classifier_buffer=${size_classifier_buffer} \
    --noise_level_attack=${noise_level_attack} \
    --is_eval_mode=${is_eval_mode}
