#!/bin/bash
# input argument:
#   nic: the network interface card name (e.g., eno2)
#   machine_rank: the rank of the node (only applicable when using multiple nodes)


################### modify the setting here ###########################
dataset='CIFAR10'     # CIFAR-10 dataset
# dataset='CIFAR100'     # CIFAR-100 dataset

is_eval_mode='False'   # training mode
# is_eval_mode='True'  # evaluation mode

epochs=200
#########################################################################






if [[ $# -ne 0 ]] ; then
    export GLOO_SOCKET_IFNAME=$1
    export NCCL_SOCKET_IFNAME=$1
    # export NCCL_DEBUG=INFO
fi
export PYTHONOPTIMIZE=1  # remove the __debug__ code

master_addr='0.0.0.0'  # the central node IP address

iter_attack=10

num_gpus_per_node=4
num_gpus_per_node_phase_1=1

n_classifiers=1

num_nodes=1
batch_per_gpu=256
node_rank=$2
data_root='./data/'
model_name='ResNet18'
reg=0.001
eps_attack=0.03137254901960784  # 8/255
n_restarts=5
loss='ce'
lr_reg=0.001
# attack='APGD'
# attack='PGD'
attack='TRADES'
ckpt_run="results_aeg_${dataset}_meunier_${attack}_${loss}_${n_classifiers}_models_${iter_attack}_attacks"
result_file="${data_root}/${ckpt_run}/result.csv"

# python3 -m torch.distributed.launch --nproc_per_node=${num_gpus_per_node} \
torchrun --nproc_per_node=${num_gpus_per_node} \
    --nnodes=${num_nodes} \
    --node_rank=${node_rank} \
    --master_addr=${master_addr} \
    --master_port=2500 \
    ./bat_main.py \
    --num_nodes=${num_nodes} \
    --node_rank=${node_rank} \
    --num_gpus_per_node=${num_gpus_per_node_phase_1} \
    --epochs=${epochs} \
    --batch_per_gpu=${batch_per_gpu} \
    --data_root=${data_root} \
    --model_name=${model_name} \
    --n_classifiers=${n_classifiers} \
    --reg=${reg} \
    --eps_attack=${eps_attack} \
    --iter_attack=${iter_attack} \
    --n_restarts=${n_restarts} \
    --loss=${loss} \
    --result_file=${result_file} \
    --lr_reg=${lr_reg} \
    --attack=${attack} \
    --ckpt_run=${ckpt_run} \
    --dataset=${dataset} \
    --phase=1

# python3 -m torch.distributed.launch --nproc_per_node=${num_gpus_per_node} \
torchrun --nproc_per_node=${num_gpus_per_node} \
    --nnodes=${num_nodes} \
    --node_rank=${node_rank} \
    --master_addr=${master_addr} \
    --master_port=2500 \
    ./bat_main.py \
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
    --n_restarts=${n_restarts} \
    --loss=${loss} \
    --result_file=${result_file} \
    --lr_reg=${lr_reg} \
    --attack=${attack} \
    --ckpt_run=${ckpt_run} \
    --dataset=${dataset} \
    --phase=2

# python3 -m torch.distributed.launch --nproc_per_node=${num_gpus_per_node} \
torchrun --nproc_per_node=${num_gpus_per_node} \
    --nnodes=${num_nodes} \
    --node_rank=${node_rank} \
    --master_addr=${master_addr} \
    --master_port=2500 \
    ./bat_main.py \
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
    --n_restarts=${n_restarts} \
    --loss=${loss} \
    --result_file=${result_file} \
    --lr_reg=${lr_reg} \
    --attack=${attack} \
    --ckpt_run=${ckpt_run} \
    --dataset=${dataset} \
    --phase=3
