from typing import NamedTuple
from typing import List as List


class ClusterConfig(NamedTuple):
    dist_backend: str  # 'nccl'
    dist_url: str  # for pytorch.distributed shared file-system initialization
    num_nodes: int  # number of machines
    num_gpus_per_node: int  # number of gpus per node


class TrainerConfig(NamedTuple):
    epochs: int  # the total number of epochs (200 * n_classifiers)
    batch_per_gpu: int  # split a minibatch to multiple GPUs (128)
    data_root: str  # where the data and result files are stored
    models: List[str]  # names of models (length: n_classifiers)
    reg: float  # regularization parameter (1e-3)
    eps_attack: float  # the adversarial level \epsilon (l_infty: 8/255)
    iter_attack: int  # the number of iterations of the attack (20)
    loss: str  # ce loss or dlr loss for attack
    result_file: str  # file to record each epoch's accuracy
    lr_reg: float  # the mixing weight's learning rate
    attack: str  # which attack to use, "APGD" or "PGD" or "TRADES"
    ckpt_run: str  # the name of the folder that stores checkpoints
    dataset: str  # 'CIFAR10' or 'CIFAR100' 
    n_restarts: int = 3  # number of restarts, each run returns one attack point (5)
    size_classifier_buffer: int = 20  # the size of the classifier buff
    noise_level_attack: float = 0.001  # noise level of the randomized attack

class JobEnvironment(NamedTuple):
    local_rank: int  # the local device id
    num_tasks: int  # the number of processes in total
    global_rank: int  # global_rank: the process id
    # job_id: int  # job_id refers to different settings/datasets to run
