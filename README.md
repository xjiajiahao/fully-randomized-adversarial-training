# Dependencies
# Dependencies
This project depends on the following python packages:
1. pytorch 1.10.2+
2. torchvision 0.11.3+
3. numpy 1.19.2+
3. scipy 1.5.2+
5. matplotlib 3.3.2+


# Synthetic Experiment

1. Go to the directory to `./toy/`.

2. In terminal, type the following commands.
``` bash
python sat_toy.py
python bat_toy.py
python oracle_toy.py
python regularized_toy.py
python atm_toy.py
python frat_toy.py
```

The results can be found in `./toy/data/`.


# CIFAR-10 and CIFAR-100 Experiment

This experiment is conducted on a machine with 4 GPUs.
To test the algorithms on CIFAR-10, type the following commands in terminal
``` bash
bash ./test_script/test_sat_cifar.sh nic_name 0
bash ./test_script/test_bat_cifar.sh nic_name 0
bash ./test_script/test_atm_cifar.sh nic_name 0  # 2 models by default
bash ./test_script/test_frat_cifar.sh nic_name 0  # 2 models by default
```
Here, `nic_name` denotes the network interface card name.

To run the algorithms on CIFAR-100, change the variable `dataset` in the scripts from `CIFAR10` to `CIFAR100`.
To test ATM (resp. FRAT) with different sizes of the mixture, change the `n_classifiers` variable in `./test_script/test_atm_cifar.sh` (resp., `./test_script/test_frat_cifar.sh`) from `2` to other numbers, e.g., `3` or `4`.
