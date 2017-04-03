# About

Network In Network by chainer

# Paper

[Network In Network](https://arxiv.org/abs/1312.4400)

# Model

Network In Network

# How to run

git clone git@github.com:amazarashi/network_in_network.git

cd ./network_in_network

python main.py -g 1

# Inspection

### dataset
Cifar10 [(link)](https://www.cs.toronto.edu/~kriz/cifar.html)

### Result

|             | accuracy(%) |
|:-----------:|:------------:|
| paper       |        91.10 |
| my experiment |        90.97 |


※For my experiment, i adjust optimizer's learning rate to 0.07
and scheduled on 150-epoch and 225-epoch to mutiply 10%  to learning rate

optimizer: MomentumSGD
  - weight decay : 1.0e-4
  - momentum : 0.9
  - schedule[default:0.07,150:0.007,225:0.0007]


![accuracy](https://github.com/amazarashi/network_in_network/blob/develop/result/accuracy.png "accuracy")

![loss](https://github.com/amazarashi/network_in_network/blob/develop/result/loss.png "loss")
