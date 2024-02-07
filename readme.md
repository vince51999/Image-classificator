# Image classificator based on resnet and Tiny ImageNet

Description

## Setup environment

To run this project I have used the unipr HPC and I have created personal environment with conda.


```
ssh wn44
```
Load conda:
```
module load miniconda3
source "$CONDA_PREFIX/etc/profile.d/conda.sh"
```

Create environment :
```
conda create -n myenv python=3.9
```
Remember to update pip before installing packages using 
```
pip install --upgrade pip
```
## Install packages

Install pyTorch
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

Install tinyImageNet
```
pip install tinyimagenet
```
