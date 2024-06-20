# Image classificator based on resnet and Tiny ImageNet

**Author:** Vincenzo Barbato - 345728

**Place:** University of Parma

**Date:** 2024

Image classificator based on Tiny ImageNet and resNet architecture.

##  Import project
 If you have ssh key access you can clone project with:
```
git clone git@github.com:vince51999/Image-classificator.git
```

## Studies
In the ```notes``` folder is possible see all the studies make with this project. The file ```studies.ipynb``` is the master file with studies and tensorboard command to see the training results. The command to run tensorboard are write for semplify the visualization. To prevent setup problem is suggested open a terminal in the ```notes``` folder and run the command that you prefer.

In each tensorboard logs is possible see:

Scalar:
- Loss
- F-Measure
- Accuracy
- Recall
- Precision

Graph:
- Model architecture

Images:
- Simple example of training images
- Confusion matrix of the best results

In each results folder there is an ```output.txt``` with training print and the ```sbatch_run.sh``` file with the parameters used during the train.

## Configuration file and project setup
```Files.sh``` and ```file.bat``` are completly configurable to help the user to manage and done other test.
In each configuration file there is a description of each configurable parameter to improve the reusability of the project.

The Model folder contains all the files necessary for the project and recalled from the ```main.py``` file to perform various types of training, resume checkpoints and more.

The only external file is ```mergeLogs.py``` useful to merge logs of different train, for example, after training the model from a checkpoint you can merge the two training sessions into one to make it easier for the user to see.

## Setup environment
To run this project I have used the unipr HPC and I have created personal environment with conda.
If you want to use a different system, follow the steps from **Create enviroment** if you want to use Conda, or from **Install packages** if you don't want to use Conda.

### After log on HPC
```
ssh wn44
```
### Load conda
```
module load miniconda3
source "$CONDA_PREFIX/etc/profile.d/conda.sh"
```
### Create environment
```
conda create -n myenv python=3.9
```
### Activate your enviroment
```
conda activate myenv
```

## Install packages
### Update pip before installing packages using
```
pip install --upgrade pip
```
### Install pyTorch
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```
**N.B. This command has been tested on the unipr HPC, in case of installations on different systems follow** [pytorch](https://pytorch.org/)

### Install torchvision-tinyimagenet
```
pip install tinyimagenet==0.9.9
```
### Install matplotlib
```
pip install matplotlib==3.9.0
```
### Install pandas
```
pip install pandas==2.2.2
```
### Install numpy
```
pip install numpy==1.26.4
```
### Install seaborn
```
pip install seaborn==0.13.2
```
### Install tensorboard
```
pip install tensorboard==2.16.2
```
### Install tensorflow
```
pip install tensorflow==2.16.1
```
### Go out from your enviroment
```
conda deactivate
```

## Run application
If you follow the installation steps you can run the project on unipr HPC with the command below from the project folder (**remember to setup your user occount**)
```
sbatch sbatch_run.sh
```
Else in the project directory there are 2 configurable files for Windows or Linux without HPC setup.