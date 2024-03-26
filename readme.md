
# Image classificator based on resnet and Tiny ImageNet

 Image classificator based on Tiny InageNet and resNet architecture
 
  ##  Import project
 If you have ssh key access you can clone project with:
```
git clone git@github.com:vince51999/Image-classificator.git
```
 
 ## Setup environment
 
To run this project I have used the unipr HPC and I have created personal environment with conda
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
## Install packages

Go on your enviroment
```
source activate myenv
```
Update pip before installing packages using
```
pip install --upgrade pip
```
Install pyTorch
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```
Install torchvision-tinyimagenet
```
pip install tinyimagenet
```
Install matplotlib
```
pip install matplotlib
```
Install pandas
```
pip install pandas
```
Install seaborn
```
pip install seaborn
```

## Run application

Run application
```
sbatch run.sh
```