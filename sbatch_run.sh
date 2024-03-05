#!/bin/bash
#SBATCH --job-name=TinyImageNetResNet
#SBATCH --output=%x.oR
#SBATCH --error=%x.eR
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:p100:1
#SBATCH --qos=gpu
#SBATCH --time 01:30:00
#SBATCH --mem=32gb
#SBATCH --ntasks-per-node 10

##SBATCH --account=vincenzo.barbato

echo $SLURM_JOB_NODELIST

echo  #OMP_NUM_THREADS : $OMP_NUM_THREADS

architecture="resnet18"
cls=0
num_classes=10
num_epochs=100
eval_batch_size=1
train_batch_size=5
tolerance=8
min_delta=0.5
gamma_train_batch_size=2
lr=0.001
gamma_lr=0.3
momentum=0.0
weight_decay=0.001
dropout_rate_bb=0.0
dropout_rate_fc=0.0
pretrained=1
is_test=1
increases_trainset=0
step=0

module load miniconda3
source "$CONDA_PREFIX/etc/profile.d/conda.sh"
conda activate myenv

python ./main.py --architecture $architecture --cls $cls --num_classes $num_classes --num_epochs $num_epochs --eval_batch_size $eval_batch_size --train_batch_size $train_batch_size --gamma_train_batch_size $gamma_train_batch_size --tolerance $tolerance --min_delta $min_delta --lr $lr --gamma_lr $gamma_lr --momentum $momentum --weight_decay $weight_decay --dropout_rate_bb $dropout_rate_bb --dropout_rate_fc $dropout_rate_fc --pretrained $pretrained --test $is_test --increases_trainset $increases_trainset --step $step

conda deactivate