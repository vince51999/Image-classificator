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
num_epochs=50
eval_batch_size=32
train_batch_size=32
gamma_train_batch_size=1
tolerance=2
min_delta=0.1
lr=0.001
gamma_lr=1
momentum=0.9
weight_decay=0.0
dropout_rate_bb=0.0
dropout_rate_fc=0.0
fine_tune=0
transfer_learning=1
is_test=1
increases_trainset=5
image_size=112
step=1

module load miniconda3
source "$CONDA_PREFIX/etc/profile.d/conda.sh"
conda activate myenv

python ./main.py --architecture $architecture --cls $cls --num_classes $num_classes --num_epochs $num_epochs --eval_batch_size $eval_batch_size --train_batch_size $train_batch_size --gamma_train_batch_size $gamma_train_batch_size --tolerance $tolerance --min_delta $min_delta --lr $lr --gamma_lr $gamma_lr --momentum $momentum --weight_decay $weight_decay --dropout_rate_bb $dropout_rate_bb --dropout_rate_fc $dropout_rate_fc --fine_tune $fine_tune --transfer_learning $transfer_learning --test $is_test --increases_trainset $increases_trainset --image_size $image_size --step $step

conda deactivate