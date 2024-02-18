#!/bin/bash
#SBATCH --job-name=TinyImageNetResNet
#SBATCH --output=%x.oR
#SBATCH --error=%x.eR
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:p100:1
#SBATCH --qos=gpu
#SBATCH --time 01:00:00
#SBATCH --mem=32gb
#SBATCH --ntasks-per-node 10

##SBATCH --account=vincenzo.barbato

echo $SLURM_JOB_NODELIST

echo  #OMP_NUM_THREADS : $OMP_NUM_THREADS

architecture="resnet50"
cls=0
num_classes=10
num_epochs=25
eval_batch_size=1
train_batch_size=9
tolerance=3
min_delta=0.5
lr=0.001
momentum=0.0
weight_decay=0
dropout=0.2

module load miniconda3
source "$CONDA_PREFIX/etc/profile.d/conda.sh"
conda activate myenv

python ./main.py --architecture $architecture --cls $cls --num_classes $num_classes --num_epochs $num_epochs --eval_batch_size $eval_batch_size --train_batch_size $train_batch_size --tolerance $tolerance --min_delta $min_delta --lr $lr --momentum $momentum --weight_decay $weight_decay --dropout $dropout

conda deactivate