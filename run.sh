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

cls=0
num_classes=10
train_batch_size=9
eval_batch_size=1
num_epochs=22

module load miniconda3
source "$CONDA_PREFIX/etc/profile.d/conda.sh"
conda activate myenv

rm -r ./*.eR
rm -r ./*.oR

python ./main.py --cls $cls --num_classes $num_classes --train_batch_size $train_batch_size --eval_batch_size $eval_batch_size --num_epochs $num_epochs

conda deactivate