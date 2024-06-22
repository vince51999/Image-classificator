#!/bin/bash
#SBATCH --job-name=TinyImageNetResNet1
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

# Architecture of the model
# resnet18, resnet50, resnet101
architecture="resnet18"
# If fine_tune=0, transfer_learning=0, the model is trained from scratch
# If fine_tune=1, transfer_learning=1, the model is fine-tuned
fine_tune=0
transfer_learning=0
# Position of dropout layer in the residual block
# 0: after each residual block
# 1: before each ReLU
# 2: after each ReLU
dropout_pos_rb=0
# Dropout rate of dropout layer in the residual block, 0.0 means no dropout
dropout_rate_rb=0.5
# Position of dropout layer in the head
# 0: before avgpool
# 1: after avgpool
dropout_pos_fc=1
# Dropout rate of dropout layer in the head, 0.0 means no dropout
dropout_rate_fc=0.5

# cls is the classes that we want in the classification task
# num_classes is the number of classes in the classification task, if cls length is < than num_classes, the other classes are selected randomly
# test is the flag to test the model on a predefined classes (10 classes from Tiny ImageNet dataset)
cls=(0)
num_classes=10
is_test=1

# Parameters for early stopping
# tolerance minimum value 1
# min_delta minimum value 0.0
tolerance=3
min_delta=0.01

# Parameters for training, if momentum=0.0, the optimizer is Adam else the optimizer is SGD
num_epochs=70
lr=0.1
eval_batch_size=4
train_batch_size=32
momentum=0.9
weight_decay=0.0

# Parameters for schedulers
gamma_lr=1.0
gamma_train_batch_size=3.0
step=14
# lr_scheduler: StepLR, CosineAnnealingWarmRestarts
lr_scheduler="StepLR"

# Parameters for data augmentation, the minimum size of the image is 64x64
increases_trainset=3
image_size=128
# if online_aug_step_size=0, the augumentation is appliend only at the start of the training
online_aug_step_size=2

# Load the model from the checkpoint
# checkpoint is the path to the checkpoint
# none: train the model from scratch
checkpoint="none"

module load miniconda3
source "$CONDA_PREFIX/etc/profile.d/conda.sh"
conda activate myenv

python ./main.py --architecture "$architecture" --cls "$cls" --num_classes "$num_classes" --num_epochs "$num_epochs" --eval_batch_size "$eval_batch_size" --train_batch_size "$train_batch_size" --gamma_train_batch_size "$gamma_train_batch_size" --tolerance "$tolerance" --min_delta "$min_delta" --lr "$lr" --gamma_lr "$gamma_lr" --momentum "$momentum" --weight_decay "$weight_decay" --dropout_rate_rb "$dropout_rate_rb" --dropout_rate_fc "$dropout_rate_fc" --fine_tune "$fine_tune" --transfer_learning "$transfer_learning" --test "$is_test" --increases_trainset "$increases_trainset" --image_size "$image_size" --step "$step" --dropout_pos_rb "$dropout_pos_rb" --dropout_pos_fc "$dropout_pos_fc" --checkpoint "$checkpoint" --lr_scheduler "$lr_scheduler" --online_aug_step_size "$online_aug_step_size"

conda deactivate
