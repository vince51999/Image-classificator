#!/bin/bash

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
dropout_pos_fc=0
# Dropout rate of dropout layer in the head, 0.0 means no dropout
dropout_rate_fc=0.7

# cls is the class that we want in the classification task
# test is the flag to test the model on a predefined classes (10 classes from Tiny ImageNet dataset)
cls=0
num_classes=2
is_test=0

# Parameters for early stopping
# tolerance minimum value 1
# min_delta minimum value 0.0
tolerance=2
min_delta=0.05

# Parameters for training, if momentum=0.0, the optimizer is Adam else the optimizer is SGD
num_epochs=10
lr=0.001
eval_batch_size=4
train_batch_size=16
momentum=0.0
weight_decay=0.0005

# Parameters for schedulers
gamma_lr=0.1
gamma_train_batch_size=2
step=2
# lr_scheduler: StepLR, CosineAnnealingWarmRestarts
lr_scheduler="CosineAnnealingWarmRestarts"

# Parameters for data augmentation, the minimum size of the image is 64x64
increases_trainset=0
image_size=64
# if online_aug_step_size=0, the augumentation is appliend only at the start of the training
online_aug_step_size=0

# Load the model from the checkpoint
# checkpoint is the path to the checkpoint
# none: train the model from scratch
checkpoint="none"

python ./main.py --architecture "$architecture" --cls "$cls" --num_classes "$num_classes" --num_epochs "$num_epochs" --eval_batch_size "$eval_batch_size" --train_batch_size "$train_batch_size" --gamma_train_batch_size "$gamma_train_batch_size" --tolerance "$tolerance" --min_delta "$min_delta" --lr "$lr" --gamma_lr "$gamma_lr" --momentum "$momentum" --weight_decay "$weight_decay" --dropout_rate_rb "$dropout_rate_rb" --dropout_rate_fc "$dropout_rate_fc" --fine_tune "$fine_tune" --transfer_learning "$transfer_learning" --test "$is_test" --increases_trainset "$increases_trainset" --image_size "$image_size" --step "$step" --dropout_pos_rb "$dropout_pos_rb" --dropout_pos_fc "$dropout_pos_fc" --checkpoint "$checkpoint" --lr_scheduler "$lr_scheduler" --online_aug_step_size "$online_aug_step_size"
