#!/bin/bash
architecture="resnet18"
cls=(0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.0)
num_classes=10
num_epochs=100
eval_batch_size=128
train_batch_size=128
gamma_train_batch_size=2
tolerance=8
min_delta=0.5
lr=0.001
gamma_lr=0.3
momentum=0.9
weight_decay=0.00001
dropout_rate_bb=0.1
dropout_rate_fc=0.2
fine_tune=0
transfer_learning=1
is_test=1
increases_trainset=3
step=0

python ./main.py --architecture $architecture --cls $cls --num_classes $num_classes --num_epochs $num_epochs --eval_batch_size $eval_batch_size --train_batch_size $train_batch_size --gamma_train_batch_size $gamma_train_batch_size --tolerance $tolerance --min_delta $min_delta --lr $lr --gamma_lr $gamma_lr --momentum $momentum --weight_decay $weight_decay --dropout_rate_bb $dropout_rate_bb --dropout_rate_fc $dropout_rate_fc --fine_tune $fine_tune --transfer_learning $transfer_learning --test $is_test --increases_trainset $increases_trainset --step $step
