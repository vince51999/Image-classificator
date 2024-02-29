#!/bin/bash
architecture="resnet18"
cls=0
num_classes=10
num_epochs=100
eval_batch_size=1
train_batch_size=5
tolerance=8
min_delta=0.5
lr=0.001
momentum=0.0
weight_decay=0.001
dropout_rate_bb=0.0
dropout_rate_fc=0.0
pretrained=1
is_test=1
increases_trainset=0

python ./main.py --architecture $architecture --cls $cls --num_classes $num_classes --num_epochs $num_epochs --eval_batch_size $eval_batch_size --train_batch_size $train_batch_size --tolerance $tolerance --min_delta $min_delta --lr $lr --momentum $momentum --weight_decay $weight_decay --dropout_rate_bb $dropout_rate_bb --dropout_rate_fc $dropout_rate_fc --pretrained $pretrained --test $is_test --increases_trainset $increases_trainset
