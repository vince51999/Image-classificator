@echo off

REM architecture of the model
REM resnet18, resnet50, resnet101
SET architecture=resnet18
REM if fine_tune=0, transfer_learning=0, the model is trained from scratch
REM if fine_tune=1, transfer_learning=1, the model is fine-tuned
SET fine_tune=0
SET transfer_learning=0
REM position of dropout layer in the residual block
REM 0: after each residual block
REM 1: before each ReLU
REM 2: after each ReLU
SET dropout_pos_rb=0
REM dropout rate of dropout layer in the residual block, 0.0 means no dropout
SET dropout_rate_rb=0.5
REM position of dropout layer in the head
REM 0: before avgpool
REM 1: after avgpool
SET dropout_pos_fc=0
REM dropout rate of dropout layer in the head, 0.0 means no dropout
SET dropout_rate_fc=0.7

REM cls is the class that we want in the classification task
REM test is the flag to test the model on a pre-defined classes (10 classes from Tiny ImageNet dataset)
SET cls=0
SET num_classes=2
SET is_test=0

REM parameters for early stopping
REM tolerance minimum value 1
REM min_delta minimum value 0.0
SET tolerance=2
SET min_delta=0.05

REM parameters for training, if momentum=0.0, the optimizer is Adam else the optimizer is SGD
SET num_epochs=10
SET lr=0.001
SET eval_batch_size=4
SET train_batch_size=16
SET momentum=0.0
SET weight_decay=0.0005

REM parameters for schedulers
SET gamma_lr=0.1
SET gamma_train_batch_size=2
SET step=2
REM lr_scheduler: StepLR, CosineAnnealingWarmRestarts
SET lr_scheduler="CosineAnnealingWarmRestarts"

REM parameters for data augmentation, the minimum size of the image is 64x64
SET increases_trainset=0
SET image_size=64
REM if online_aug_step_size=0, the augumentation is appliend only at the start of the training
SET online_aug_step_size=2

REM load the model from the checkpoint
REM checkpoint is the path to the checkpoint
REM none: train the model from scratch
SET checkpoint="none"


python "%CD%\main.py" "--architecture" "%architecture%" "--cls" "%cls%" "--num_classes" "%num_classes%" "--num_epochs" "%num_epochs%" "--eval_batch_size" "%eval_batch_size%" "--train_batch_size" "%train_batch_size%" "--gamma_train_batch_size" "%gamma_train_batch_size%" "--tolerance" "%tolerance%" "--min_delta" "%min_delta%" "--lr" "%lr%" "--gamma_lr" "%gamma_lr%" "--momentum" "%momentum%" "--weight_decay" "%weight_decay%" "--dropout_rate_rb" "%dropout_rate_rb%" "--dropout_rate_fc" "%dropout_rate_fc%" "--fine_tune" "%fine_tune%" "--transfer_learning" "%transfer_learning%" "--test" "%is_test%" "--increases_trainset" "%increases_trainset%" "--image_size" "%image_size%" "--step" "%step%" "--dropout_pos_rb" "%dropout_pos_rb%" "--dropout_pos_fc" "%dropout_pos_fc%" "--checkpoint" "%checkpoint%" "--lr_scheduler" "%lr_scheduler%" "--online_aug_step_size" "%online_aug_step_size%"