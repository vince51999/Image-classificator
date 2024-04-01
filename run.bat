@echo off
SET architecture=resnet18
SET cls=0
SET num_classes=3
SET num_epochs=3
SET eval_batch_size=16
SET train_batch_size=16
SET gamma_train_batch_size=2
SET tolerance=6
SET min_delta=0.05
SET lr=0.01
SET gamma_lr=0.2
SET momentum=0.9
SET weight_decay=0.0005
SET dropout_rate_bb=0.3
SET dropout_rate_fc=0.5
SET fine_tune=1
SET transfer_learning=1
SET is_test=0
SET increases_trainset=0
SET step=2

python "%CD%\main.py" "--architecture" "%architecture%" "--cls" "%cls%" "--num_classes" "%num_classes%" "--num_epochs" "%num_epochs%" "--eval_batch_size" "%eval_batch_size%" "--train_batch_size" "%train_batch_size%" "--gamma_train_batch_size" "%gamma_train_batch_size%" "--tolerance" "%tolerance%" "--min_delta" "%min_delta%" "--lr" "%lr%" "--gamma_lr" "%gamma_lr%" "--momentum" "%momentum%" "--weight_decay" "%weight_decay%" "--dropout_rate_bb" "%dropout_rate_bb%" "--dropout_rate_fc" "%dropout_rate_fc%" "--fine_tune" "%fine_tune%" "--transfer_learning" "%transfer_learning%" "--test" "%is_test%" "--increases_trainset" "%increases_trainset%" "--step" "%step%"