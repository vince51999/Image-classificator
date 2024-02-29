@echo off
SET architecture=resnet18
SET cls=0
SET num_classes=10
SET num_epochs=100
SET eval_batch_size=1
SET train_batch_size=5
SET tolerance=8
SET min_delta=0.5
SET lr=0.001
SET momentum=0.0
SET weight_decay=0.001
SET dropout_rate_bb=0.0
SET dropout_rate_fc=0.0
SET pretrained=1
SET is_test=1
SET increases_trainset=0
python "%CD%\main.py" "--architecture" "%architecture%" "--cls" "%cls%" "--num_classes" "%num_classes%" "--num_epochs" "%num_epochs%" "--eval_batch_size" "%eval_batch_size%" "--train_batch_size" "%train_batch_size%" "--tolerance" "%tolerance%" "--min_delta" "%min_delta%" "--lr" "%lr%" "--momentum" "%momentum%" "--weight_decay" "%weight_decay%" "--dropout_rate_bb" "%dropout_rate_bb%" "--dropout_rate_fc" "%dropout_rate_fc%" "--pretrained" "%pretrained%" "--test" "%is_test%" "--increases_trainset" "%increases_trainset%"