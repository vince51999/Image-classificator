import torch
from typing import List


class Statistics:
    def __init__(self, classes: List[int], sample_class: int):
        self.classes = classes
        self.sample_class = sample_class
        self.epochs = []
        self.losses = []
        self.accuracy = []
        self.f_measure = []
        self.recall = []
        self.precision = []
        self.conf_matrix = torch.zeros(len(classes), len(classes))

    def update(self, preds, labels):
        _, preds = torch.max(preds, 1)
        for t, p in zip(labels.view(-1), preds.view(-1)):
            self.conf_matrix[t.long(), p.long()] += 1

    def save_epoch(self, epoch: int, loss: float):
        self.epochs.append(epoch)
        self.losses.append(loss)
        self.accuracy.append(self.get_accuracy())
        self.f_measure.append(self.get_f_measure())
        self.recall.append(self.get_recall())
        self.precision.append(self.get_precision())

    def print(self, str: str):
        print(f"------- {str} -------")
        print(f"------- Accuracy: {self.get_accuracy():.3f}")
        print(f"------- F-Measure: {self.get_f_measure():.3f}")
        print(f"------- Recall: {self.get_recall():.3f}")
        print(f"------- Precision: {self.get_precision():.3f}\n")

    def reset(self):
        self.conf_matrix = torch.zeros(len(self.classes), len(self.classes))

    def get_accuracy(self):
        # accuracy = (TP + TN) / (TP + TN + FP + FN)
        return self.conf_matrix.diag().sum() / self.conf_matrix.sum()

    def get_classes_recall(self):
        # recall = TP / (TP + FN) for one class
        return self.conf_matrix.diag() / self.conf_matrix.sum(1)

    def get_classes_precision(self):
        # precision = TP / (TP + FP) for one class
        return self.conf_matrix.diag() / self.conf_matrix.sum(0)

    # We calc recall and precision for all classes with macro-averaging because each class have the same number of samples
    def get_recall(self):
        return self.get_classes_recall().sum() / len(self.classes)

    def get_precision(self):
        return self.get_classes_precision().sum() / len(self.classes)

    def get_f_measure(self):
        precision = self.get_precision()
        recall = self.get_recall()
        return 2 * (precision * recall) / (precision + recall)

    def get_confusion_matrix(self):
        return self.conf_matrix
