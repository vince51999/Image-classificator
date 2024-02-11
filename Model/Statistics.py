import torch
from typing import List


class Statistics:
    def __init__(self, classes: List[int], sample_class: int):
        self.classes = classes
        self.sample_class = sample_class
        self.conf_matrix = torch.zeros(len(classes), len(classes))

    def update(self, preds, labels):
        _, preds = torch.max(preds, 1)
        for t, p in zip(labels.view(-1), preds.view(-1)):
            self.conf_matrix[t.long(), p.long()] += 1

    def reset(self):
        self.conf_matrix = torch.zeros(len(self.classes), len(self.classes))

    def accuracy(self):
        # accuracy = (TP + TN) / (TP + TN + FP + FN)
        return self.conf_matrix.diag().sum() / self.conf_matrix.sum()

    def classes_recall(self):
        # recall = TP / (TP + FN) for one class
        return self.conf_matrix.diag() / self.conf_matrix.sum(1)

    def classes_precision(self):
        # precision = TP / (TP + FP) for one class
        return self.conf_matrix.diag() / self.conf_matrix.sum(0)

    # We calc recall and precision for all classes with macro-averaging because each class have the same number of samples
    def recall(self):
        return self.classes_recall().sum() / len(self.classes)

    def precision(self):
        return self.classes_precision().sum() / len(self.classes)
    def confusion_matrix(self):
        return self.conf_matrix
