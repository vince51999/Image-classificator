import torch
from typing import List


class Statistics:
    def __init__(self, classes: List[int], sample_class: int):
        self.correct = 0.0
        self.classes = classes
        self.sample_class = sample_class
        self.conf_matrix = torch.zeros(200, 200)

    def update(self, preds, labels):
        self.correct += Statistics.__get_num_correct(preds, labels)
        _, preds = torch.max(preds, 1)
        for t, p in zip(labels.view(-1), preds.view(-1)):
            self.conf_matrix[t.long(), p.long()] += 1

    def accuracy(self):
        return self.correct / (self.sample_class * len(self.classes))

    def classes_recall(self):
        # recall = TP / (TP + FN) for one class
        return self.conf_matrix.diag() / self.conf_matrix.sum(1)

    def classes_precision(self):
        # precision = TP / (TP + FP) for one class
        return self.conf_matrix.diag() / self.conf_matrix.sum(0)

    def confusion_matrix(self):
        return self.conf_matrix

    def __get_num_correct(preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()
