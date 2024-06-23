import torch
from typing import List

from Model.Results import Results as Res


class Statistics:
    """
    A class to calculate and store statistics for a classification model.

    Attributes:
        num_classes (int): The number of classes.
        classes (List[int]): The list of class labels.
        epochs (List[int]): The list of epochs.
        losses (List[float]): The list of losses.
        accuracy (List[float]): The list of accuracies.
        f_measure (List[float]): The list of F-measures.
        recall (List[float]): The list of recalls.
        precision (List[float]): The list of precisions.
        conf_matrix (torch.Tensor): The confusion matrix.
        res (Res): The results class to print the statistics.


    Methods:
        update(preds, labels): Update the confusion matrix with the predictions and labels.
        step(epoch, loss, name, verbose, log): Update the statistics and print them.
        print(name): Print the statistics.
        log(name, epoch): Log the statistics.
        reset(): Reset the confusion matrix.
        get_accuracy(): Calculate the accuracy.
        get_classes_recall(): Calculate the recall for each class.
        get_classes_precision(): Calculate the precision for each class.
        get_recall(): Calculate the recall.
        get_precision(): Calculate the precision.
        get_f_measure(): Calculate the F-measure.
        get_confusion_matrix(): Get the confusion matrix.
        get_classes(): Get the list of classes.
    """

    def __init__(self, classes: List[int], res: Res):
        """
        Initializes the statistics class.

        Args:
            classes (List[int]): The list of class labels.
            res (Res): The results class to print the statistics.
        """
        self.num_classes = len(classes)
        if self.num_classes == 1:
            self.num_classes += 1
        self.classes = classes
        self.epochs = []
        self.losses = []
        self.accuracy = []
        self.f_measure = []
        self.recall = []
        self.precision = []
        self.conf_matrix = torch.zeros(self.num_classes, self.num_classes)
        self.res = res

    def update(self, preds, labels):
        if len(self.classes) == 1:
            preds = (preds > 0.5).float()
            for p, l in zip(preds, labels):
                p = bool(p)
                l = l > 0.5
                self.conf_matrix[int(not l), int(not p)] += 1
        else:
            _, preds = torch.max(preds, 1)
            for t, p in zip(labels.view(-1), preds.view(-1)):
                self.conf_matrix[t.long(), p.long()] += 1

    def step(
        self, epoch: int, loss: float, name: str, verbose: bool = True, log: bool = True
    ):
        self.epochs.append(epoch)
        self.losses.append(loss)
        accuracy = self.get_accuracy()
        self.accuracy.append(accuracy)
        f_measure = self.get_f_measure()
        self.f_measure.append(f_measure)
        recall = self.get_recall()
        self.recall.append(recall)
        precision = self.get_precision()
        self.precision.append(precision)
        if verbose:
            self.print(name)
        if log:
            self.log(name, epoch)

    def print(self, name: str):
        l = len(self.epochs) - 1
        self.res.print(f"{name}")
        self.res.print(f"Loss: {self.losses[l]:.3f}")
        self.res.print(f"Accuracy: {self.accuracy[l]:.3f}")
        self.res.print(f"F-Measure: {self.f_measure[l]:.3f}")
        self.res.print(f"Recall: {self.recall[l]:.3f}")
        self.res.print(f"Precision: {self.precision[l]:.3f}\n")

    def log(self, name: str, epoch: int):
        l = len(self.epochs) - 1
        self.res.addScalar(f"Loss", name, self.losses[l], epoch)
        self.res.addScalar(f"Accuracy", name, self.accuracy[l], epoch)
        self.res.addScalar(f"F-Measure", name, self.f_measure[l], epoch)
        self.res.addScalar(f"Recall", name, self.recall[l], epoch)
        self.res.addScalar(f"Precision", name, self.precision[l], epoch)

    def reset(self):
        self.conf_matrix = torch.zeros(self.num_classes, self.num_classes)

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
        return self.get_classes_recall().sum() / self.num_classes

    def get_precision(self):
        return self.get_classes_precision().sum() / self.num_classes

    def get_f_measure(self):
        precision = self.get_precision()
        recall = self.get_recall()
        return 2 * (precision * recall) / (precision + recall)

    def get_confusion_matrix(self):
        return self.conf_matrix

    def get_classes(self):
        return self.classes
