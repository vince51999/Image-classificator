import torch
from typing import List

from Model.Results import Results as Res


class Statistics:
    """
    A class to calculate and store statistics for a classification model.

    Attributes:
        classes (List[int]): The list of class labels.
        sample_class (int): The class label of the sample.
        epochs (List[int]): The list of epochs.
        losses (List[float]): The list of loss values.
        accuracy (List[float]): The list of accuracy values.
        f_measure (List[float]): The list of F-measure values.
        recall (List[float]): The list of recall values.
        precision (List[float]): The list of precision values.
        conf_matrix (torch.Tensor): The confusion matrix.

    Methods:
        update(preds, labels): Updates the confusion matrix based on the predicted and true labels.
        save_epoch(epoch, loss): Saves the epoch and loss values.
        print(str): Prints the statistics.
        reset(): Resets the confusion matrix.
        get_accuracy(): Calculates the accuracy.
        get_classes_recall(): Calculates the recall for each class.
        get_classes_precision(): Calculates the precision for each class.
        get_recall(): Calculates the macro-averaged recall.
        get_precision(): Calculates the macro-averaged precision.
        get_f_measure(): Calculates the F-measure.
        get_confusion_matrix(): Returns the confusion matrix.
        get_classes(): Returns the list of class labels.
    """

    def __init__(self, classes: List[int], sample_class: int, res: Res):
        self.classes = classes
        self.sample_class = sample_class
        self.epochs = []
        self.losses = []
        self.accuracy = []
        self.f_measure = []
        self.recall = []
        self.precision = []
        self.conf_matrix = torch.zeros(len(classes), len(classes))
        self.res = res

    def update(self, preds, labels):
        if len(self.classes) == 1:
            preds = (preds > 0.5).float()
            self.conf_matrix[0, 0] += torch.sum(preds == True).item()
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
        if len(self.classes) > 1:
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
        if len(self.classes) > 1:
            self.res.print(f"F-Measure: {self.f_measure[l]:.3f}")
            self.res.print(f"Recall: {self.recall[l]:.3f}")
            self.res.print(f"Precision: {self.precision[l]:.3f}\n")

    def log(self, name: str, epoch: int):
        l = len(self.epochs) - 1
        self.res.addScalar(f"Loss", name, self.losses[l], epoch)
        self.res.addScalar(f"Accuracy", name, self.accuracy[l], epoch)
        if len(self.classes) > 1:
            self.res.addScalar(f"F-Measure", name, self.f_measure[l], epoch)
            self.res.addScalar(f"Recall", name, self.recall[l], epoch)
            self.res.addScalar(f"Precision", name, self.precision[l], epoch)

    def reset(self):
        self.conf_matrix = torch.zeros(len(self.classes), len(self.classes))

    def get_accuracy(self):
        if len(self.classes) == 1:
            self.res.print(self.conf_matrix[0, 0], self.sample_class)
            return self.conf_matrix[0, 0] / self.sample_class
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

    def get_classes(self):
        return self.classes
