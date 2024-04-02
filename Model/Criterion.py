import numpy as np
import torch
import torch.nn as nn


class Criterion:
    """
    The criterion class that is used to calculate the loss based on the number of classes in the dataset.

    Attributes:
        criterion (nn.Module): The criterion to calculate the loss.
        num_classes (int): The number of classes in the dataset.
        
    Methods:
        step(conf_matrix, verbose): Update the classes weights based on the confusion matrix.
    """

    def __init__(self, num_classes: int):
        """
        Initialize the criterion based on the number of classes.

        Args:
            num_classes (int): The number of classes in the dataset.
            DEVICE (_type_): The device where the model is trained.
            step_size (int): The step size to update the classes weights.
        """
        self.criterion = None
        self.num_classes = num_classes
        if self.num_classes == 1:
            self.criterion = nn.BCELoss()
            print(f"Criterion: BCELoss")
        else:
            self.criterion = nn.CrossEntropyLoss()
            print(f"Criterion: CrossEntropyLoss")
