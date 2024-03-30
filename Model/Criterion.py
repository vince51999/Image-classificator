import numpy as np
import torch
import torch.nn as nn


class Criterion:
    """
    The criterion class that is used to calculate the loss based on the number of classes in the dataset.
    
    Attributes:
        criterion (nn.Module): The criterion to calculate the loss.
        num_classes (int): The number of classes in the dataset.
        DEVICE (_type_): The device where the model is trained.
        
    Methods:
        step(conf_matrix): Update the classes weights based on the confusion matrix.
    """

    def __init__(self, num_classes: int, DEVICE):
        """
        Initialize the criterion based on the number of classes.

        Args:
            num_classes (int): The number of classes in the dataset.
            DEVICE (_type_): The device where the model is trained.
        """
        self.criterion = None
        self.num_classes = num_classes
        self.DEVICE = DEVICE
        if self.num_classes == 1:
            self.criterion = nn.BCELoss()
            print(f"Criterion: BCELoss")
        else:
            class_weights = torch.tensor([1.0 for _ in range(self.num_classes)])
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.DEVICE))
            print(f"Criterion: CrossEntropyLoss")

    def step(self, conf_matrix: np.ndarray) -> None:
        """
        Update the classes weights based on the confusion matrix.
        If the number of classes is 1, the function returns without doing anything.
        Args:
            conf_matrix (np.ndarray): The confusion matrix representing the classification results.

        Returns:
            None
        """
        if self.num_classes == 1:
            return
        samples_per_class = conf_matrix.sum(axis=1)
        diag = np.diag(conf_matrix)
        diag = np.where(diag == 0, 1, diag)
        print("Diag: ", diag)
        # Compute the weight adjustment based on confusion matrix
        # We use **2 to give more weight to the classes with fewer true positives
        weights_adjustment = (samples_per_class / diag) ** 2
        new_weights = weights_adjustment / weights_adjustment.sum()
        print("Updated classes weights: ", new_weights)

        new_weights = new_weights.to(self.DEVICE)
        self.criterion.weight = new_weights
