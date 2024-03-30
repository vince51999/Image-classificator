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
        itr (int): The number of iterations.
        step_size (int): The step size to update the classes weights.

    Methods:
        step(conf_matrix, verbose): Update the classes weights based on the confusion matrix.
    """

    def __init__(self, num_classes: int, DEVICE, step_size: int = 0):
        """
        Initialize the criterion based on the number of classes.

        Args:
            num_classes (int): The number of classes in the dataset.
            DEVICE (_type_): The device where the model is trained.
            step_size (int): The step size to update the classes weights.
        """
        self.criterion = None
        self.num_classes = num_classes
        self.DEVICE = DEVICE
        self.itr = 0
        self.step_size = step_size
        if self.num_classes == 1:
            self.criterion = nn.BCELoss()
            print(f"Criterion: BCELoss")
        else:
            class_weights = torch.tensor([1.0 for _ in range(self.num_classes)])
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.DEVICE))
            print(f"Criterion: CrossEntropyLoss")

    def step(self, conf_matrix: np.ndarray, verbose: bool = False) -> None:
        """
        Update the classes weights based on the confusion matrix.
        If the number of classes is 1, the function returns without doing anything.
        Args:
            conf_matrix (np.ndarray): The confusion matrix representing the classification results.
            verbose (bool): The flag to print the updated classes weights.

        Returns:
            None
        """
        if self.num_classes == 1:
            return
        if self.step_size <= 0:
            return
        self.itr += 1
        if self.itr % self.step_size == 0:
            samples_per_class = conf_matrix.sum(axis=1)
            diag = np.diag(conf_matrix)
            diag = np.where(diag == 0, 1, diag)
            # Compute the weight adjustment based on confusion matrix
            # We use **3 to mark the difference between the classes
            weights_adjustment = (samples_per_class / diag) ** 3
            new_weights = ((weights_adjustment / weights_adjustment.sum()) + 1) * 1.5

            new_weights = new_weights.to(self.DEVICE)
            self.criterion.weight = new_weights
        if verbose:
            print("Diag: ", diag)
            print("Updated classes weights: ", new_weights)
