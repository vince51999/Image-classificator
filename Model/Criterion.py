import torch.nn as nn

from Model.Results import Results as Res


class Criterion:
    """
    The criterion class that is used to calculate the loss based on the number of classes in the dataset.

    Attributes:
        criterion (nn.Module): The criterion to calculate the loss.
        num_classes (int): The number of classes in the dataset.

    Methods:
        state_dict(): Returns the state dictionary of the criterion.
        load_state_dict(state_dict): Load the state dictionary of the criterion.
    """

    def __init__(self, num_classes: int, res: Res):
        """
        Initialize the criterion based on the number of classes.

        Args:
            num_classes (int): The number of classes in the dataset.
            res (Res): The results class to print the criterion details.
        """
        res = res
        self.criterion = None
        self.num_classes = num_classes
        if self.num_classes == 1:
            self.criterion = nn.BCELoss()
            res.print(f"Criterion: BCELoss")
        else:
            self.criterion = nn.CrossEntropyLoss()
            res.print(f"Criterion: CrossEntropyLoss")

    def state_dict(self):
        """
        Returns the state dictionary of the criterion.
        """
        return self.criterion.state_dict()

    def load_state_dict(self, state_dict):
        """
        Load the state dictionary of the criterion.
        """
        self.criterion.load_state_dict(state_dict)
