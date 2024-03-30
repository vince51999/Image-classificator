class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after a certain number of epochs.

    Attributes:
        apply (bool): A flag to apply early stopping.
        tolerance (int): The number of epochs to wait for improvement before stopping.
        min_delta (float): The minimum change in the monitored quantity to qualify as an improvement.
        counter (int): The counter to check the number of epochs without improvement.
        early_stop (bool): A flag to stop the training.

    Methods:
        __init__(tolerance, min_delta): Initialize the EarlyStopping class.
        __call__(train_loss, validation_loss): Check if the training should stop.
    """

    def __init__(self, tolerance: int = 5, min_delta: float = 0) -> None:
        """
        Initialize the EarlyStopping class.

        Args:
            tolerance (int, optional): The number of epochs to wait for improvement before stopping. Defaults to 5.
            min_delta (float, optional): The minimum change in the monitored quantity to qualify as an improvement. Defaults to 0.

        Returns:
            None
        """
        self.apply = True
        if tolerance < 1 or min_delta <= 0:
            self.apply = False
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss: float, validation_loss: float) -> None:
        """
        Check if the training should stop.

        Args:
            train_loss (float): Loss on the training set
            validation_loss (float): Loss on the validation set

        Returns:
            None
        """
        if not self.apply:
            return
        if (validation_loss - train_loss) > self.min_delta:
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True
