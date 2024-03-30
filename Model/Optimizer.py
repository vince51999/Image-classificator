import torch
import torch.optim as optim


class Optimizer:
    """
    Optimizer class to initialize the optimizer and the learning rate scheduler.
    """

    def __init__(
        self,
        momentum: float,
        lr: float,
        step: int,
        gamma_lr: float,
        weight_decay: float,
        model: torch.nn.Module,
    ):
        """
        Initialize the optimizer and the learning rate scheduler.
        Args:
            momentum (float): Value of the momentum for the optimizer. If 0, Adam optimizer is used.
            lr (float): Learning rate for the optimizer.
            step (int): Step size for the learning rate scheduler.
            gamma_lr (float): Multiplicative factor of learning rate decay.
            weight_decay (float): Weight decay for the optimizer.
            model (torch.nn.Module): The model to optimize.
        """
        self.optimizer = None
        if momentum == 0.0:
            self.optimizer = optim.Adam(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
            print(f"Optimizer: Adam, lr: {lr}, weight_decay: {weight_decay}")
        else:
            self.optimizer = optim.SGD(
                model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
            )
            print(
                f"Optimizer: SGD, lr: {lr}, momentum: {momentum}, weight_decay: {weight_decay}"
            )
        print(f"LR scheduler: StepLR, step size: {step}, gamma: {gamma_lr}")

        self.scheduler1 = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=step, gamma=gamma_lr
        )

    def step(self, verbose: bool = False) -> None:
        """
        Update the learning rate of the optimizer.

        Args:
            verbose (bool, optional): Print the learning rate. Defaults to False.

        Returns:
            None
        """
        if self.optimizer.param_groups[0]["lr"] > 0.000001:
            self.scheduler1.step()
        if self.optimizer.param_groups[0]["lr"] < 0.000001:
            self.optimizer.param_groups[0]["lr"] = 0.000001
        if verbose:
            print("Learning rate:", self.optimizer.param_groups[0]["lr"])
