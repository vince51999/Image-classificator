import torch
import torch.optim as optim
from Model.Results import Results as Res


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
        res: Res,
        scheduler: str = "StepLR",
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
        self.res = res
        self.optimizer = None
        if momentum == 0.0:
            self.optimizer = optim.Adam(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
            self.res.print(f"Optimizer: Adam, lr: {lr}, weight_decay: {weight_decay}")
        else:
            self.optimizer = optim.SGD(
                model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
            )
            self.res.print(
                f"Optimizer: SGD, lr: {lr}, momentum: {momentum}, weight_decay: {weight_decay}"
            )
        self.res.print(f"LR scheduler: StepLR, step size: {step}, gamma: {gamma_lr}")

        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=step, gamma=gamma_lr
        )

        if scheduler == "CosineAnnealingWarmRestarts":
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=step, T_mult=1, eta_min=0.00001
            )
            self.res.print(
                f"LR scheduler: CosineAnnealingWarmRestarts, T_max: {step}, eta_min: 0.00001"
            )

    def state_dict(self):
        return self.optimizer.state_dict(), self.scheduler.state_dict()

    def load_state_dict(self, optimizer_state_dict, scheduler_state_dict):
        self.optimizer.load_state_dict(optimizer_state_dict)
        self.scheduler.load_state_dict(scheduler_state_dict)

    def step(self, verbose: bool = False) -> None:
        """
        Update the learning rate of the optimizer.

        Args:
            verbose (bool, optional): Print the learning rate. Defaults to False.

        Returns:
            None
        """
        if self.optimizer.param_groups[0]["lr"] > 0.00001:
            self.scheduler.step()
        if self.optimizer.param_groups[0]["lr"] < 0.00001:
            self.optimizer.param_groups[0]["lr"] = 0.00001
        if verbose:
            lr = self.optimizer.param_groups[0]["lr"]
            self.res.print(f"Learning rate: {lr}")
