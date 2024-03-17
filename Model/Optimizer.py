import torch.optim as optim


class Optimizer:
    def __init__(
        self,
        momentum: float,
        lr: float,
        step: int,
        gamma_lr: float,
        weight_decay,
        model,
    ):
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

    def step(self, verbose=False):
        if self.optimizer.param_groups[0]["lr"] > 0.000001:
            self.scheduler1.step()
        if self.optimizer.param_groups[0]["lr"] < 0.000001:
            self.optimizer.param_groups[0]["lr"] = 0.000001
        if verbose:
            print("Learning rate:", self.optimizer.param_groups[0]["lr"])
