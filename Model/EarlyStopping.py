class EarlyStopping:
    def __init__(self, tolerance: int = 5, min_delta: float = 0):
        self.apply = True
        if tolerance < 1 or min_delta <= 0:
            self.apply = False
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss: float, validation_loss: float):
        if not self.apply:
            return
        if (validation_loss - train_loss) > self.min_delta:
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True
