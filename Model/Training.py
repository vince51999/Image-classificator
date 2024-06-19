import torch
import Model.EarlyStopping as EarlyStopping
import Model.Statistics as Statistics
import Model.NNArchitecture as NNArchitecture

from Model.Optimizer import Optimizer
from Model.Criterion import Criterion
from Model.Results import Results as Res
from Model.TinyImageNetDataset import TinyImageNetDataset


def train_loop(
    dataset: TinyImageNetDataset,
    model,
    optimizer: Optimizer,
    criterion: Criterion,
    device,
    start_epoch,
    epochs,
    tolerance,
    min_delta,
    train_stats: Statistics,
    val_stats: Statistics,
    res: Res,
):
    early_stopping = EarlyStopping.EarlyStopping(tolerance, min_delta)
    best_epoch_val_loss = float("inf")
    last_epoch = start_epoch + epochs

    for epoch in range(start_epoch, last_epoch):
        res.print(f"\nEPOCH {epoch+1} of {last_epoch}\n")
        train_stats.reset()
        val_stats.reset()

        train_loss_list = __train(
            dataset.train_dataloader,
            model,
            optimizer.optimizer,
            criterion.criterion,
            device,
            train_stats,
        )
        val_loss_list = val(
            dataset.val_dataloader, model, criterion.criterion, device, val_stats
        )
        epoch_train_loss = sum(train_loss_list) / len(dataset.train_dataloader)
        epoch_val_loss = sum(val_loss_list) / len(dataset.val_dataloader)
        train_stats.step(epoch + 1, epoch_train_loss, "Train")
        val_stats.step(epoch + 1, epoch_val_loss, "Val")
        dataset.step(verbose=True)
        optimizer.step(verbose=True)
        dataset.augumentation(verbose=True)

        res.createConfusionMatrix(
            val_stats.get_confusion_matrix(),
            val_stats.get_classes(),
            "Val conf matrix",
            epoch + 1,
        )
        if epoch_val_loss < best_epoch_val_loss:
            NNArchitecture.save_checkpoint(
                res.directory,
                epoch,
                model,
                optimizer,
                dataset,
                criterion,
            )
            best_epoch_val_loss = epoch_val_loss
            res.print(
                f"Best model saved at epoch {epoch+1} with loss {best_epoch_val_loss}"
            )
        # early stopping
        early_stopping(epoch_train_loss, epoch_val_loss)
        if early_stopping.early_stop:
            res.print(f"Early stopping at epoch {epoch+1}")
            NNArchitecture.save_checkpoint(
                res.directory, epoch, model, optimizer, dataset, criterion
            )
            return
    end_epoch = last_epoch - 1
    NNArchitecture.save_checkpoint(
        res.directory, end_epoch, model, optimizer, dataset, criterion
    )


# Training function
def __train(trainset, model, optimizer, criterion, device, train_stats):
    model.train()
    train_loss_list = []

    for i, data in enumerate(trainset):
        # Forward path
        optimizer.zero_grad()
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # put data to device
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        if len(train_stats.get_classes()) == 1:
            outputs = outputs.view(-1)
            labels = torch.tensor([1.0 for l in labels]).to(device)

        loss = criterion(outputs, labels)
        loss_value = loss.item()
        train_loss_list.append(loss_value)
        train_stats.update(outputs, labels)

        loss.backward()
        optimizer.step()

    return train_loss_list


# Validation function
@torch.no_grad()
def val(valset, model, criterion, device, val_stats):
    model.eval()
    val_loss_list = []

    for i, data in enumerate(valset):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # put data to device
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)

        if len(val_stats.get_classes()) == 1:
            outputs = outputs.view(-1)
            labels = torch.tensor([1.0 for l in labels]).to(device)

        loss = criterion(outputs, labels)

        loss_value = loss.item()
        val_loss_list.append(loss_value)
        val_stats.update(outputs, labels)

    return val_loss_list
