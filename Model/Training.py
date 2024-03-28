import torch
import Model.EarlyStopping as EarlyStopping
from Model.Optimizer import Optimizer
from Model.Criterion import Criterion
from Model.TinyImageNetDataset import TinyImageNetDataset


def train_loop(
    dataset: TinyImageNetDataset,
    model,
    optimizer: Optimizer,
    criterion: Criterion,
    device,
    epochs,
    tolerance,
    min_delta,
    train_stats,
    val_stats,
):
    early_stopping = EarlyStopping.EarlyStopping(tolerance, min_delta)

    for epoch in range(epochs):
        print(f"\nEPOCH {epoch+1} of {epochs}")
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
        val_loss_list = __val(
            dataset.val_dataloader, model, criterion.criterion, device, val_stats
        )
        epoch_train_loss = sum(train_loss_list) / len(dataset.train_dataloader)
        epoch_val_loss = sum(val_loss_list) / len(dataset.val_dataloader)
        print(f"\nEpoch #{epoch+1}")
        print(f"Train loss: {epoch_train_loss:.3f}")
        train_stats.print("Training")
        print(f"Val loss: {epoch_val_loss:.3f}")
        val_stats.print("Validation")
        train_stats.save_epoch(epoch + 1, epoch_train_loss)
        val_stats.save_epoch(epoch + 1, epoch_val_loss)
        dataset.step(verbose=True)
        optimizer.step(verbose=True)
        criterion.step(val_stats.get_confusion_matrix())
        # early stopping
        early_stopping(epoch_train_loss, epoch_val_loss)
        if early_stopping.early_stop:
            print("Early stop at epoch:", epoch + 1)
            return


# Training function
def __train(trainset, model, optimizer, criterion, device, train_stats):
    model.train()
    print("Training")
    train_loss_list = []

    for i, data in enumerate(trainset):
        # Forward path
        optimizer.zero_grad()
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # put data to device
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss_value = loss.item()
        train_loss_list.append(loss_value)
        train_stats.update(outputs, labels)

        loss.backward()
        optimizer.step()

    print("Finished training")
    return train_loss_list


# Validation function
@torch.no_grad()
def __val(valset, model, criterion, device, val_stats):
    model.eval()
    print("Validating")
    val_loss_list = []

    for i, data in enumerate(valset):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # put data to device
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss_value = loss.item()
        val_loss_list.append(loss_value)
        val_stats.update(outputs, labels)

    print("Finished validating")
    return val_loss_list
