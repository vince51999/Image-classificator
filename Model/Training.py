import torch
import matplotlib.pyplot as plt
import Model.EarlyStopping as EarlyStopping

from tqdm import tqdm


def createChart(xlabel, ylabel, xdata, ydata, path, dataNames=["chart"]):
    plt.xlabel(xlabel, fontsize=10)
    plt.ylabel(ylabel, fontsize=10)

    for index, name in enumerate(dataNames):
        plt.plot(
            xdata,
            ydata[index],
            marker="o",
            linestyle="dashed",
            linewidth=2,
            markersize=6,
            label=name,
        )
    plt.title(f"Training results", fontsize=12)
    if dataNames[0] != "chart":
        plt.legend()
    plt.savefig(path, format="pdf", bbox_inches="tight")
    plt.clf()


def train_loop(trainset, valset, model, optimizer, criterion, device, epochs):
    train_losses = []
    valid_losses = []
    early_stopping = EarlyStopping.EarlyStopping(tolerance=5, min_delta=0.5)

    for epoch in range(epochs):
        print(f"\nEPOCH {epoch+1} of {epochs}")

        train_loss_list = __train(trainset, model, optimizer, criterion, device)
        val_loss_list = __val(valset, model, criterion, device)
        epoch_train_loss = sum(train_loss_list) / len(trainset)
        epoch_val_loss = sum(val_loss_list) / len(valset)
        train_losses.append(epoch_train_loss)
        valid_losses.append(epoch_val_loss)

        print(f"Epoch #{epoch+1} train loss: {epoch_train_loss:.3f}")
        print(f"Epoch #{epoch+1} validation loss: {epoch_val_loss:.3f}")
        # early stopping
        early_stopping(epoch_train_loss, epoch_val_loss)
        if early_stopping.early_stop:
            print("We are at epoch:", epoch + 1)
            break
    return train_losses, valid_losses



# Training function
def __train(trainset, model, optimizer, criterion, device):
    model.train()
    print("Training")
    train_loss_list = []

    # initialize tqdm progress bar
    prog_bar = tqdm(trainset, total=len(trainset))

    for i, data in enumerate(prog_bar):
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

        loss.backward()
        optimizer.step()

        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")

    print("Finished training")
    return train_loss_list


# Validation function
@torch.no_grad()
def __val(valset, model, criterion, device):
    model.eval()
    print("Validating")
    val_loss_list = []

    # initialize tqdm progress bar
    prog_bar = tqdm(valset, total=len(valset))

    for i, data in enumerate(prog_bar):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # put data to device
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss_value = loss.item()
        val_loss_list.append(loss_value)

        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")

    print("Finished validating")
    return val_loss_list
