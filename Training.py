from tqdm import tqdm
import torch


# Training function
def train(trainset, model, optimizer, criterion, device):
    print("Training")
    model.train()
    train_itr = 0
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

        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        train_loss_list.append(loss_value)

        train_itr += 1

        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")

    print("Finished Training")
    return train_loss_list
