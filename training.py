def __train(trainset, model, optimizer, criterion, device):
    model.train()

        loss_value = loss.item()
        train_loss_list.append(loss_value)
