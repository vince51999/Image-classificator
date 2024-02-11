import torch


def save_model(epoch, model, optimizer):
    """
    Function to save the trained model till current epoch, or whenver called
    """
    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        "res/faster_rcnn.pth",
    )


# Test function
@torch.no_grad()
def test(testset, model, criterion, device, test_stats):
    model.eval()
    print("Testing")
    test_loss_list = []

    for i, data in enumerate(testset):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # put data to device
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss_value = loss.item()
        test_loss_list.append(loss_value)
        test_stats.update(outputs, labels)

    print("Finished testing")
    return test_loss_list
