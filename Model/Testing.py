import torch
import torch.nn as nn

from Model.NNArchitecture import (
    append_dropout,
    append_dropout_after_each_relu,
    append_dropout_before_each_relu,
    init_weights,
)


def get_nn_architecture(
    type="resnet50",
    num_classes=200,
    fine_tune=False,
    transfer_learning=False,
    dropout_rate_rb=0.2,
    dropout_rate_fc=0.5,
    dropout_pos_rb=0,
    dropout_pos_fc=0,
):
    """
    Classification architecture is like a Resnet.

    Args:
        type (str, optional): Architecture type. Defaults to "resnet50".
        num_classes (int, optional): Number of output classes. Defaults to 200.
        fine_tune (bool, optional): Fine-tune the model. Defaults to False.
        transfer_learning (bool, optional): Transfer learning. Defaults to False.
        dropout_rate_rb (float, optional): Dropout rate in the residual block. Defaults to 0.2.
        dropout_rate_fc (float, optional): Dropout rate in the fully connected layer. Defaults to 0.5.
        dropout_pos_rb (int, optional): Dropout position in the residual block. Defaults to 0.
        dropout_pos_fc (int, optional): Dropout position in the fully connected layer. Defaults to 0.
    Returns:
        The model with the specified architecture.
    """
    if type != "resnet18" and type != "resnet50" and type != "resnet101":
        type = "resnet50"
    if fine_tune is False and transfer_learning is False:
        print(f"Architecture: not-pretrained {type}")
        model = torch.hub.load("pytorch/vision:v0.17.0", type)
        model.apply(init_weights)
    elif fine_tune is True or transfer_learning is True:
        print(f"Architecture: pretrained {type}")
        model = torch.hub.load("pytorch/vision:v0.17.0", type, weights="IMAGENET1K_V1")

    if fine_tune is True:
        print("Fine-tune")
    elif transfer_learning is True and fine_tune is False:
        print("Transfer learning")
        for param in model.parameters():
            param.requires_grad = False

    print(
        f"Dropout rate basicBlock: {dropout_rate_rb}, Dropout rate final layer: {dropout_rate_fc}"
    )
    if dropout_rate_rb > 0:
        if dropout_pos_rb == 1:
            append_dropout_before_each_relu(model, rate=dropout_rate_rb)
            print("Dropout before each ReLU")
        elif dropout_pos_rb == 2:
            append_dropout_after_each_relu(model, rate=dropout_rate_rb)
            print("Dropout after each ReLU")
        else:
            append_dropout(model, type, rate=dropout_rate_rb)
            print("Dropout before the last convolutional layer")
    # define a new head for the detector with required number of classes
    if dropout_rate_fc > 0:
        avgpool = model.avgpool
        if dropout_pos_fc == 1:
            model.avgpool = nn.Sequential(avgpool, nn.Dropout(dropout_rate_fc))
            print("Dropout after the average pooling layer")
        else:
            model.avgpool = nn.Sequential(nn.Dropout(dropout_rate_fc), avgpool)
            print("Dropout before the average pooling layer")

    in_features = model.fc.in_features
    fc = nn.Linear(in_features, num_classes)
    if num_classes == 1:
        fc = nn.Sequential(fc, nn.Sigmoid())
    model.fc = fc
    return model


def load_model(
    path: str,
    model: torch.nn.Module,
):
    """
    Function to load model from the specified path.
    The model is loaded and set to evaluation mode.

    Args:
        path (str): The path to load the model.
        model (torch.nn.Module): The model to load.

    Returns:
        torch.nn.Module: The model.
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    return model


def test_model(model, input, device, oneClass = False):
    """
    Function to test the model with the specified image.

    Args:
        model (torch.nn.Module): The model to test.
        image (torch.Tensor): The image to test.
        device (torch.device): The device to put the image.

    Returns:
        torch.Tensor: The output of the model.
    """
    model.eval()
    with torch.no_grad():
        input = input.unsqueeze(0).to(device)
        output = model(input)
        if oneClass:
            output = output.view(-1)
            predicted = (output > 0.5).float()
            print(output)
        else:
            _, predicted = torch.max(output, 1)
    return predicted
