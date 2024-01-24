import torch
import torch.nn as nn


def get_model(type="resnet50", num_classes=200, wieghts=None):
    """
    Classification architecture is like a Resnet.

    Args:
        type (str, optional): Architecture type. Defaults to "resnet50".
        num_classes (int, optional): Number of output classes. Defaults to 200.
        wieghts (_type_, optional): Set pretreined model or not. Defaults to None.

    Returns:
        The model with the specified architecture.
    """
    model = torch.hub.load("pytorch/vision:v0.10.0", type, weights=wieghts)
    # get the number of input features
    in_features = model.fc.in_features
    # define a new head for the detector with required number of classes
    model.fc = nn.Linear(in_features, num_classes)
    print(model)
    return model
