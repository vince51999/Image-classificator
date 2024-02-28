import torch
import torch.nn as nn


def get_nn_architecture(
    type="resnet50",
    num_classes=200,
    pretrained=False,
    dropout_rate_bb=0.2,
    dropout_rate_fc=0.5,
):
    """
    Classification architecture is like a Resnet.

    Args:
        type (str, optional): Architecture type. Defaults to "resnet50".
        num_classes (int, optional): Number of output classes. Defaults to 200.
        wieghts (_type_, optional): Set pretreined model or not. Defaults to None.

    Returns:
        The model with the specified architecture.
    """
    if pretrained is False:
        model = torch.hub.load("pytorch/vision:v0.10.0", type)
    else:
        model = torch.hub.load("pytorch/vision:v0.10.0", type, weights="IMAGENET1K_V1")

    # get the number of input features
    if dropout_rate_bb > 0:
        __append_dropout(model, rate=dropout_rate_bb)
    in_features = model.fc.in_features
    # define a new head for the detector with required number of classes
    fc = nn.Linear(in_features, num_classes)
    if num_classes == 1:
        fc = nn.Sequential(fc, nn.Sigmoid())
    if dropout_rate_fc > 0:
        model.fc = nn.Sequential(nn.Dropout(dropout_rate_fc), fc)
    else:
        model.fc = fc
    return model


def __append_dropout(model, rate=0.2):
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            __append_dropout(module)
        if isinstance(module, nn.ReLU):
            # inplace=false to avoid the error: one of the variables needed for gradient computation has been modified by an inplace operation
            # When we set implace=true we overwrite input tensor (can give error when we use this tensor but use less memory)
            # When we set implace=false we work on a copy of tensor (not give error but
            # Dropout2d before relu: This order encourages the network to learn robust features while maintaining non-linearity.
            # Relu before Dropout2d: The idea is to apply dropout after the non-linearity to prevent overfitting on specific features.
            new = nn.Sequential(module, nn.Dropout2d(p=rate, inplace=False))
            setattr(model, name, new)
