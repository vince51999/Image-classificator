import torch
import torch.nn as nn


def get_nn_architecture(
    type="resnet50",
    num_classes=200,
    wieghts=None,
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
    model = torch.hub.load("pytorch/vision:v0.10.0", type, weights=wieghts)
    # get the number of input features
    if dropout_rate_bb > 0:
        __append_dropout(model, rate=dropout_rate_bb)
    in_features = model.fc.in_features
    in_features_d2 = round(in_features / 2)
    # # define a new head for the detector with required number of classes
    fc = nn.Sequential(
        nn.Dropout(dropout_rate_fc), nn.Linear(in_features, in_features), nn.ReLU()
    )
    fc1 = nn.Sequential(
        nn.Dropout(dropout_rate_fc),
        nn.Linear(in_features, in_features_d2),
        nn.ReLU(),
    )
    fc2 = nn.Sequential(nn.Linear(in_features_d2, num_classes))
    model.fc = nn.Sequential(fc, fc1, fc2)
    return model


def __append_dropout(model, rate=0.2):
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            __append_dropout(module)
        if isinstance(module, nn.ReLU):
            # inplace=false to avoid the error: one of the variables needed for gradient computation has been modified by an inplace operation
            # When we set implace=true we overwrite input tensor (can give error when we use this tensor but use less memory)
            # When we set implace=false we work on a copy of tensor (not give error but
            new = nn.Sequential(module, nn.Dropout2d(p=rate, inplace=False))
            setattr(model, name, new)
