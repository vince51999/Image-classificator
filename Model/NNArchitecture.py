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
    if type != "resnet18" and type != "resnet50":
        type = "resnet50"
    if pretrained is False:
        print(f"Architecture: not-pretrained {type}")
        model = torch.hub.load("pytorch/vision:v0.10.0", type)
        model.apply(__init_weights)
    else:
        print(f"Architecture: pretrained {type}")
        model = torch.hub.load("pytorch/vision:v0.10.0", type, weights="IMAGENET1K_V1")

    # get the number of input features
    if dropout_rate_bb > 0:
        __append_dropout(model, type, rate=dropout_rate_bb)
    # define a new head for the detector with required number of classes
    if dropout_rate_fc > 0:
        avgpool = model.avgpool
        model.avgpool = nn.Sequential(nn.Dropout(dropout_rate_fc), avgpool)
    in_features = model.fc.in_features
    fc = nn.Linear(in_features, num_classes)
    if num_classes == 1:
        fc = nn.Sequential(fc, nn.Sigmoid())
    model.fc = fc
    return model


def __append_dropout(model, type, rate=0.2):
    dropout_before = ""
    if type == "resnet50":
        dropout_before = "conv3"
        ## Where insert the dropout layer in the bottleneck block
        # identity = x
        # out = self.conv1(x)
        # out = self.bn1(out)
        # out = self.relu(out)
        # out = self.conv2(out)
        # out = self.bn2(out)
        # out = self.relu(out)
        ### Dropout2d
        # out = self.conv3(out)
        # out = self.bn3(out)
        # if self.downsample is not None:
        #     identity = self.downsample(x)
        # out += identity
        # out = self.relu(out)
    if type == "resnet18":
        dropout_before = "conv2"
        ## Where insert the dropout layer in the basic block
        # identity = x
        # out = self.conv1(x)
        # out = self.bn1(out)
        # out = self.relu(out)
        ### Dropout2d
        # out = self.conv2(out)
        # out = self.bn2(out)
        # if self.downsample is not None:
        #     identity = self.downsample(x)
        # out += identity
        # out = self.relu(out)
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            __append_dropout(module, type, rate=rate)
        if isinstance(module, nn.Conv2d):
            # inplace=false to avoid the error: one of the variables needed for gradient computation has been modified by an inplace operation
            # When we set implace=true we overwrite input tensor (can give error when we use this tensor but use less memory)
            # When we set implace=false we work on a copy of tensor (not give error but use more memory)
            # Dropout2d before relu: This order encourages the network to learn robust features while maintaining non-linearity.
            # Relu before Dropout2d: The idea is to apply dropout after the non-linearity to prevent overfitting on specific features.
            # Another research paper suggests that dropout before or after the ReLU layer does not make a significant difference and proof that.
            if dropout_before == name:
                new = nn.Sequential(nn.Dropout2d(p=rate, inplace=False), module)
                setattr(model, name, new)


def __init_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            # Specifically designed for deep neural network with the ReLU activation
            # that helps to reduce the vanishing gradient problem,
            # allows the network to learn deeper representations.
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm2d):
            # Batch norm layers normalize the distribution of layer outputs during training,
            # which reduces the dependency of the deep network on weight initialization strategies.
            # The weights and biases of all batch normalization layers are usually initialized as 1s and 0s, respectively.
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Linear):
            # Specifically designed for deep neural network with the ReLU activation
            # that helps to reduce the vanishing gradient problem,
            # allows the network to learn deeper representations.
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
