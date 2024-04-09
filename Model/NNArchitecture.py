import torch
import torch.nn as nn

from Model.Results import Results as Res


def get_nn_architecture(
    res: Res,
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
        dropout_rate_bb (float, optional): Dropout rate before the fully connected layer. Defaults to 0.2.
        dropout_rate_fc (float, optional): Dropout rate before the fully connected layer. Defaults to 0.5.
    Returns:
        The model with the specified architecture.
    """
    res = res
    if type != "resnet18" and type != "resnet50" and type != "resnet101":
        type = "resnet50"
    if fine_tune is False and transfer_learning is False:
        res.print(f"Architecture: not-pretrained {type}")
        model = torch.hub.load("pytorch/vision:v0.17.0", type)
        model.apply(__init_weights)
    elif fine_tune is True or transfer_learning is True:
        res.print(f"Architecture: pretrained {type}")
        model = torch.hub.load("pytorch/vision:v0.17.0", type, weights="IMAGENET1K_V1")

    if transfer_learning is True:
        res.print("Transfer learning")
        for param in model.parameters():
            param.requires_grad = False
    else:
        if dropout_rate_rb > 0:
            if dropout_pos_rb == 1:
                __append_dropout_before_each_relu(model, rate=dropout_rate_rb)
                res.print("Dropout before each ReLU")
            elif dropout_pos_rb == 2:
                __append_dropout_after_each_relu(model, rate=dropout_rate_rb)
                res.print("Dropout after each ReLU")
            else:
                __append_dropout(model, type, rate=dropout_rate_rb)
                res.print("Dropout before the last convolutional layer")
        # define a new head for the detector with required number of classes
        if dropout_rate_fc > 0:
            avgpool = model.avgpool
            if dropout_pos_fc == 1:
                model.avgpool = nn.Sequential(avgpool, nn.Dropout(dropout_rate_fc))
                res.print("Dropout after the average pooling layer")
            else:
                model.avgpool = nn.Sequential(nn.Dropout(dropout_rate_fc), avgpool)
                res.print("Dropout before the average pooling layer")

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


def __append_dropout_before_each_relu(model, rate=0.2):
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            __append_dropout_before_each_relu(module, rate=rate)
        if isinstance(module, nn.ReLU):
            new = nn.Sequential(nn.Dropout2d(p=rate, inplace=False), module)
            setattr(model, name, new)


def __append_dropout_after_each_relu(model, rate=0.2):
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            __append_dropout_after_each_relu(module, rate=rate)
        if isinstance(module, nn.ReLU):
            new = nn.Sequential(module, nn.Dropout2d(p=rate, inplace=False))
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
