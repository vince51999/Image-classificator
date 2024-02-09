import argparse
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import random

import Model.TinyImageNetDataset as TinyImageNetDataset
import Model.NNArchitecture as NNArchitecture
import Model.Testing as Testing
import Model.Training as Training

TRAIN_BATCH_SIZE = 100  # increase / decrease according to GPU memeory
EVAL_BATCH_SIZE = 10  # increase / decrease according to GPU memeory
NUM_EPOCHS = 30  # number of epochs to train for
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def random_classes(given_class, num_classes=200):
    """
    Randomly select classes from the dataset.
    Args:
        given_class (int): Specific class that we want in the set of classes. The rest of the classes will be random.
        num_classes (int, optional): Number of classes to return. Defaults to 200.
    """
    classes = [i for i in range(num_classes)]
    if num_classes == 200:
        return classes
    classes.remove(given_class)
    size = num_classes - 1
    classes = random.sample(classes, size)
    classes.append(given_class)
    return classes


def main(c=0, num_classes=200):
    classes = random_classes(c, num_classes)
    dataset = TinyImageNetDataset.TinyImageNetDataset(
        TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE, classes=classes
    )
    print(f"Training on {num_classes} classes")
    print(f"Classes: {classes}")
    model = NNArchitecture.get_nn_architecture(
        type="resnet50", num_classes=num_classes, wieghts=None
    )
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_losses, valid_losses = Training.train_loop(
        dataset.train_dataloader,
        dataset.val_dataloader,
        model,
        optimizer,
        criterion,
        DEVICE,
        NUM_EPOCHS,
    )

    xdata = [t for t in range(NUM_EPOCHS)]

    Training.createChart(
        "Epochs",
        "Losses",
        xdata,
        [train_losses, valid_losses],
        "./results/model_loss.pdf",
        ["train_losses", "valid_losses"],
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="ImageNet Training",
        description="Train and test an image classification model based on ResNet and Tiny ImageNet dataset",
    )
    parser.add_argument(
        "--cls",
        help="Specific class that we want in the set of classes. The rest of the classes will be random.",
        required=True,
        default="",
        type=int,
    )
    parser.add_argument(
        "--num_classes",
        help="Total number of classes to train on. Default is 200. If you want to train on a subset of classes, specify the number of classes here.",
        required=True,
        default="",
        type=int,
    )
    args = parser.parse_args()
    c = args.cls
    num_classes = args.num_classes
    if c < 0 or c > 199:
        print("Class should be between 0 and 199")
        exit(1)
    if num_classes < 1 or num_classes > 200:
        print("Number of classes should be between 1 and 200")
        exit(1)
    main(c, num_classes)
