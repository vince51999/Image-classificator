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
import Model.Statistics as Statistics
import Model.CreateChart as CreateChart

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def random_classes(given_class, num_classes=200):
    """
    Randomly select classes from the dataset.
    Args:
        given_class (int): Specific class that we want in the set of classes. The rest of the classes will be random.
        num_classes (int, optional): Number of classes to return. Defaults to 200.
    """
    classes = [i for i in range(200)]
    if num_classes == 200:
        return classes
    classes.remove(given_class)
    size = num_classes - 1
    classes = random.sample(classes, size)
    classes.append(given_class)
    return classes

def createCharts(train_stats: Statistics, val_stats: Statistics):
    epochs = train_stats.epochs
    CreateChart.createChart(
        "Epochs",
        "Losses",
        epochs,
        [train_stats.losses, val_stats.losses],
        "./results/loss.pdf",
        ["train_losses", "val_losses"],
    )
    CreateChart.createChart(
        "Epochs",
        "Accuracy",
        epochs,
        [train_stats.accuracy, val_stats.accuracy],
        "./results/accuracy.pdf",
        ["train_accuracy", "val_accuracy"],
    )
    CreateChart.createChart(
        "Epochs",
        "F-Measure",
        epochs,
        [train_stats.f_measure, val_stats.f_measure],
        "./results/f_measure.pdf",
        ["train_f_measure", "val_f_measure"],
    )
    CreateChart.createChart(
        "Epochs",
        "Recall",
        epochs,
        [train_stats.recall, val_stats.recall],
        "./results/recall.pdf",
        ["train_recall", "val_recall"],
    )
    CreateChart.createChart(
        "Epochs",
        "Precision",
        epochs,
        [train_stats.precision, val_stats.precision],
        "./results/precision.pdf",
        ["train_precision", "val_precision"],
    )


def main(
    c=0,
    num_classes=200,
    num_epochs=10,
    eval_batch_size=100,
    train_batch_size=100,
    tolerance=5,
    min_delta=0.5,
):
    classes = random_classes(c, num_classes)
    dataset = TinyImageNetDataset.TinyImageNetDataset(
        train_batch_size, eval_batch_size, classes=classes
    )
    print(f"Training on {num_classes} classes: {classes}")
    print(
        f"Num pochs: {num_epochs}, Train batch size: {train_batch_size}, Eval batch size: {eval_batch_size}"
    )
    print(f"EarlyStopping tolerance:{tolerance} min delta:{min_delta}")

    model = NNArchitecture.get_nn_architecture(
        type="resnet50", num_classes=num_classes, wieghts=None
    )
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_stats = Statistics.Statistics(
        classes,
        (len(dataset.train_dataloader) * train_batch_size) / num_classes,
    )
    val_stats = Statistics.Statistics(
        classes,
        (len(dataset.val_dataloader) * eval_batch_size) / num_classes,
    )

    Training.train_loop(
        dataset.train_dataloader,
        dataset.val_dataloader,
        model,
        optimizer,
        criterion,
        DEVICE,
        num_epochs,
        tolerance,
        min_delta,
        train_stats,
        val_stats,
    )
    
    createCharts(train_stats, val_stats)

    test_stats = Statistics.Statistics(
        classes,
        (len(dataset.test_dataloader) * eval_batch_size) / num_classes,
    )
    test_losses = Testing.test(
        dataset.test_dataloader, model, criterion, DEVICE, test_stats
    )

    print(f"Test loss: {sum(test_losses) / len(dataset.test_dataloader):.3f}")
    test_stats.print("Test")
    test_stats.reset()


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
    parser.add_argument(
        "--num_epochs",
        help="Number of epochs to train for.",
        required=True,
        default="",
        type=int,
    )
    parser.add_argument(
        "--eval_batch_size",
        help="Batch size for evaluation. Increase / decrease according to GPU memeory.",
        required=True,
        default="",
        type=int,
    )
    parser.add_argument(
        "--train_batch_size",
        help="Batch size for training. Increase / decrease according to GPU memeory.",
        required=True,
        default="",
        type=int,
    )
    parser.add_argument(
        "--tolerance",
        help="Early stopping tolerance. If the validation loss does not improve for this many epochs, the training will stop.",
        required=True,
        default="",
        type=int,
    )
    parser.add_argument(
        "--min_delta",
        help="Minimum change in validation loss to be considered as an improvement.",
        required=True,
        default="",
        type=float,
    )
    args = parser.parse_args()

    c = args.cls
    num_classes = args.num_classes
    num_epochs = args.num_epochs
    eval_batch_size = args.eval_batch_size
    train_batch_size = args.train_batch_size
    tolerance = args.tolerance
    min_delta = args.min_delta

    if c < 0 or c > 199:
        print("Class should be between 0 and 199")
        exit(1)
    if num_classes < 1 or num_classes > 200:
        print("Number of classes should be between 1 and 200")
        exit(1)

    main(
        c,
        num_classes,
        num_epochs,
        eval_batch_size,
        train_batch_size,
        tolerance,
        min_delta,
    )
