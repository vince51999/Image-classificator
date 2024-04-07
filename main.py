import argparse
import datetime
import torch
import torchvision

import Model.NNArchitecture as NNArchitecture
import Model.Training as Training
import Model.Statistics as Statistics

from Model.TinyImageNetDataset import TinyImageNetDataset as Tind
from Model.Optimizer import Optimizer as Op
from Model.Criterion import Criterion as Crit
from Model.Results import Results as Res

parser = argparse.ArgumentParser(
    prog="ImageNet Training",
    description="Train and test an image classification model based on ResNet and Tiny ImageNet dataset",
)
parser.add_argument(
    "--architecture",
    help="Type of resNet architecture to use.",
    required=True,
    default="",
    type=str,
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
    "--gamma_train_batch_size",
    help="Increase the train batch size by this factor. This is used to increase the number of training samples.",
    required=True,
    default="",
    type=float,
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
parser.add_argument(
    "--lr",
    help="Learning rate for the optimizer. Step size for the optimizer.",
    required=True,
    default="",
    type=float,
)
parser.add_argument(
    "--gamma_lr",
    help="Gamma for the learning rate scheduler. Learning rate is multiplied by this factor after every step.",
    required=True,
    default="",
    type=float,
)
parser.add_argument(
    "--momentum",
    help="Momentum for the optimizer. Helps in faster convergence. If 0, optimizer is Adam. If non-zero, optimizer is SGD.",
    required=True,
    default="",
    type=float,
)
parser.add_argument(
    "--weight_decay",
    help="Weight decay for the optimizer. Regularization parameter.",
    required=True,
    default="",
    type=float,
)
parser.add_argument(
    "--dropout_rate_bb",
    help="Dropout rate for basicBlock of resNet model. If 0, no dropout. If non-zero, dropout is added after every ReLU layer.",
    required=True,
    default="",
    type=float,
)
parser.add_argument(
    "--dropout_rate_fc",
    help="Dropout rate for final layer of resNet model. If 0, no dropout. If non-zero, dropout is added after every ReLU layer.",
    required=True,
    default="",
    type=float,
)
parser.add_argument(
    "--fine_tune",
    help="Fine tune the model or not. If 1, the model is fine-tuned. If False, the model is trained from scratch.",
    required=True,
    default="",
    type=int,
)

parser.add_argument(
    "--transfer_learning",
    help="Use transfer learning or not. If 1, the model is pre-trained on ImageNet. If False, the model is trained from scratch.",
    required=True,
    default="",
    type=int,
)
parser.add_argument(
    "--test",
    help="Test mode. If 1, the model is tested on a fixed set of classes.",
    required=True,
    default="",
    type=int,
)
parser.add_argument(
    "--increases_trainset",
    help="Increment the trainset by this factor. Default is 2. This is used to increase the number of training samples.",
    required=True,
    default="",
    type=int,
)
parser.add_argument(
    "--image_size",
    help="Image size to use for training. Default is 64.",
    required=True,
    default="",
    type=int,
)
parser.add_argument(
    "--step",
    help="Step size for the learning rate and batch size scheduler. Learning rate is multiplied by this factor after every step.",
    required=True,
    default="",
    type=int,
)
args = parser.parse_args()

architecture = args.architecture
c = args.cls
num_classes = args.num_classes
num_epochs = args.num_epochs
eval_batch_size = args.eval_batch_size
train_batch_size = args.train_batch_size
gamma_train_batch_size = args.gamma_train_batch_size
if gamma_train_batch_size <= 0:
    gamma_train_batch_size = 1
tolerance = args.tolerance
min_delta = args.min_delta
lr = args.lr
gamma_lr = args.gamma_lr
if gamma_lr <= 0:
    gamma_lr = 1
momentum = args.momentum
weight_decay = args.weight_decay
dropout_rate_bb = args.dropout_rate_bb
dropout_rate_fc = args.dropout_rate_fc
fine_tune = False
if args.fine_tune == 1:
    fine_tune = True
transfer_learning = False
if args.transfer_learning == 1 and fine_tune == False:
    transfer_learning = True
test = False
if args.test == 1:
    test = True
    num_classes = 10
increases_trainset = args.increases_trainset
image_size = 64
if args.image_size > 64:
    image_size = args.image_size
step = args.step
if step < 1:
    step = 1

if c < 0 or c > 199:
    c = 0
if num_classes < 1 or num_classes > 200:
    num_classes = 200

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def classes_list(given_class, num_classes=200, test=False):
    """
    Randomly select classes from the dataset.
    Args:
        given_class (int): Specific class that we want in the set of classes. The rest of the classes will be random.
        num_classes (int, optional): Number of classes to return. Defaults to 200.
    """
    if test:
        return [183, 93, 137, 124, 69, 134, 23, 165, 39, 0]
    classes = [i for i in range(200)]
    if num_classes == 200:
        return classes
    classes.remove(given_class)
    size = num_classes - 1
    classes = [x for i, x in enumerate(classes) if i < size]
    classes.append(given_class)
    return classes


def main(
    architecture: str = "resnet50",
    c: int = 0,
    num_classes: int = 200,
    num_epochs: int = 10,
    eval_batch_size: int = 100,
    train_batch_size: int = 100,
    gamma_train_batch_size: int = 2,
    tolerance: int = 5,
    min_delta: float = 0.5,
    lr: float = 0.001,
    gamma_lr: float = 0.3,
    momentum: float = 0.0,
    weight_decay: float = 0,
    dropout_rate_bb: float = 0.2,
    dropout_rate_fc: float = 0.5,
    fine_tune: bool = False,
    transfer_learning: bool = False,
    test: bool = False,
    increases_trainset: int = 2,
    image_size: int = 64,
    step: int = 1,
):
    """
    Train the model on the dataset ot search best parameters for the model.

    Args:
        architecture (str, optional): Type of resnet architecture. Defaults to "resnet50".
        c (int, optional): Class that you want include in the classes list to classify. Defaults to 0.
        num_classes (int, optional): Number of class to classify. Defaults to 200.
        num_epochs (int, optional): Number of epochs. Defaults to 10.
        eval_batch_size (int, optional): Batch size of validation and test set. Defaults to 100.
        train_batch_size (int, optional): Batch size of train set. Defaults to 100.
        tolerance (int, optional): Early stop tolerance. Defaults to 5.
        min_delta (float, optional): Early stop min delta. Defaults to 0.5.
        lr (float, optional): Learning rate. Defaults to 0.001.
        momentum (float, optional): Momentum of SGD optimizer. If 0.0 optimizer is used Adam optimizar. Defaults to 0.0.
        weight_decay (float, optional): _description_. Defaults to 0.
        dropout_rate_bb (float, optional): Dropout convolutional layers. Defaults to 0.2.
        dropout_rate_fc (float, optional): Dropout fc layers. Defaults to 0.5.
        fine_tune (bool, optional): If true we use pre-trained model. Defaults to False.
        transfer_learning (bool, optional): If true we use transfer learning. Defaults to False.
        test (bool, optional): If true we use 10 class careful selected. Defaults to False.
        increases_trainset (int, optional): Number of times that we increse trainig set with data augmentation. Defaults to 2.
        image_size (int, optional): Image size. Defaults to 64.
        step (int, optional): Step size for the learning rate scheduler. Defaults to 1.
    """
    classes = classes_list(c, num_classes, test)
    dataset = Tind(
        train_batch_size,
        eval_batch_size,
        classes=classes,
        increment=increases_trainset,
        image_size=image_size,
        step_size=step,
        gamma=gamma_train_batch_size,
    )
    if test:
        print(f"Test mode on {num_classes} classes: {classes}")
    else:
        print(f"Not test mode on {num_classes} classes: {classes}")
    print(
        f"Num epochs: {num_epochs}, Train batch size: {train_batch_size}, Eval batch size: {eval_batch_size}"
    )
    print(
        f"Train size: {len(dataset.train)}, Val size: {len(dataset.val)}, Test size: {len(dataset.test)}, Image size: {image_size}"
    )
    print(f"EarlyStopping tolerance:{tolerance} min delta:{min_delta}")
    print(
        f"Dropout rate basicBlock: {dropout_rate_bb}, Dropout rate final layer: {dropout_rate_fc}"
    )

    model = NNArchitecture.get_nn_architecture(
        type=architecture,
        num_classes=num_classes,
        fine_tune=fine_tune,
        transfer_learning=transfer_learning,
        dropout_rate_bb=dropout_rate_bb,
        dropout_rate_fc=dropout_rate_fc,
    )
    model = model.to(DEVICE)

    optimizer = Op(
        momentum=momentum,
        lr=lr,
        step=step,
        gamma_lr=gamma_lr,
        weight_decay=weight_decay,
        model=model,
    )
    print(f"BR scheduler: stepBR, step size: {step}, gamma: {gamma_train_batch_size}")
    criterion = Crit(num_classes)
    now = datetime.datetime.now()
    trainig_model(
        classes,
        dataset,
        model,
        optimizer,
        criterion,
        num_classes,
        num_epochs,
        tolerance,
        min_delta,
    )
    later = datetime.datetime.now()
    difference = (later - now).total_seconds()
    seconds = difference
    minutes = seconds / 60
    hours = minutes / 60
    if hours > 1:
        print(f"Training time: {hours:.3f} hours")
    elif minutes > 1:
        print(f"Training time: {minutes:.3f} minutes")
    else:
        print(f"Training time: {difference:.3f} seconds")


def trainig_model(
    classes: list,
    dataset: Tind,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    num_classes: int,
    num_epochs: int,
    tolerance: int,
    min_delta: float,
    res: Res,
):
    """
    Train the model on the dataset.

    Args:
        classes (list): List of classes to train on.
        dataset (TinyImageNetDataset): The dataset to train on.
        model (torch.nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        criterion (torch.nn.Module): The criterion to use for training.
        num_classes (int): Number of classes in the dataset.
        num_epochs (int): Number of epochs to train for.
        tolerance (int): Early stopping tolerance.
        min_delta (float): Minimum change in validation loss to be considered as an improvement.
    """

    train_stats = Statistics.Statistics(classes, len(dataset.train) / num_classes)
    val_stats = Statistics.Statistics(classes, len(dataset.val) / num_classes)

    inputs, labels = next(iter(dataset.train_dataloader))
    grid = torchvision.utils.make_grid(inputs)
    res.writer.add_image("images", grid, 0)
    res.writer.add_graph(model.to(DEVICE), inputs.to(DEVICE))

    Training.train_loop(
        dataset,
        model,
        optimizer,
        criterion,
        DEVICE,
        num_epochs,
        tolerance,
        min_delta,
        train_stats,
        val_stats,
        res,
    )

    res.createCharts(train_stats, val_stats)

    test_stats = Statistics.Statistics(classes, len(dataset.test) / num_classes)
    test_losses = Training.val(
        dataset.test_dataloader, model, criterion.criterion, DEVICE, test_stats
    )

    loss = sum(test_losses) / len(dataset.test_dataloader)
    test_stats.step(0, loss, "Test", verbose=True)
    res.createConfusionMatrix(test_stats, "Test conf matrix")
    test_stats.reset()
    res.close()


main(
    architecture,
    c,
    num_classes,
    num_epochs,
    eval_batch_size,
    train_batch_size,
    gamma_train_batch_size,
    tolerance,
    min_delta,
    lr,
    gamma_lr,
    momentum,
    weight_decay,
    dropout_rate_bb,
    dropout_rate_fc,
    fine_tune,
    transfer_learning,
    test,
    increases_trainset,
    image_size,
    step,
)
