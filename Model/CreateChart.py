import pandas as pd
import seaborn as sn
import torch
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image
from Model.Statistics import Statistics

from torch.utils.tensorboard import SummaryWriter


def createConfusionMatrix(
    stat: Statistics, name: str, writer: SummaryWriter, epoch: int = 0
) -> None:
    """
    Creates a confusion matrix using the provided statistics and saves it to TensorBoard.

    Args:
        stat (Statistics): An instance of the Statistics class containing the necessary data for creating the confusion matrix.
        writer (SummaryWriter): TensorBoard SummaryWriter object.

    Returns:
        None
    """
    df_cm = pd.DataFrame(
        stat.get_confusion_matrix(),
        index=stat.get_classes(),
        columns=stat.get_classes(),
    )

    # Plot the confusion matrix
    plt.figure(figsize=(15 + 0.7 * len(df_cm.columns), 12 + 0.7 * len(df_cm.index)))
    sn.heatmap(df_cm, annot=True, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    # Convert the plot to a NumPy array
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    cm_image_bytes = buf.getvalue()
    buf.close()

    # Convert the byte buffer to a NumPy array
    img = Image.open(io.BytesIO(cm_image_bytes))
    np_array = np.array(img)

    # Convert the NumPy array to a tensor
    tensor_image = torch.tensor(np_array)

    # Write the image to TensorBoard
    writer.add_image(name, tensor_image, dataformats="HWC", global_step=epoch)


def createChart(
    ylabel: str,
    xdata: list,
    ydata: list,
    writer: SummaryWriter,
    dataNames: list = ["chart"],
):
    """
    Creates a chart using the provided data and saves it to TensorBoard.

    Args:
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        xdata (list): The data for the x-axis.
        ydata (list): The data for the y-axis.
        writer (SummaryWriter): TensorBoard SummaryWriter object to write the chart.
        dataNames (list, optional): The names of the data series. Defaults to ["chart"].

    Returns:
        None
    """
    for index, name in enumerate(dataNames):
        for x, y in zip(xdata, ydata[index]):
            writer.add_scalar(f"{ylabel}/{name}", y, x)


def createCharts(
    train_stats: Statistics,
    val_stats: Statistics,
    writer: SummaryWriter,
) -> None:
    """
    Creates charts for various statistics using the provided training and validation statistics.

    Args:
        train_stats (Statistics): Training statistics object containing the data for each epoch.
        val_stats (Statistics): Validation statistics object containing the data for each epoch.

    Returns:
        None
    """
    epochs = train_stats.epochs
    createChart(
        "Losses",
        epochs,
        [train_stats.losses, val_stats.losses],
        writer,
        ["train_losses", "val_losses"],
    )
    createChart(
        "Accuracy",
        epochs,
        [train_stats.accuracy, val_stats.accuracy],
        writer,
        ["train_accuracy", "val_accuracy"],
    )
    if len(train_stats.classes) > 1:
        createChart(
            "F-Measure",
            epochs,
            [train_stats.f_measure, val_stats.f_measure],
            writer,
            ["train_f_measure", "val_f_measure"],
        )
        createChart(
            "Recall",
            epochs,
            [train_stats.recall, val_stats.recall],
            writer,
            ["train_recall", "val_recall"],
        )
        createChart(
            "Precision",
            epochs,
            [train_stats.precision, val_stats.precision],
            writer,
            ["train_precision", "val_precision"],
        )
