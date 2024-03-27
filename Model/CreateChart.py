import matplotlib.pyplot as plt
import Model.Statistics as Statistics
import seaborn as sn
import pandas as pd


def createConfusionMatrix(stat: Statistics, path: str):
    """
    Creates a confusion matrix using the provided statistics and saves it to the provided path.

    Args:
        stat (Statistics): An instance of the Statistics class containing the necessary data for creating the confusion matrix.
        path (str): The file path where the confusion matrix will be saved.

    Returns:
        None
    """
    df_cm = pd.DataFrame(
        stat.get_confusion_matrix(),
        index=stat.get_classes(),
        columns=stat.get_classes(),
    )
    size_x = 15 + 0.7 * len(df_cm.columns)
    size_y = 12 + 0.7 * len(df_cm.index)
    plt.figure(figsize=(size_x, size_y))
    sn.heatmap(df_cm, annot=True, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig(path, format="pdf", bbox_inches="tight")
    plt.clf()


def createChart(xlabel, ylabel, xdata, ydata, path, dataNames=["chart"]):
    """
    Creates a chart using the provided data and saves it to the provided path.

    Args:
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        xdata (list): The data for the x-axis.
        ydata (list): The data for the y-axis.
        path (str): The path where the chart will be saved.
        dataNames (list, optional): The names of the data series. Defaults to ["chart"].
    """
    plt.xlabel(xlabel, fontsize=10)
    plt.ylabel(ylabel, fontsize=10)

    for index, name in enumerate(dataNames):
        plt.plot(
            xdata,
            ydata[index],
            linestyle="solid",
            linewidth=2,
            label=name,
        )
    plt.title(f"Training results", fontsize=12)
    if dataNames[0] != "chart":
        plt.legend()
    plt.savefig(path, format="pdf", bbox_inches="tight")
    plt.clf()


def createCharts(train_stats: Statistics, val_stats: Statistics):
    """
    Creates charts for various statistics using the provided training and validation statistics.

    Args:
        train_stats (Statistics): Training statistics object containing the data for each epoch.
        val_stats (Statistics): Validation statistics object containing the data for each epoch.
    """
    epochs = train_stats.epochs
    createChart(
        "Epochs",
        "Losses",
        epochs,
        [train_stats.losses, val_stats.losses],
        "./results/loss.pdf",
        ["train_losses", "val_losses"],
    )
    createChart(
        "Epochs",
        "Accuracy",
        epochs,
        [train_stats.accuracy, val_stats.accuracy],
        "./results/accuracy.pdf",
        ["train_accuracy", "val_accuracy"],
    )
    createChart(
        "Epochs",
        "F-Measure",
        epochs,
        [train_stats.f_measure, val_stats.f_measure],
        "./results/f_measure.pdf",
        ["train_f_measure", "val_f_measure"],
    )
    createChart(
        "Epochs",
        "Recall",
        epochs,
        [train_stats.recall, val_stats.recall],
        "./results/recall.pdf",
        ["train_recall", "val_recall"],
    )
    createChart(
        "Epochs",
        "Precision",
        epochs,
        [train_stats.precision, val_stats.precision],
        "./results/precision.pdf",
        ["train_precision", "val_precision"],
    )
