import matplotlib.pyplot as plt
import Model.Statistics as Statistics
import seaborn as sn
import pandas as pd


def createConfusionMatrix(stat: Statistics, path):
    df_cm = pd.DataFrame(
        stat.get_confusion_matrix(),
        index=stat.get_classes(),
        columns=stat.get_classes(),
    )
    plt.figure(figsize=(15, 12))
    sn.heatmap(df_cm, annot=True, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig(path, format="pdf", bbox_inches="tight")
    plt.clf()


def createChart(xlabel, ylabel, xdata, ydata, path, dataNames=["chart"]):
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
