import matplotlib.pyplot as plt

def createChart(xlabel, ylabel, xdata, ydata, path, dataNames=["chart"]):
    plt.xlabel(xlabel, fontsize=10)
    plt.ylabel(ylabel, fontsize=10)

    for index, name in enumerate(dataNames):
        plt.plot(
            xdata,
            ydata[index],
            marker="o",
            linestyle="dashed",
            linewidth=2,
            markersize=6,
            label=name,
        )
    plt.title(f"Training results", fontsize=12)
    if dataNames[0] != "chart":
        plt.legend()
    plt.savefig(path, format="pdf", bbox_inches="tight")
    plt.clf()
