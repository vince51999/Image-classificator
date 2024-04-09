import datetime
from typing import List
import pandas as pd
import seaborn as sn
import torch
import numpy as np
import matplotlib.pyplot as plt
import io
import os
from PIL import Image

from torch.utils.tensorboard import SummaryWriter


class Results:
    def __init__(self) -> None:
        directory = "./results"
        if not os.path.exists(directory):
            os.makedirs(directory)
        now = datetime.datetime.now()
        directory = "./results/res_" + str(now.strftime("%d_%m_%Y__%H_%M_%S"))

        if not os.path.exists(directory):
            os.makedirs(directory)
        self.directory = directory
        output_file = directory + "/output.txt"
        with open(output_file, "w") as file:
            file.write("")
        self.output = output_file

        logs_directory = directory + "/logs"
        if not os.path.exists(logs_directory):
            os.makedirs(logs_directory)
        self.logs = logs_directory

    def open(self) -> None:
        self.writer = SummaryWriter(self.logs)

    def close(self) -> None:
        self.writer.close()

    def print(self, content):
        print(content)
        with open(self.output, "a") as file:
            file.write(content + "\n")

    def createConfusionMatrix(
        self, cm: torch.Tensor, classes: List[int], name: str, epoch: int = 0
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
            cm,
            index=classes,
            columns=classes,
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
        self.writer.add_image(name, tensor_image, dataformats="HWC", global_step=epoch)

    def addScalar(
        self,
        name: str,
        value: list,
        step: list,
    ):
        self.writer.add_scalar(name, value, step)
