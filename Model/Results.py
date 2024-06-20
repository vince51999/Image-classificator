import datetime
import pandas as pd
import seaborn as sn
import torch
import numpy as np
import matplotlib.pyplot as plt
import io
import os

from typing import List
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

from torch.utils.tensorboard import SummaryWriter


class Results:
    """
    The results class to store the results of the training and validation.
    
    Attributes:
        directory (str): The directory to store the results.
        output (str): The output file to store the results.
        logs (str): The base path for the logs.
        trainWriter (SummaryWriter): The writer for the training logs.
        valWriter (SummaryWriter): The writer for the validation logs.
    
    Methods:
        open(): Opens the writers for the training and validation logs.
        close(): Closes the writers.
        print(content): Prints the content and writes it to a file.
        createConfusionMatrix(cm, classes, name, epoch): Creates a confusion matrix using the provided statistics and saves it to TensorBoard.
        addScalar(label, name, value, step): Adds a scalar to the TensorBoard.
    """
    def __init__(self) -> None:
        """
        Initializes the results class.
        """
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
        self.logs = logs_directory

    def open(self) -> None:
        """
        Opens the writers for the training and validation logs.
        """
        log_dir = self.logs + "Train"
        self.trainWriter = SummaryWriter(log_dir)
        log_dir = self.logs + "Val"
        self.valWriter = SummaryWriter(log_dir)

    def close(self) -> None:
        """
        Closes the writers.
        """
        self.trainWriter.close()
        self.valWriter.close()

    def print(self, content):
        """
        Prints the content and writes it to a file.
        """
        print(content)
        with open(self.output, "a") as file:
            file.write(content + "\n")

    def createConfusionMatrix(
        self, cm: torch.Tensor, classes: List[int], name: str, epoch: int = 0
    ) -> None:
        """
        Creates a confusion matrix using the provided statistics and saves it to TensorBoard.

        Args:
            cm (torch.Tensor): The confusion matrix.
            classes (List[int]): The list of class labels.
            name (str): The name of the confusion matrix.
            epoch (int, optional): The epoch number. Defaults to 0.

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
        self.valWriter.add_image(
            name, tensor_image, dataformats="HWC", global_step=epoch
        )

    def addScalar(
        self,
        label: str,
        name: str,
        value: list,
        step: list,
    ):
        """
        Adds a scalar to the TensorBoard.

        Args:
            label (str): The label for the scalar.
            name (str): String to check if it is train or val.
            value (list): The value of the scalar.
            step (list): The step of the scalar.
            
        Returns:
            None
        """
        name = name.lower()
        if "train" in name:
            self.trainWriter.add_scalar(label, value, step)
        else:
            self.valWriter.add_scalar(label, value, step)
