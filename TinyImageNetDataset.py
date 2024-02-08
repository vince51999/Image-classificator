from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from tinyimagenet import TinyImageNet
from pathlib import Path


class TinyImageNetDataset(Dataset):
    """
    Tiny ImageNet contains 100000 images of 200 classes (500 for each class)
    downsized to 64x64 colored images.
    Each class has 500 training images, 50 validation images and 50 test images.
    Test set is not labeled.

    The format of each image is: (3, 64, 64).
    """

    def __init__(self, batch_size):
        split = "train"
        train = TinyImageNet(
            Path("~/.torchvision/tinyimagenet/"), split=split, imagenet_idx=False
        )

        split = "val"
        val = TinyImageNet(
            Path("~/.torchvision/tinyimagenet/"), split=split, imagenet_idx=False
        )

        split = "test"
        test = TinyImageNet(
            Path("~/.torchvision/tinyimagenet/"), split=split, imagenet_idx=False
        )

        self.train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True)
        self.val_dataloader = DataLoader(val, batch_size=batch_size, shuffle=False)
        self.test_dataloader = DataLoader(test, batch_size=batch_size, shuffle=False)

    def __split_classes(dataset, num_classes, itr_break=500):
        """
        Split the dataset into classes
        """
        new_dataset = []
        for c in num_classes:
            itr = 0
            for i in range(c * itr_break, (c + 1) * itr_break):
                new_dataset.append(dataset[i])
                itr += 1
        return new_dataset

    def __create_testset(dataset, num_classes, num_el_per_class, split=0.1):
        """
        Split the dataset into test and otherset
        """
        testset = []
        otherset = []
        for c in range(num_classes):
            itr = 0
            for i in range(c * num_el_per_class, (c + 1) * num_el_per_class):
                if itr < split * num_el_per_class:
                    testset.append(dataset[i])
                else:
                    otherset.append(dataset[i])
                itr += 1
        return otherset, testset
