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

    def __init__(self, train_batch_size, eval_batch_size, classes=None):
        train = TinyImageNet(
            Path("~/.torchvision/tinyimagenet/"), split="train", imagenet_idx=False
        )
        val = TinyImageNet(
            Path("~/.torchvision/tinyimagenet/"), split="val", imagenet_idx=False
        )
        if classes is not None:
            self.num_classes = len(classes)
            train = TinyImageNetDataset.__split_classes(train, classes, 500)
            val = TinyImageNetDataset.__split_classes(val, classes, 50)
            train = TinyImageNetDataset.__update_labels(train, classes)
            val = TinyImageNetDataset.__update_labels(val, classes)
        else:
            self.num_classes = 200

        train, test = TinyImageNetDataset.__create_testset(
            train, self.num_classes, 500, split=0.1
        )

        self.train_dataloader = DataLoader(
            train, batch_size=train_batch_size, shuffle=True
        )
        self.val_dataloader = DataLoader(val, batch_size=eval_batch_size, shuffle=False)
        self.test_dataloader = DataLoader(
            test, batch_size=eval_batch_size, shuffle=False
        )

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

    def __update_labels(dataset, classes):
        """
        Update the labels of the dataset
        """
        tmp = []
        for i in range(len(dataset)):
            tmp.append(tuple([dataset[i][0], classes.index(dataset[i][1])]))
        return tmp
