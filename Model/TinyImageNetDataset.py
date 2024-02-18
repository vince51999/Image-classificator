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
        mean, std = TinyImageNetDataset.__mean_std(classes)
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        train = TinyImageNet(
            Path("~/.torchvision/tinyimagenet/"),
            split="train",
            imagenet_idx=False,
            transform=transform,
        )
        val = TinyImageNet(
            Path("~/.torchvision/tinyimagenet/"),
            split="val",
            imagenet_idx=False,
            transform=transform,
        )
        if classes is not None and len(classes) < 200:
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

    def __mean_std(classes):
        if classes is None or len(classes) > 100:
            return [0.497, 0.500, 0.502], [0.137, 0.139, 0.144]
        train = TinyImageNet(
            Path("~/.torchvision/tinyimagenet/"),
            split="train",
            imagenet_idx=False,
            transform=None,
        )
        train = TinyImageNetDataset.__split_classes(train, classes, 500)
        train_dataloader = DataLoader(train, batch_size=1, shuffle=False)
        return TinyImageNetDataset.__mean_std_rgb_channels(
            train_dataloader, "mean"
        ), TinyImageNetDataset.__mean_std_rgb_channels(train_dataloader, "std")

    def __mean_std_rgb_channels(dataloader, t: str):
        """
        Calculate the mean of the RGB channels
        """
        r = 0
        g = 0
        b = 0
        length = len(dataloader)
        for i, data in enumerate(dataloader):
            # get the inputs; data is a list of [inputs, labels]
            batch, labels = data
            for sample in range(len(batch)):
                match t:
                    case "mean":
                        r += batch[sample][0][0].mean()
                        g += batch[sample][0][1].mean()
                        b += batch[sample][0][2].mean()
                    case "std":
                        r += batch[sample][0][0].std()
                        g += batch[sample][0][1].std()
                        b += batch[sample][0][2].std()
        return [r / length, g / length, b / length]

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
