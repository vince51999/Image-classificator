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

    def __init__(self, train_batch_size, eval_batch_size, classes=None, increment=2):
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
        test = TinyImageNet(
            Path("~/.torchvision/tinyimagenet/"),
            split="val",
            imagenet_idx=False,
            transform=transform,
        )
        if classes is not None and len(classes) < 200:
            self.num_classes = len(classes)
            train = TinyImageNetDataset.__split_classes(train, classes, 500)
            test = TinyImageNetDataset.__split_classes(test, classes, 50)
            train = TinyImageNetDataset.__update_labels(train, classes)
            test = TinyImageNetDataset.__update_labels(test, classes)
        else:
            self.num_classes = 200

        train, val = TinyImageNetDataset.__split_set(
            train, self.num_classes, 500, split=0.2
        )

        train = TinyImageNetDataset.__get_trainset(increment, train)
        self.train_dataloader = DataLoader(
            train, batch_size=train_batch_size, shuffle=True
        )
        self.val_dataloader = DataLoader(val, batch_size=eval_batch_size, shuffle=False)
        self.test_dataloader = DataLoader(
            test, batch_size=eval_batch_size, shuffle=False
        )

    def __mean_std(classes):
        if classes is None or len(classes) > 30:
            return [0.480, 0.448, 0.398], [0.230, 0.226, 0.226]
        train = TinyImageNet(
            Path("~/.torchvision/tinyimagenet/"),
            split="train",
            imagenet_idx=False,
            transform=None,
        )
        train = TinyImageNetDataset.__split_classes(train, classes, 500)
        return TinyImageNetDataset.__mean_std_rgb_channels(
            train, "mean"
        ), TinyImageNetDataset.__mean_std_rgb_channels(train, "std")

    def __mean_std_rgb_channels(dataloader, type: str):
        """
        Calculate the mean of the RGB channels
        """
        r = 0
        g = 0
        b = 0
        length = len(dataloader)
        for i, data in enumerate(dataloader):
            # get the inputs; data is a list of [inputs, labels]
            if type == "mean":
                r += data[0][0].mean()
                g += data[0][1].mean()
                b += data[0][2].mean()
            elif type == "std":
                r += data[0][0].std()
                g += data[0][1].std()
                b += data[0][2].std()
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

    def __split_set(dataset, num_classes, num_el_per_class, split=0.1):
        """
        Split the dataset into 2 datasets
        """
        newset = []
        oldset = []
        for c in range(num_classes):
            itr = 0
            for i in range(c * num_el_per_class, (c + 1) * num_el_per_class):
                if itr < split * num_el_per_class:
                    newset.append(dataset[i])
                else:
                    oldset.append(dataset[i])
                itr += 1
        return oldset, newset

    def __update_labels(dataset, classes):
        """
        Update the labels of the dataset
        """
        tmp = []
        for i in range(len(dataset)):
            tmp.append(tuple([dataset[i][0], classes.index(dataset[i][1])]))
        return tmp

    def __get_trainset(
        increment: int = 0,
        train=None,
    ):
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size=(64, 64), scale=(0.8, 1.0), antialias=True
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=40),
            ]
        )
        if increment < 0:
            increment = 0
        if increment == 0 or train is None:
            return train
        new_train = train.copy()
        for j in range(increment):
            for i in range(len(train)):
                tmp = transform(train[i][0])
                new_train.append(tuple([tmp, train[i][1]]))
        return new_train
