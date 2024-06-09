from typing import List
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from tinyimagenet import TinyImageNet
from pathlib import Path
from Model.Results import Results as Res


class TinyImageNetDataset(Dataset):
    """
    Tiny ImageNet contains 100000 images of 200 classes (500 for each class)
    downsized to 64x64 colored images.
    Each class has 500 training images, 50 validation images and 50 test images.
    Test set is not labeled.

    The format of each image is: (3, 64, 64).
    """

    def __init__(
        self,
        res: Res,
        train_batch_size: int,
        eval_batch_size: int,
        classes=None,
        increment: int = 2,
        image_size: int = 64,
        step_size: int = 2,
        gamma=2,
    ):
        self.image_size = image_size
        self.increment = increment
        self.train_batch_size = train_batch_size
        self.step_size = step_size
        self.gamma = gamma
        self.itr = 0
        self.res = res
        res.print(f"BR scheduler: stepBR, step size: {step_size}, gamma: {gamma}")
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.480, 0.448, 0.398], std=[0.230, 0.226, 0.226]
                ),
            ]
        )
        if image_size > 64:
            transform = transforms.Compose(
                [
                    transforms.Resize(
                        [image_size, image_size],
                        interpolation=transforms.InterpolationMode.BICUBIC,
                    ),
                    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(1)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.480, 0.448, 0.398], std=[0.230, 0.226, 0.226]
                    ),
                ]
            )
        train = TinyImageNet(
            Path("~/.torchvision/tinyimagenet/"),
            split="train",
            imagenet_idx=False,
            transform=transforms.RandomHorizontalFlip(p=0),
        )
        val = TinyImageNet(
            Path("~/.torchvision/tinyimagenet/"),
            split="val",
            imagenet_idx=False,
            transform=transform,
        )
        if classes is not None and len(classes) < 200:
            self.num_classes = len(classes)
            train = self.__split_classes(train, classes, 500)
            val = self.__split_classes(val, classes, 50)
            train = self.__update_labels(train, classes)
            val = self.__update_labels(val, classes)
        else:
            self.num_classes = 200

        # pointer to same memory
        self.val = val
        self.test = val

        self.train = train
        self.aug_train = self.__get_trainset(transform, self.train)

        self.train_dataloader = DataLoader(
            self.aug_train, batch_size=train_batch_size, shuffle=True
        )
        self.val_dataloader = DataLoader(
            self.val, batch_size=eval_batch_size, shuffle=False
        )
        self.test_dataloader = self.val_dataloader

    def __split_classes(self, dataset, num_classes, itr_break=500):
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

    def __update_labels(self, dataset, classes):
        """
        Update the labels of the dataset
        """
        tmp = []
        for i in range(len(dataset)):
            tmp.append(tuple([dataset[i][0], classes.index(dataset[i][1])]))
        return tmp

    def __get_trainset(
        self,
        basic_trasform,
        train=None,
    ):
        policy = transforms.AutoAugmentPolicy.IMAGENET
        augmenter = transforms.AutoAugment(policy)
        transform = transforms.Compose(
            [
                augmenter,
                basic_trasform,
            ]
        )
        new_train = self.__apply_transforms(train, transform)
        if self.increment < 0:
            self.increment = 0
        if self.increment == 0 or train is None:
            return new_train
        for j in range(self.increment):
            new_train += self.__apply_transforms(train, transform)
        return new_train

    def __apply_transforms(self, dataset, transform):
        newset = []
        for i in range(len(dataset)):
            newset.append(tuple([transform(dataset[i][0]), dataset[i][1]]))
        return newset

    def step(self, verbose: bool = False):
        if self.step_size <= 0:
            return
        self.itr += 1
        if self.itr % self.step_size == 0:
            self.train_batch_size = int(self.train_batch_size * self.gamma)
            if self.train_batch_size > 256:
                self.train_batch_size = 256
            if self.train_batch_size < 4:
                self.train_batch_size = 4
            self.train_dataloader = DataLoader(
                self.aug_train, batch_size=self.train_batch_size, shuffle=True
            )
        if verbose:
            self.res.print(f"Batch size: {self.train_batch_size}")

    def augumentation(self, verbose: bool = False):
        if self.itr % self.step_size == 0:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.480, 0.448, 0.398], std=[0.230, 0.226, 0.226]
                    ),
                ]
            )
            if self.image_size > 64:
                transform = transforms.Compose(
                    [
                        transforms.Resize(
                            [self.image_size, self.image_size],
                            interpolation=transforms.InterpolationMode.BICUBIC,
                        ),
                        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(1)),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.480, 0.448, 0.398], std=[0.230, 0.226, 0.226]
                        ),
                    ]
                )
            self.aug_train = self.__get_trainset(transform, self.train)

            self.train_dataloader = DataLoader(
                self.aug_train, batch_size=self.train_batch_size, shuffle=True
            )
            if verbose:
                self.res.print(f"Train set augmented.")

    def state_dict(self):
        return [self.itr, self.train_batch_size]

    def load_state_dict(self, state_dict: List[int]):
        self.itr = state_dict[0]
        self.train_batch_size = state_dict[1]
