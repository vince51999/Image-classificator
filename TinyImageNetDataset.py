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
