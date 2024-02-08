import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

import TinyImageNetDataset
import Model
import Training
import Testing

TRAIN_BATCH_SIZE = 100  # increase / decrease according to GPU memeory
EVAL_BATCH_SIZE = 10  # increase / decrease according to GPU memeory
NUM_EPOCHS = 30  # number of epochs to train for
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    dataset = TinyImageNetDataset.TinyImageNetDataset(BATCH_SIZE)
    model = Model.get_model(type="resnet50", num_classes=200, wieghts=None)
    dataset = TinyImageNetDataset.TinyImageNetDataset(
        TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE, classes=classes
    )
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    Training.train_loop(dataset.train_dataloader,dataset.val_dataloader, model, optimizer, criterion, DEVICE, NUM_EPOCHS)
    
if __name__ == "__main__":
    main()