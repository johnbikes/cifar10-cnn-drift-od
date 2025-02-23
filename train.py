import os
import sys
import tarfile

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.datasets.utils import download_url
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import make_grid
import matplotlib
import matplotlib.pyplot as plt

from model import Cifar10CnnModel
from data import get_default_device, to_device, DeviceDataLoader


def show_example(img, label, dataset):
    print('Label: ', dataset.classes[label], "("+str(label)+")")
    plt.imshow(img.permute(1, 2, 0))
    plt.show()

def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
        break
    plt.show()

@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs')
    plt.show()

def plot_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')
    plt.show()

def main():
    dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
    download_url(dataset_url, '.')

    # Extract from archive
    with tarfile.open('./cifar10.tgz', 'r:gz') as tar:
        tar.extractall(path='./data')

    data_dir = './data/cifar10'

    print(f"sub dirs: {os.listdir(data_dir)}")
    classes = os.listdir(data_dir + "/train")
    print(f"{classes = }")

    # airplane
    airplane_files = os.listdir(data_dir + "/train/airplane")
    print('No. of training examples for airplanes:', len(airplane_files))
    print(f"{airplane_files[:5] = }")

    # ship
    ship_test_files = os.listdir(data_dir + "/test/ship")
    print("No. of test examples for ship:", len(ship_test_files))
    print(f"{ship_test_files[:5]}")

    # load train
    dataset = ImageFolder(data_dir+'/train', transform=ToTensor())

    img, label = dataset[0]
    print(img.shape, label) # torch.Size([3, 32, 32]) 0
    print(img)

    # dataset will have knowledge of the classes
    print(dataset.classes)

    # args fun
    print(*([1, 2] + [3]))

    # show some examples - airplane
    show_example(*(list(dataset[0]) + [dataset]))
    show_example(*(list(dataset[1099]) + [dataset]))

    random_seed = 42
    torch.manual_seed(random_seed)

    # splits
    val_size = 5000
    train_size = len(dataset) - val_size

    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    print(f"{len(train_ds) = }, {len(val_ds) = }")

    batch_size=128

    # set up train and val data loaders
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True)
    
    show_batch(train_dl)

    print(f"{torch.cuda.is_available() = }")
    
    model = Cifar10CnnModel()
    print(model)

    # verify output shape
    for images, labels in train_dl:
        print('images.shape:', images.shape)
        out = model(images)
        print('out.shape:', out.shape)
        print('out[0]:', out[0])
        break

    device = get_default_device()
    print(f"{device = }")

    # data and model to device
    train_dl = DeviceDataLoader(train_dl, device)
    val_dl = DeviceDataLoader(val_dl, device)
    to_device(model, device)

    # train
    history = []
    train_loader = train_dl
    val_loader = val_dl
    num_epochs = 10 # enough to overfit
    lr = 0.001
    # opt_func = torch.optim.SGD
    opt_func = torch.optim.Adam
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(num_epochs):
        # Training Phase 
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    
    print(f"{history = }")
    
    plot_accuracies(history)
    
    plot_losses(history)

    # save model
    torch.save(model.state_dict(), 'cifar10-cnn.pt')


if __name__ == '__main__':
    main()