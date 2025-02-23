from pathlib import Path

import torch
from torch import nn
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt

from model import Cifar10CnnModel
from data import get_default_device, to_device, DeviceDataLoader
from train import evaluate


def predict_image(img, model, dataset, device):
    # Convert to a batch of 1
    xb = to_device(img.unsqueeze(0), device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    _, preds  = torch.max(yb, dim=1)
    # Retrieve the class label
    return dataset.classes[preds[0].item()]

def main():
    weights_path: Path = Path('cifar10-cnn.pt')
    assert weights_path.is_file(), f"{weights_path} does not exist"

    device = get_default_device()
    print(f"{device = }")

    model = to_device(Cifar10CnnModel(), device)
    model.load_state_dict(torch.load(str(weights_path)))

    print(model)

    # verify weights loaded
    batch_size = 128
    data_dir = './data/cifar10'
    test_dataset = ImageFolder(data_dir+'/test', transform=ToTensor())
    test_loader = DeviceDataLoader(DataLoader(test_dataset, batch_size*2), device)
    results = evaluate(model, test_loader)
    print(f"{results = }") # {'val_loss': 0.8982319235801697, 'val_acc': 0.765332043170929}

    # frog
    dataset = ImageFolder(data_dir+'/train', transform=ToTensor())
    img, label = test_dataset[6153]
    plt.imshow(img.permute(1, 2, 0))
    plt.show()
    # Label: frog , Predicted: frog
    print('Label:', dataset.classes[label], ', Predicted:',
           predict_image(img, model, dataset, device))

    # drift
    # can we look at some previous features
    for name, layer in model.network.named_parameters():
        print(f"Layer name: {name}, Layer: {layer}")
    for i in range(5, 0, -1):
        # print(f"{model.network[-i].shape}")
        print(f"{-i}, {model.network[-i] = }")

    # cut off last linear layer
    model_network_short = nn.Sequential(*list(model.network)[:-1])
    print(model_network_short)
    img_input = to_device(img.unsqueeze(0), device)
    features_out = model_network_short(img_input)
    print(f"{features_out.shape = }") # features_out.shape = torch.Size([1, 512])

if __name__ == '__main__':
    main()