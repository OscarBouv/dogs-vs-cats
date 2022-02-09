from torch import nn
from collections import OrderedDict

import torch
class BaseCNN(nn.Module):
    """
        Base CNN composed of 4 conv blocks (Conv layer, BatchNorm, Relu, Dropout)
        1 FC classification layer.

        First block : 64 output channels.
        Second block : 128 outputs channels.
        Third block : 256 outputs channels.
        Fourth block : 512 outputs channels.

        FC layer : output dim 100

        Classification layer : output dim 2

        -------
        Input : torch tensor of size (B, 3, 224)

        -------
        Parameters

        dropout : dropout rate.
    """

    def __init__(self, dropout=0.2):

        super(BaseCNN, self).__init__()

        self.conv_block1 = nn.Sequential(OrderedDict([
                          ('conv', nn.Conv2d(3, 64, kernel_size=3)),
                          ('batch_norm', nn.BatchNorm2d(num_features=64)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(dropout))
        ]))

        self.maxpool1 = nn.MaxPool2d(2)

        self.conv_block2 = nn.Sequential(OrderedDict([
                          ('conv', nn.Conv2d(64, 128, kernel_size=3)),
                          ('batch_norm', nn.BatchNorm2d(num_features=128)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(dropout))
        ]))

        self.maxpool2 = nn.MaxPool2d(2)

        self.conv_block3 = nn.Sequential(OrderedDict([
                          ('conv', nn.Conv2d(128, 256, kernel_size=3)),
                          ('batch_norm', nn.BatchNorm2d(num_features=256)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(dropout))
        ]))

        self.maxpool3 = nn.MaxPool2d(2)

        self.conv_block4 = nn.Sequential(OrderedDict([
                          ('conv', nn.Conv2d(256, 512, kernel_size=3)),
                          ('batch_norm', nn.BatchNorm2d(num_features=512)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(dropout))
        ]))

        self.maxpool4 = nn.MaxPool2d(2)

        self.flatten = nn.Flatten()

        self.fc_block1 = nn.Sequential(OrderedDict([
                          ('fc', nn.Linear(12 * 12 * 512, 100)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(dropout))
        ]))

        self.fc2 = nn.Linear(100, 2)

    def forward(self, x):

        x = self.maxpool1(self.conv_block1(x))
        x = self.maxpool2(self.conv_block2(x))
        x = self.maxpool3(self.conv_block3(x))
        x = self.maxpool4(self.conv_block4(x))
        x = self.flatten(x)
        x = self.fc_block1(x)
        x = self.fc2(x)

        return x


if __name__ == "__main__":

    print("Generating a batch of random images of size (50, 3, 224, 224)")

    x = torch.randn(size=(50, 3, 224, 224))
    model = BaseCNN()

    print("Output size should be (50, 2)")
    print(f'Output shape : {model(x).size}')

