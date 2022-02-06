from torch import nn
from collections import OrderedDict


class BaseCNN(nn.Module):
    """
    Documentation # TODO
    """

    def __init__(self):

        super(BaseCNN, self).__init__()

        self.conv_block1 = nn.Sequential(OrderedDict([
                          ('conv', nn.Conv2d(3, 64, kernel_size=3)),
                          ('batch_norm', nn.BatchNorm2d(num_features=64)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(0.5))
        ]))

        self.maxpool1 = nn.MaxPool2d(2)

        self.conv_block2 = nn.Sequential(OrderedDict([
                          ('conv', nn.Conv2d(64, 64, kernel_size=3)),
                          ('batch_norm', nn.BatchNorm2d(num_features=64)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(0.5))
        ]))

        self.maxpool2 = nn.MaxPool2d(2)

        self.flatten = nn.Flatten()

        self.fc_block1 = nn.Sequential(OrderedDict([
                          ('fc', nn.Linear(18 * 18 * 64, 64)),
                          ('relu', nn.ReLU())
        ]))

        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):

        x = self.maxpool1(self.conv_block1(x))
        x = self.maxpool2(self.conv_block2(x))
        x = self.flatten(x)
        x = self.fc_block1(x)
        x = self.fc2(x)

        return x
