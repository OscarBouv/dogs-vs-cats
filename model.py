from torch import nn
import torch

class BaseCNN(nn.Module):
    """
    Documentation # TODO
    """

    def __init__(self):

        super(BaseCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)
        self.maxpool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.maxpool2 = nn.MaxPool2d(2)

        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(in_features=18*18*64,
                                 out_features=64)

        self.linear2 = nn.Linear(in_features=64,
                                 out_features=1)

        self.relu_activation = nn.ReLU()

    def forward(self, x):

        x = self.maxpool1(self.relu_activation(self.conv1(x)))
        x = self.maxpool2(self.relu_activation(self.conv2(x)))
        x = self.relu_activation(self.linear1(self.flatten(x)))
        x = self.linear2(x)

        return x


if __name__ == "__main__":

    #Testing for image shape (80, 80, 3)

    x = torch.randn(50, 3, 80, 80)
    print(BaseCNN()(x).shape)
