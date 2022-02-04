from torch import nn
import torch
from torchvision import models
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

        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):

        x = self.maxpool1(self.conv_block1(x))
        x = self.maxpool2(self.conv_block2(x))
        x = self.flatten(x)
        x = self.fc_block1(x)
        x = self.fc2(x)

        return x


class PretrainedVGG19(nn.Module):

    def __init__(self):

        super(PretrainedVGG19, self).__init__()

        self.pretrained_model = models.vgg19_bn(pretrained=True)

        self.pretrained_model.classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 1028)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(0.5)),
                          ('fc2', nn.Linear(1028, 512)),
                          ('relu2', nn.ReLU()),
                          ('dropout2', nn.Dropout(0.5)),
                          ('fc3', nn.Linear(512, 1))
                          ]))

    def forward(self, x):

        x = self.pretrained_model(x)

        return x


if __name__ == "__main__":

    # Testing BaseCCN with image shape (80, 80, 3)
    x = torch.randn(50, 3, 80, 80)
    model = BaseCNN()
    print(model(x).shape)

    # Testing PretrainedVGG19 image shape (224, 224, 3)
    x = torch.randn(50, 3, 224, 224)
    model = PretrainedVGG19()
    print(model(x).shape)
