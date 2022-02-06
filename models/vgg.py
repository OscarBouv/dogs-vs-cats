from torch import nn
import torch
from torchvision import models
from collections import OrderedDict


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

    # Testing PretrainedVGG19 image shape (224, 224, 3)
    x = torch.randn(50, 3, 224, 224)
    model = PretrainedVGG19()
    print(model(x).shape)
