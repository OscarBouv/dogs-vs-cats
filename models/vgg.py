from torch import nn
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
                          ('fc3', nn.Linear(512, 2))
                          ]))

    def forward(self, x):

        x = self.pretrained_model(x)

        return x
