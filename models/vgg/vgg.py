from torch import nn
from torchvision import models
from collections import OrderedDict

import torch

class PretrainedVGG19(nn.Module):

    """
        Pretrained VGG19, classifier adapted to 2-classes classification.

        Classifier composed of 3 FC blocks (FC layer, Relu, Dropout)

        First FC block : output of dim 1028
        Second FC block : output of dim 512
        Third FC block : output of dim 2

        ---------
        Input : tensor of size (B, 3, 224, 224)

        ---------
        Parameters

        dropout : dropout rate.
    """

    def __init__(self, dropout=0.5):

        super(PretrainedVGG19, self).__init__()

        self.pretrained_model = models.vgg19_bn(pretrained=True)

        self.pretrained_model.classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 1028)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(dropout)),
                          ('fc2', nn.Linear(1028, 512)),
                          ('relu2', nn.ReLU()),
                          ('dropout2', nn.Dropout(dropout)),
                          ('fc3', nn.Linear(512, 2))
                          ]))

    def forward(self, x):

        x = self.pretrained_model(x)

        return x

if __name__ == "__main__":

    print("Sanity test :")

    print("Generating a batch of random images of size (50, 3, 224, 224)")

    x = torch.randn(size=(50, 3, 224, 224))

    print("Passing it to model ...")

    model = PretrainedVGG19()

    print("Output size should be (50, 2)")
    print(f'Output shape : {model(x).shape}')
