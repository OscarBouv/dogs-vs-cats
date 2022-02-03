import os
from matplotlib import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import ToTensor
import torch

import numpy as np
import matplotlib.pyplot as plt
import PIL

class DogsVsCatsDataset(Dataset):
    """
        Documentation #TODO
    """

    def __init__(self, img_dir, image_size=(80, 80), transform=None):

        self.img_dir = img_dir
        self.image_size=image_size
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.img_dir))

    def __getitem__(self, idx):

        file_name = os.listdir(self.img_dir)[idx]
        img_path = os.path.join(self.img_dir, file_name)

        image = PIL.Image.open(img_path)
        image = image.resize(self.image_size)


        #image = read_image(img_path)
        #image = Resize(self.image_size)(image)

        label = int(file_name.split(".")[0] == "dog")

        if self.transform:
            image = self.transform(image)

        return image, label

# class DogsVsCatLoader():

#     def __init__(self, path, transform=None, target_transform=None):

#         self.path = path
#         self.transform = transform
#         self.target_transform = target_transform

#     def get_loader(batch_size):

#         data = DogsVsCatsDataset(self.path, )

#         return DataLoader(dataset=self.dataset,
#                           batch_size=batch_size,
#                           shuffle=True,
#                           transforms=self.transform,
#                           )



if __name__ == "__main__":

    dataset = DogsVsCatsDataset("data/train/", transform = ToTensor())
    idx = np.random.randint(0, len(dataset))

    img, label = dataset.__getitem__(idx)

    print(type(img))
    print(img.shape)