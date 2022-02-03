import os
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import Resize

import numpy as np
import matplotlib.pyplot as plt

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

        image = read_image(img_path)
        image = Resize(self.image_size)(image)

        print(file_name.split(".")[0])

        label = int(file_name.split(".")[0] == "dog")

        if self.transform:
            image = self.transform(image)

        return image, label


if __name__ == "__main__":

    idx = np.random.randint(0, 25000)

    dataset = DogsVsCatsDataset("data/train/")
    img, label = dataset.__getitem__(idx)

    plt.imshow(img.permute(1, 2, 0).numpy())
    plt.savefig("test.png")
    
    print(label)