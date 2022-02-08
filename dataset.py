import os
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

import PIL
import numpy as np
import matplotlib.pyplot as plt

class DogsVsCatsDataset(Dataset):
    """
        Class to handle special case of dataset loading for dogs-vs-cats data.
    """

    def __init__(self, img_dir, transform=None):

        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.img_dir))

    def __getitem__(self, idx):

        file_name = os.listdir(self.img_dir)[idx]
        img_path = os.path.join(self.img_dir, file_name)

        # load image
        image = PIL.Image.open(img_path)

        # If
        label = int(file_name.split(".")[0] == "dog")

        if self.transform:
            image = self.transform(image)

        return image, label

class ValSplit():

    """
        Validation split class. Random split.
    """

    def __init__(self, validation_split):
        self.validation_split = validation_split

    def get_train_val_loader(self, dataset, batch_size=32):
        """
            Return train and validation dataloaders for training.
        """

        if self.validation_split > 0:

            indices = np.arange(0, len(dataset), dtype=int)
            np.random.shuffle(indices)

            train_indices = indices[int(self.validation_split * len(dataset)):]
            val_indices = indices[0:int(self.validation_split * len(dataset))]

            train_loader = DataLoader(Subset(dataset, train_indices),
                                      batch_size=batch_size,
                                      shuffle=True)

            val_loader = DataLoader(Subset(dataset, val_indices),
                                    batch_size=batch_size,
                                    shuffle=True)

            return train_loader, val_loader

        else:
            #No split
            train_loader = DataLoader(dataset,
                                      batch_size=batch_size,
                                      shuffle=True)

            return train_loader, None


if __name__ == "__main__":

    transform = Compose([Resize((224, 224)),
                         ToTensor()
                         ])

    dataset = DogsVsCatsDataset("data/train/", transform=transform)

    val_split = ValSplit(0.1)
    train_loader, val_loader = val_split.get_train_val_loader(dataset, 1)

    x, y = next(iter(train_loader))

    plt.imshow(x[0, :, :, :].permute(1, 2, 0))
    plt.savefig("test_train.png")

    print(y)

    x, y = next(iter(val_loader))

    plt.imshow(x[0, :, :, :].permute(1, 2, 0))
    plt.savefig("test_val.png")

    print(y)