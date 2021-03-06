import os
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms import ToTensor
import PIL
import numpy as np


class DogsVsCatsDataset(Dataset):
    """
        Dataset, inherit from torch.utils.data.Dataset
        to load and preprocess dogs-vs-cats data from Kaggle.
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

        -----
        Parameters :

        dataset : dataset of class that inherits torch.utils.data.Dataset
        batch_size : batch size for loader
        """

        if self.validation_split > 0:

            indices = np.arange(0, len(dataset), dtype=int)
            np.random.shuffle(indices)

            # Define train and valid indices to subset dataset
            train_indices = indices[int(self.validation_split * len(dataset)):]
            val_indices = indices[0:int(self.validation_split * len(dataset))]

            # Define train and valid loader
            train_loader = DataLoader(Subset(dataset, train_indices),
                                      batch_size=batch_size,
                                      shuffle=True)

            val_loader = DataLoader(Subset(dataset, val_indices),
                                    batch_size=batch_size,
                                    shuffle=True)

            return train_loader, val_loader

        else:
            # No split
            train_loader = DataLoader(dataset,
                                      batch_size=batch_size,
                                      shuffle=True)

            return train_loader, None


if __name__ == "__main__":

    print("Sanity test :")

    ("Testing train/val split (10%) rate ...")

    dataset = DogsVsCatsDataset("data/train/", transform=ToTensor())

    val_split = ValSplit(0.1)
    train_loader, val_loader = val_split.get_train_val_loader(dataset, 1)

    print(f"Number of train images {len(train_loader)}")
    print(f"Number of valid images {len(val_loader)}")
