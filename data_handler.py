import os
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.io import read_image
from torchvision.transforms import ToTensor

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
        #image = image.resize(self.image_size)

        label = int(file_name.split(".")[0] == "dog")

        if self.transform:
            image = self.transform(image)

        return image, label

class ValSplitLoader():

    def __init__(self, dataset):
        self.dataset = dataset

    def get_train_val_loader(self, batch_size, validation_split):

        indices = np.arange(len(self.dataset))
        np.random.shuffle(indices)

        train_indices = indices[0:validation_split * len(self.dataset)]
        val_indices = indices[validation_split * len(self.dataset):]

        train_loader = DataLoader(Subset(self.dataset, train_indices),
                                  batch_size=batch_size,
                                  shuffle=True)

        val_loader = Subset(self.dataset, val_indices)

        return train_loader, val_loader






        


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