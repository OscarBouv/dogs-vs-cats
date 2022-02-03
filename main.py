import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


from data_handler import DogsVsCatsDataset
from model import BaseCNN
from train import TrainingSession

def main():
    """
        Documentation #TODO
    """

    model = BaseCNN()

    batch_size = 32

    base_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.485, 0.456, 0.406),
                                                              (0.229, 0.224, 0.225))])

    train_dataset = DogsVsCatsDataset("data/train/", transform=base_transform)
    test_dataset = DogsVsCatsDataset("data/test1/", transform=base_transform)

    print("Downloaded dataset done.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.SGD(model.parameters(),
                                    #lr=args.lr,
                                    lr = 0.1,
                                    momentum=0.9,
                                    weight_decay=1e-4,
                                    nesterov=True)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=len(train_loader) * 10,
                                                           eta_min=0.03 * 0.004)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter()

    training_session = TrainingSession(model=model,
                                       train_loader=train_loader,
                                       test_loader=test_loader,
                                       optimizer=optimizer,
                                       scheduler=scheduler,
                                       epochs=4,
                                       device=device,
                                       writer=writer)

    training_session.run_train()


if __name__ == "__main__":

    main()
