import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


from data_handler import DogsVsCatsDataset
from model import BaseCNN
from train import TrainingSession

from parser import get_parser

def main(args):
    """
        Documentation #TODO
    """

    model = BaseCNN()

    train_transforms = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.485, 0.456, 0.406),
                                                              (0.229, 0.224, 0.225))])

    test_transforms = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.485, 0.456, 0.406),
                                                              (0.229, 0.224, 0.225))])

    train_dataset = DogsVsCatsDataset("data/train/", transform=train_transforms)
    test_dataset = DogsVsCatsDataset("data/test1/", transform=test_transforms)

    print("Downloaded dataset done.")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.SGD(model.parameters(),
                                lr = args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter()

    print(f"Running training session on {device} ...")

    training_session = TrainingSession(model=model,
                                       train_loader=train_loader,
                                       test_loader=test_loader,
                                       optimizer=optimizer,
                                       epochs=args.epochs,
                                       device=device,
                                       writer=writer,
                                       display_ratio=args.display_ratio,
                                       )

    training_session.run_train()


if __name__ == "__main__":

    args = get_parser().parse_args()
    main(args)
