import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from dataset import DogsVsCatsDataset, ValSplit
from models import BaseCNN, PretrainedVGG19
from train import TrainingSession

from parsers.train_parser import get_parser


def main(args):
    """
        Documentation #TODO
    """

    if args.conv_net == "base_cnn":
        model = BaseCNN()

    elif args.conv_net == "vgg":
        model = PretrainedVGG19()

    train_transforms = Compose([Resize((args.image_size, args.image_size)),
                                ToTensor(),
                                Normalize((0.485, 0.456, 0.406),
                                          (0.229, 0.224, 0.225))])


    train_dataset = DogsVsCatsDataset("data/train/", transform=train_transforms)

    print("Downloaded dataset done.")

    val_split = ValSplit(args.validation_split)
    train_loader, val_loader = val_split.get_train_val_loader(train_dataset, args.batch_size)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay,
                                 betas=(0.9, 0.999))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter()

    print(f"Running training session on {device} ...")

    training_session = TrainingSession(model=model,
                                       train_loader=train_loader,
                                       val_loader=val_loader,
                                       optimizer=optimizer,
                                       epochs=args.epochs,
                                       device=device,
                                       writer=writer,
                                       display_ratio=args.display_ratio,
                                       model_path=args.model_path
                                       )

    training_session.run_train()


if __name__ == "__main__":

    args = get_parser().parse_args()
    main(args)
