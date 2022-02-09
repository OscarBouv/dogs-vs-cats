import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from dataset import DogsVsCatsDataset, ValSplit
from models.base_cnn.base_cnn import BaseCNN
from models.vgg.vgg import PretrainedVGG19
from train import TrainingSession

from parsers.train_parser import get_parser


def main(args):
    """
        Main function to run training session.

        ------
        Parameters

        args : parsed arguments from train_parser
    """

    # Load model
    if args.conv_net == "base_cnn":
        print("Use Base CNN model.")
        model = BaseCNN(args.dropout)

    elif args.conv_net == "vgg":
        print("Use Pretrained VGG model.")
        model = PretrainedVGG19(args.dropout)

    # Define train transform
    train_transform = Compose([Resize((224, 224)),
                                ToTensor(),
                                Normalize((0.485, 0.456, 0.406),
                                          (0.229, 0.224, 0.225))])

    #Load dataset
    train_dataset = DogsVsCatsDataset("data/train/", transform=train_transform)

    print("Dataset loaded.")

    # Define validation split
    val_split = ValSplit(args.validation_split)

    # Define training and valid loader
    train_loader, val_loader = val_split.get_train_val_loader(train_dataset, args.batch_size)

    #Define optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay,
                                 betas=args.betas)

    # Define device (cuda if available)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Running training session on {device} ...")

    # Define training session
    training_session = TrainingSession(model=model,
                                       train_loader=train_loader,
                                       val_loader=val_loader,
                                       optimizer=optimizer,
                                       epochs=args.epochs,
                                       device=device,
                                       model_path=args.model_path
                                       )

    # Run session
    training_session.run()


if __name__ == "__main__":

    args = get_parser().parse_args()
    main(args)
