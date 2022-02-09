import argparse


def get_parser():

    """
        Outputs argument parser for train_main script.
    """

    parser = argparse.ArgumentParser("", description="Model training")

    # Model
    parser.add_argument("-cn", "--conv_net", default="base_cnn", type=str,
                        help="Convnet classifier to train.")

    parser.add_argument("-d", "--dropout", default=0.2, type=float,
                        help="Dropout probability in convnet.")

    # Data
    parser.add_argument("-v", "--validation_split", default=0.1, type=float,
                        help="Validation split for training.")

    parser.add_argument("-b", "--batch_size", default=32, type=int,
                        help="Batch size.")

    # Training
    parser.add_argument("-lr", "--lr", default=1e-3, type=float,
                        help="Learning rate for optimizer.")

    parser.add_argument("-wd", "--weight_decay", default=0, type=float,
                        help="Weight decay for Adam Optimizer.")

    parser.add_argument("-bet", "--betas", default=(0.9, 0.999), nargs=2, type=float,
                        help="Beta parameters for Adam optimizer.")

    parser.add_argument("-e", "--epochs", default=10, type=int,
                        help="Number of epochs during training.")

    # Model saving
    parser.add_argument("-p", "--model_path", default="models/model.pth",
                        type=str, help="Model path for saving.")

    return parser
