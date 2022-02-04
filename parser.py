import argparse


def get_parser():

    parser = argparse.ArgumentParser("", description="Model training")

    # Data

    parser.add_argument("-i", "--image_size", default=80, type=int,
                        help="Input image input size.")

    # Model
    parser.add_argument("-cn", "--convnet", default="base_cnn", type=str,
                        help="Convnet classifier to train.")

    parser.add_argument("-b", "--batch_size", default=32, type=int,
                        help="Batch size.")

    parser.add_argument("-v", "--validation", default=0., type=float,
                        help="Validation split.")

    # Training
    parser.add_argument("-lr", "--lr", default=0.01, type=float,
                        help="Learning rate for optimizer.")

    parser.add_argument("-wd", "--weight_decay", default=0, type=float,
                        help="Weight decay.")

    parser.add_argument("-m", "--momentum", default=0.9, type=str,
                        help="Momentum for optimizer.")

    parser.add_argument("-opt", "--optimizer", default="sgd", type=str,
                        help="Optimizer for optimization process.")

    parser.add_argument("-e", "--epochs", default=10, type=int,
                        help="Number of epochs during training.")

    # Display

    parser.add_argument("-d", "display", default=True, type=bool,
                        help="Option to display training informations.")

    parser.add_argument("-dr", "--display_ratio", default=10, type=int,
                        help="Number of iterations between displaying steps.")

    return parser
