import argparse
import numpy as np


def get_parser():

    """
        Documentation #TODO
    """

    parser = argparse.ArgumentParser("", description="Model predict.")

    # Data
    parser.add_argument("-d", "--directory_path", default="./data/test1/",
                        type=str, help="Image directory path.")

    parser.add_argument("-idx", "--img_idx", default=np.random.randint(1000),
                        type=int, help="Index image in directory to predict")

    parser.add_argument("-i", "--image_size", default=224, type=int,
                        help="Input image size.")

    # Model loading
    parser.add_argument("-p", "--model_path", default="model_vgg.pth",
                        type=str, help="Model path for loading.")

    return parser
