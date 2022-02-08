import argparse
import numpy as np


def get_parser():

    """
        Outputs argument parser for predict_main script.
    """

    parser = argparse.ArgumentParser("", description="Model predict.")

    # Model
    parser.add_argument("-cn", "--conv_net", default="vgg", type=str,
                        help="Trained convnet classifier used for prediction.")

    # Data
    parser.add_argument("-i", "--img_path", default="./data/test1/1.jpg",
                        type=str, help="Image to predict directory path.")

    # Model loading
    parser.add_argument("-p", "--model_path", default="./models/model_vgg.pth",
                        type=str, help="Model path for loading.")

    return parser
