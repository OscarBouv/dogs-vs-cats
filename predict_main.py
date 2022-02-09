import torch
from models.vgg.vgg import PretrainedVGG19
from models.base_cnn.base_cnn import BaseCNN
import PIL

from torchvision.transforms import Resize, ToTensor, Normalize, Compose
from parsers.predict_parser import get_parser


def predict(output, classes):

    """
        Outputs label given model output and classes dict.
    """

    if output.size()[1] > 1:
        y_pred = output.argmax(dim=1, keepdim=True).item()

    else:
        y_pred = int(output > 0)

    return classes[y_pred]


def main(args):

    """
       Main function for showing and predicting image label.

       -------
       Parameters

       args : parsed arguments from predict_parser.
    """

    img = PIL.Image.open(args.img_path)

    transform = Compose([Resize((224, 224)),
                         ToTensor(),
                         Normalize((0.485, 0.456, 0.406),
                                   (0.229, 0.224, 0.225))])

    input = torch.unsqueeze(transform(img), dim=0)

    if args.conv_net == "base_cnn":
        model = BaseCNN()
        model.load_state_dict(torch.load(args.model_path, map_location="cpu"))

    if args.conv_net == "vgg":
        model = PretrainedVGG19()
        model.load_state_dict(torch.load(args.model_path, map_location="cpu"))

    model.eval()

    classes = {0: "cat", 1: "dog"}
    output = model(input)

    pred = predict(output, classes)

    img.show()
    print(pred)


if __name__ == "__main__":

    args = get_parser().parse_args()
    main(args)

