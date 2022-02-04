import torch
from models import PretrainedVGG19
import os
import PIL

from torchvision.transforms import Resize, ToTensor, Normalize, Compose
from parsers.serve_parser import get_parser


def predict(output, classes):

    if len(output) > 1:
        y_pred = torch.argmax(output, dim=1)

    else:
        y_pred = int(output > 0)

    return classes[y_pred]


def main(args):

    file_name = os.listdir(args.directory_path)[args.img_idx]

    IMG_PATH = os.path.join(args.directory_path, file_name)

    img = PIL.Image.open(IMG_PATH)

    transform = Compose([Resize((args.image_size, args.image_size)),
                         ToTensor(),
                         Normalize((0.485, 0.456, 0.406),
                                   (0.229, 0.224, 0.225))])

    input = torch.unsqueeze(transform(img), dim=0)

    MODEL_PATH = os.path.join("./model_dir/", args.model_path)

    model = PretrainedVGG19()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))

    model.eval()

    classes = ["cat", "dog"]
    output = model(input)

    pred = predict(output, classes)

    img.show()
    print(pred)


if __name__ == "__main__":

    args = get_parser().parse_args()

    main(args)

