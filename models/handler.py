from ts.torch_handler.image_classifier import ImageClassifier
from torchvision.transforms import Compose, ToTensor, Resize, Normalize


class DogsVsCatsClassifier(ImageClassifier):
    """
        Classifier handler class, outputs only most probable label.
        Use standard train transformation before passing to model.
    """
    topk = 1
    image_processing = Compose([Resize((224, 224)),
                                ToTensor(),
                                Normalize((0.485, 0.456, 0.406),
                                          (0.229, 0.224, 0.225))])
