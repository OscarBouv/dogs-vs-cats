from ts.torch_handler.image_classifier import ImageClassifier


class PretrainedVGG19(ImageClassifier):
    """
        Classifier handler class, outputs only most probable label.
    """
    topk = 1
