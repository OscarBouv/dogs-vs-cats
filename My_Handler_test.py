import torch
import io
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize


from models import PretrainedVGG19
import os
import json
import numpy as np

class MyHandler():
    """
    Custom handler for pytorch serve. This handler supports batch requests.
    For a deep description of all method check out the doc:
    https://pytorch.org/serve/custom_service.html
    """

    def __init__(self):

        #super().__init__()
        #self.model = PretrainedVGG19()
        #self.model.load_state_dict(torch.load("model_store/model_vgg.pth", map_location="cpu"))
        self.model = torch.load("model_store/model_vgg.pt")
        self.model.eval()

        jsonFile = open("index_to_name.json")
        self.mapping = json.load(jsonFile)
        jsonFile.close()

        self.transform = Compose([Resize((224, 224)),
                                  ToTensor(),
                                  Normalize((0.485, 0.456, 0.406),
                                            (0.229, 0.224, 0.225))])

    def preprocess_one_image(self, req):
        """
        Process one single image.
        """
        # get image from the request
        #image = req.get("data")
        #if image is None:
        #    image = req.get("body")

        # create a stream from the encoded image
        image = Image.open(req)
        image.show()
        image = self.transform(image)
        # add batch dim
        image = image.unsqueeze(0)

        return image

    def preprocess(self, requests):
        """
        Process all the images from the requests and batch them in a Tensor.
        """
        # images = [self.preprocess_one_image(req) for req in requests]
        # images = torch.cat(images)
        return self.preprocess_one_image(requests)

    def inference(self, x):
        """
        Given the data from .preprocess, perform inference using the model.
        We return the predicted label for each image.
        """
        output = self.model(x)
        probs = torch.sigmoid(output)

        return probs

    def postprocess(self, probs):
        """
        Given the data from .inference, postprocess the output.
        In our case, we get the human readable label from the mapping 
        file and return a json. Keep in mind that the reply must always
        be an array since we are returning a batch of responses.
        """
        res = []
        # pres has size [BATCH_SIZE, 1]
        # convert it to list

        preds = (probs > 0.5).to(int)
        preds = preds.cpu().tolist()

        for i in range(len(preds)):
            pred, prob = preds[i][0], float(probs[i][0])

            if pred == 0:
                prob = 1. - prob

            label = self.mapping[str(pred)]
            res.append({'label' : label, 'probability': prob})

        return res


if __name__ == "__main__":

    handler = MyHandler()

    file_name = os.listdir("data/test1/")[np.random.randint(1000)]
    IMG_PATH = os.path.join("data/test1", file_name)

    #data = Image.open(IMG_PATH)

    # if data is None:
    #    pass

    print(handler.mapping)

    data = handler.preprocess(IMG_PATH)
    data = handler.inference(data)
    data = handler.postprocess(data)

    print(data)
