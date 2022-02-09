# Dogs-vs-cats classification : training and serving.

We are aiming to implement with Pytorch Dog vs Cat classifier and provide commands to launch server using Docker for deployment.

## Repository Structure

[models/](./models) contains all necessary files for defining models, saved models, and a handler file used for serving.

[parsers/](./parsers) contains parsers for both training and prediction Python scripts.

[deployment/](./deployment) contains necessary files for model serving using Torchserve or/and Docker.


## Install required packages

All packages neeeded for running following commands are listed in `requirements.txt`

```bash
pip install requirements.txt
```

## Dataset

In this project, I am using Dogs-vs-Cats Kaggle dataset available at this [link](https://www.kaggle.com/c/dogs-vs-cats/data).

Training data is composed of 25000 images of various sizes, 12500 are dog images, 12500 are cat images. Training directory is located at [data/train/](./data/train)

Testing data is composed of, and will be used for server inferences. Testing directory is located at [data/test1/](./data/test1)

Examples of training images :

![Example of training dog image](./data/train/dog.4.jpg "Example of training dog image") ![Example of training cat image](./data/train/cat.8.jpg "Example of training cat image")

## Train

We finetuned a VGG19 classifier, pretrained on ImageNet dataset of more than 14 million image for classification (20k classes) and adapted for binary classification  ([Pytorch link](https://pytorch.org/vision/main/generated/torchvision.models.vgg19_bn.html))

After 3 epochs, this model conveerge and reaches an accuracy of around 94% on validation set (10% split).

To run same training procedure that generated saved model, one should run following command :

```bash
python train_main.py --conv_net "vgg" --epochs 3 --validation_split 0.1 --lr 1e-3 --batch_size 32 \
                     --model_path models/model_vgg.pth
```
Train flags stands for:

``--conv_net ``: name of convolutional net to train (`"base_cnn"` </span> for 4 layers "basic" CNN, `"vgg"` for Pretrained VGG19). \
``--epochs``: number of epochs of training procedure.\
``--validation_split``: the validation split rate (0. <= v < 1.) .\
``--lr``: the learning rate.\
``--batch_size``: the batch size used for training.\
``--model_path``: the mo.


During training, at each epoch, **if validation loss decreases**, model state dictionnary is saved at chosen path (parameter ``model_path``)

## Server

To use **Torchserve**, one has to build some key files:

[`handler.py`](models/handler.py) : defines class for data preprocessing, inference, and post-processing. In our case (image classification), this class inherits from standard `ImageClassifier` define in this [torchserve file](https://github.com/pytorch/serve/blob/master/ts/torch_handler/image_classifier.py).

[`config.properties`](deployment/config.properties) : store configurations of port adresses for inference, management and metrics. By default, the inference API is listening on localhost port 8080. The management API is listening on port 8081. Both expect HTTP requests.

[`index_to_name.json`](models/index_to_name.json) : dictionnary to transcript label index to true expected label names.

[`model-store/`](deployment/model-store/) : directory where MAR "ready to serve" model files are built.

### Using Torchserve

Archiver command : 

```bash
torch-model-archiver --model-name vgg  --version 1.0 --model-file models/base_cnn/base_cnn.py \
--serialized-file models/base_cnn/model_base_cnn.pth  --extra-files models/index_to_name.json \
--handler models/handler.py --export-path deployment/model-store -f
```
So torch-model-archiver's used flags stand for:

``--model-name ``: name that the generated MAR "ready to serve" file will have. \
``--version ``: it's optional even though it's a nice practice to include the version of the models so as to keep a proper tracking over them.\
``--model-file``: file where the model architecture is defined.\
``--serialized-file``: the dumped state_dict of the trained model weights.\
``--handler``: the Python file which defines the data preprocessing, inference and postprocessing.\
``--extra-files``: as this is a classification problem you can include the dictionary/json containing the relationships between the IDs (model's target) and the labels/names and/or also additional files required by the model-file to format the output data in a cleaner way.\
``--export-path`` : the export path where the MAR file is created.\
``--f`` : forces the export if MAR file alrealdy exists.


Server command : 

```bash
torchserve --start --ncs --ts-config deployment/config.properties --model-store deployment/model-store --models vgg=vgg.mar
```

### Using Docker

Build ubuntu-torchserve image, using files defined in [deployment/](./deployment)

```bash
docker build -t ubuntu-torchserve:latest deployment/
```
Definition of latest

Launch server

```bash
docker run --rm --name torchserve_docker \
           -p8080:8080 -p8081:8081 -p8082:8082 \
           ubuntu-torchserve:latest \
           torchserve --model-store /home/model-server/model-store/ --models vgg=vgg.mar
```

### Inference

For example to infer the label of the following image, stored in [/data/test1/2124.jpg](/data/test1/2124.jpg), run the following command : 

![Example inference image](./data/test1/2124.jpg)

```bash
curl -X POST http://localhost:8080/predictions/vgg -T dogs-vs-cats/data/test1/2124.jpg
```

This should output the following answer :
 
 `
{
  "dog": 0.9533627033233643
}
`
where label and associated probability is written in a JSON file.

## Credits

Thanks to this great Git repository [serving-pytorch-models](https://github.com/alvarobartt/serving-pytorch-models) by Álvaro Bartolomé.
