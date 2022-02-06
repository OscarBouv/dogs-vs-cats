# dogs-vs-cats
Implementation of dogs-vs-cats classifier inspired by Kaggle competition (link)



## Server

Archiver command : 

```bash
torch-model-archiver --model-name vgg --version 1.0 --model-file models/vgg.py --serialized-file models/model_vgg.pth --extra-files models/index_to_name.json --handler models/handler.py --export-path serve/model-store -f
```

Server command : 

```bash
torchserve --start --ncs --ts-config serve/config.properties --model-store serve/model-store --models vgg=vgg.mar
```

Inference command :

```bash
curl -X POST http://localhost:8080/predictions/vgg -T dogs-vs-cats/data/test1/1.jpg
```
