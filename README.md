# AI Programming with Python Project

Final project Udacity's AI Programming with Python Nanodegree program. First, to develop code for an image classifier built with PyTorch, then convert it into a command line application.

## train.py

Train a new network on a data set.

- Basic usage: ```python train.py data_directory```
- Prints out training loss, validation loss, and validation accuracy as the network trains
- Options:
    1. Set directory to save checkpoints: ```python train.py data_dir --save_dir save_directory```
    2. Choose architecture, default is vgg16; supports resnet, alexnet, densenet, and vgg: ```python train.py data_dir --arch "vgg13"```
    3. Set hyperparameters:
        - Learning rate: ```python train.py data_dir --learning_rate 0.01```
        - Number of hidden units: ```python train.py data_dir  --hidden_units 512```
        - Number of epochs: ```python train.py data_dir --epochs 20```
    4. Use GPU for training: ```python train.py data_dir --gpu```

## predict.py

Predict flower name from an image.

- Basic usage: ```python predict.py /path/to/image checkpoint```
- Options:
    1. Directory of the image: ```python predict.py /path/to/image```
    2. Directory of the checkpoint: ```python predict.py /path/to/checkpoint```
    3. Return top KK most likely classes: ```python predict.py input checkpoint --top_k 3```
    4. Use a mapping of categories to real names: ```python predict.py input checkpoint --category_names cat_to_name.json```
    5. Use GPU for inference: ```python predict.py input checkpoint --gpu```
