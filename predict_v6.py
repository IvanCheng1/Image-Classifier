
# =========================================================
#                        Imports
# =========================================================

import torch
import numpy as np
import argparse
import json
import torch.nn.functional as F
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image

# =========================================================
#                        Argparses
# =========================================================

parser = argparse.ArgumentParser(description='Image Prediction')

parser.add_argument('image', type=str,
                    help='Directory of the image')

parser.add_argument('input', type=str,
                    help='Directory of the checkpoint')

parser.add_argument('--top_k', type=int,
                    help='Number of top cases to return. Default = 1')

parser.add_argument('--category_names', type=str,
                    help='Categories to names')

parser.add_argument('--gpu', type=str,
                    help='Input GPU or CPU. Default = GPU if available')

args = parser.parse_args()

# =========================================================
#                   Load checkpoint
# =========================================================

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)

    arch = checkpoint['architecture']
    exec(f"load_model = models.{arch}(pretrained=True)", globals())
    for param in load_model.parameters():
        param.requires_grad = False

    load_model.classifier = checkpoint['classifier']
    load_model.class_to_idx = checkpoint['class_to_idx']
    load_model.load_state_dict(checkpoint['state_dict'])
    return load_model

load_model = load_checkpoint(args.input)

# =========================================================
#                       Process image
# =========================================================

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    pil_image = Image.open(image)

    pil_image = pil_image.resize((224,224))

    np_image = np.array(pil_image)

    np_image = np_image / 255

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    np_image = (np_image - mean) / std

    np_image = np_image.transpose((2,0,1))

    return np_image

# =========================================================
#                       Prediction
# =========================================================

def predict(image_path, load_model, topk, gpu):
    ''' Predict the class (or classes) of an image using a trained
    deep learning model.
    '''
    image = process_image(image_path) # output is numpy array

    if gpu == None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif gpu.lower() == "gpu":
        device = "cuda"
         # convert to Tensor
        image = torch.from_numpy(image).type(torch.cuda.FloatTensor).to(device)
    else:
        device = "cpu"
         # convert to Tensor
        image = torch.from_numpy(image).type(torch.Tensor).to(device)

    load_model.to(device)
    load_model.eval()

    image = image.unsqueeze(dim=0)

    with torch.no_grad():
        logps = load_model(image)
    ps = torch.exp(logps)

    if topk == None:
        topk = 1
    top_ps, top_class = ps.topk(topk)

    # to numpy array
    top_ps = top_ps.cpu().numpy()
    top_class = top_class.cpu().numpy()

    # to list
    top_ps = top_ps.tolist()[0]
    top_class = top_class.tolist()[0]

    # invert dictionary
    dict = {
        val: key for key, val in load_model.class_to_idx.items()
    }

    classes = [dict[item] for item in top_class]

    return top_ps, classes

# =========================================================
#                       Categories
# =========================================================

with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)

# =========================================================
#                       Get Data
# =========================================================

img = process_image(args.image)

probs, classes = predict(args.image, load_model, args.top_k, args.gpu)

# =========================================================
#                       Print
# =========================================================

flower_classes = []
for flower in classes:
    flower_classes.append(cat_to_name[flower])

for flower, probability, idx in zip(flower_classes, probs, range(len(probs))):
    print(f"Top {idx+1} flower: {flower.title():>13} | Probability: "
          f"{probability*100:0.1f}%")
