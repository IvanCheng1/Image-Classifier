
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

# =========================================================
#                        Argparses
# =========================================================

parser = argparse.ArgumentParser(description='Image Classifier')

parser.add_argument('data_dir', type=str,
                    help='Directory of the images')

parser.add_argument('--save_dir', type=str,
                    help='Directory to save the training data')

parser.add_argument('--arch', type=str,
                    help='resnet / alexnet / densenet / vgg. Default = vgg16')

parser.add_argument('--learning_rate', type=float,
                    help='Learning rate. Default = 0.001')

parser.add_argument('--hidden_units', type=int,
                    help='Number of hidden units. Default = ')

parser.add_argument('--epochs', type=int,
                    help='Number of epochs. Default = 3')

parser.add_argument('--gpu', type=str,
                    help='Input GPU or CPU. Default = GPU if available')

args = parser.parse_args()

# =========================================================
#                        Sorting Data
# =========================================================

data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                   transforms.RandomResizedCrop(224),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225])
])

valid_transforms = transforms.Compose([transforms.Resize(255),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([transforms.Resize(255),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406],
                                                       [0.229, 0.224, 0.225])
])

image_train_datasets = datasets.ImageFolder(train_dir,
                                            transform=train_transforms)
image_valid_datasets = datasets.ImageFolder(valid_dir,
                                            transform=valid_transforms)
image_test_datasets = datasets.ImageFolder(test_dir,
                                           transform=test_transforms)

trainloader = torch.utils.data.DataLoader(image_train_datasets, batch_size=32,
                                          shuffle=True)
validloader = torch.utils.data.DataLoader(image_valid_datasets, batch_size=32,
                                          shuffle=False)
testloader = torch.utils.data.DataLoader(image_test_datasets, batch_size=32,
                                         shuffle=False)

# =========================================================
#                       Architecture
# =========================================================

if args.arch:
    arch = args.arch.lower()
    print(f"--- Using {arch} architecture ---")
    exec(f"model = models.{arch}(pretrained=True)")
else:
    arch = 'vgg16'
    print(f"--- Using {arch} architecture ---")
    model = models.vgg16(pretrained=True)
# Turn off gradients for our model
for param in model.parameters():
    param.requires_grad = False

# =========================================================
#                       Classifier
# =========================================================

# finding out the different in_features for different
# architectures to support different types of models
if "resnet" in arch:
    classifier_input = model.fc.in_features
elif "alexnet" in arch:
    classifier_input = model.classifier[1].in_features
elif "vgg" in arch:
    classifier_input = model.classifier[0].in_features
elif "densenet" in arch:
    classifier_input = model.classifier.in_features

# changing the hidden units if given, and the number of input
# according to the neural outputs of the architecture model
if args.hidden_units:
    hidden_units = args.hidden_units
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(classifier_input, hidden_units)),
        ('relu1', nn.ReLU()),
        ('drop1', nn.Dropout(p=0.2)),
        ('fc2', nn.Linear(hidden_units, 500)),
        ('relu2', nn.ReLU()),
        ('drop2', nn.Dropout(p=0.2)),
        ('fc3', nn.Linear(500, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
else:
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(classifier_input, 2500)),
        ('relu1', nn.ReLU()),
        ('drop1', nn.Dropout(p=0.2)),
        ('fc2', nn.Linear(2500, 500)),
        ('relu2', nn.ReLU()),
        ('drop2', nn.Dropout(p=0.2)),
        ('fc3', nn.Linear(500, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

model.classifier = classifier

# =========================================================
#                       GPU
# =========================================================

if args.gpu == None: # no gpu option given, use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if str(device) == "cuda":
        print(f"--- Using GPU for training ---")
    else:
        print(f"--- Using CPU for training ---")
elif args.gpu.lower() == 'gpu':
    device = 'cuda'
    print(f"--- Using GPU for training ---")
elif args.gpu.lower() == 'cpu':
    device = 'cpu'
    print(f"--- Using CPU for training ---")

model.to(device)

# =========================================================
#                       Optimizer
# =========================================================

criterion = nn.NLLLoss()

if args.learning_rate: # learning rate given
    learning_rate = args.learning_rate
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
else:
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

# =========================================================
#                       Epochs
# =========================================================

if args.epochs: # number of epoch given
    epochs = args.epochs
else:
    epochs = 3

steps = 0
running_loss = 0
print_every = 50

# =========================================================
#                       Training
# =========================================================

for epoch in range(epochs):
    for images, labels in trainloader:
        steps += 1

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        logps = model(images)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if steps % print_every == 0:
            model.eval()
            valid_loss = 0
            valid_accuracy = 0

            for images, labels in validloader:

                images, labels = images.to(device), labels.to(device)

                logps = model(images)
                loss = criterion(logps, labels)
                valid_loss += loss.item()

                ps = torch.exp(logps)
                top_ps, top_class = ps.topk(1, dim=1)
                equality = top_class == labels.view(*top_class.shape)
                valid_accuracy += torch.mean(equality.type(torch.FloatTensor)).item()

            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                  f"Validation accuracy: {valid_accuracy/len(validloader):.3f}")

            running_loss = 0
            model.train()

print("--- Training completed ---")

# =========================================================
#                       Checkpoint
# =========================================================

checkpoint = {'classifier': model.classifier,
              'optimizer': optimizer.state_dict(),
              'class_to_idx': image_train_datasets.class_to_idx,
              'architecture': arch,
              'state_dict': model.state_dict(),
              'num_epochs': epochs}

if args.save_dir:
    save_dir = args.save_dir
    torch.save(checkpoint, save_dir + '/checkpoint.pth')
    print(f"--- Checkpoint saved to {save_dir + '/checkpoint.pth'} ---")
else:
    torch.save(checkpoint, 'checkpoint.pth')
    print(f"--- Checkpoint saved to {'checkpoint.pth'} ---")
