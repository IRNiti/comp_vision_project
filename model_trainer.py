'''
Vasundhara Gupta
Raluca Niti

Referenced from http://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
'''

import copy
import os
import time

# import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms

ROOT_TRAINING_DIR = 'segregated_train'

def dataset_from_dir(root_dir):
    data_transform = transforms.Compose([
        transforms.RandomSizedCrop(224),  # needs to be 224 pixels at minimum,
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # analogous to numpy ndarray
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return datasets.ImageFolder(root_dir, data_transform)


def loader_from_dataset(dataset):
    return torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

def train_model(model, training_dataset_loader, criterion, optim_scheduler, num_epochs=25):
    since = time.time()

    best_model = model
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)


        optimizer = optim_scheduler(model, epoch)

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for data in training_dataset_loader:
            print('iterating')
            # get the inputs
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.data[0]
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dset_sizes[phase]
        epoch_acc = running_corrects / dset_sizes[phase]

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            phase, epoch_loss, epoch_acc))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    return best_model

def optim_scheduler_ft(model, epoch, init_lr=0.001, lr_decay_epoch=7):
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    return optimizer

if __name__ == '__main__':
    dataset = dataset_from_dir(ROOT_TRAINING_DIR)
    dataset_loader = loader_from_dataset(dataset)

    inputs, classes = next(iter(dataset_loader))

    model = models.resnet18(pretrained=True)  # pretrained on imagenet
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    criterion = nn.CrossEntropyLoss()

    model = train_model(model, dataset_loader, criterion, optim_scheduler_ft, num_epochs=5)

