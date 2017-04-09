'''
Vasundhara Gupta
Raluca Niti

Referenced from http://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
'''

import copy
import os
import time

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms

ROOT_TRAINING_DIR = 'segregated_train'
ROOT_EVAL_DIR = 'segregated_eval'

ROOT_TEST_DIR = 'X_Test'

DEFAULT_NUM_EPOCHS = 5


def training_data_transform():
    '''Data augmentation and normalization'''
    return transforms.Compose([
        transforms.RandomSizedCrop(224),  # needs to be 224 pixels at minimum,
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # analogous to numpy ndarray
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def eval_data_transform():
    '''Just normalization with no augmentation'''
    return transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),  # needs to be 224 pixels at minimum
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def training_dataset_from_dir(root_dir):
    return datasets.ImageFolder(root_dir, training_data_transform())


def eval_dataset_from_dir(root_dir):
    return datasets.ImageFolder(root_dir, eval_data_transform())


def loader_from_dataset(dataset):
    return torch.utils.data.DataLoader(dataset,
                                       batch_size=4,  # 4
                                       shuffle=True,  # True
                                       num_workers=4)  # 4


def train_model(model, phase_to_dataset_loader, phase_to_dataset_size, criterion, optim_scheduler,
                num_epochs=DEFAULT_NUM_EPOCHS):
    since = time.time()

    best_model = model
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'eval']:
            if phase == 'train':
                optimizer = optim_scheduler(model, epoch)

            running_loss = 0.0
            running_corrects = 0

            dataset = phase_to_dataset_loader[phase].dataset

            # Iterate over data.
            for i, data in enumerate(phase_to_dataset_loader[phase]):
                # get the inputs and wrapin variable
                inputs, labels = data
                
                inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)

                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

            dataset_size = phase_to_dataset_size[phase]
            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects / dataset_size

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'eval' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model)

        print('****')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    return best_model


def optim_scheduler_ft(model, epoch, init_lr=0.001, lr_decay_epoch=7):
    '''Learning rate scheduler'''
    lr = init_lr * (0.1 ** (epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('Learning rate set to {}'.format(lr))

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    return optimizer

def pretrained_resnet_model():
    # 18-layer model
    model = models.resnet18(pretrained=True)  # pretrained on imagenet
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    return model

if __name__ == '__main__':
    phase_to_dataset = {
        'train': training_dataset_from_dir(ROOT_TRAINING_DIR),
        'eval': eval_dataset_from_dir(ROOT_EVAL_DIR)
    }

    phase_to_dataset_size = {x: len(phase_to_dataset[x]) for x in phase_to_dataset
    }

    phase_to_dataset_loader = {
        x: loader_from_dataset(phase_to_dataset[x]) for x in phase_to_dataset
    }

    model = models.resnet18(pretrained=True)  # pretrained on imagenet
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    model = train_model(
        model=model,
        phase_to_dataset_loader=phase_to_dataset_loader,
        phase_to_dataset_size=phase_to_dataset_size,
        criterion=nn.CrossEntropyLoss(),
        optim_scheduler=optim_scheduler_ft,
        num_epochs=DEFAULT_NUM_EPOCHS)

    # test_dataset = eval_dataset_from_dir(ROOT_TEST_DIR)
    # test_dataset_loader = loader_from_dataset(test_dataset)

    '''
    for i, data in enumerate(phase_to_dataset_loader['eval']):
        print(data)
        break
    print(phase_to_dataset_loader['eval'].dataset.imgs)
    '''
