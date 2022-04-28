#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 08:55:07 2022

@author: shilpadeshpande
"""

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch.nn as nn
from torch import optim
import numpy as np
from torchvision import transforms
import cv2
import torch.nn.functional as F

def load_mnist():
    data_full = datasets.MNIST(
        root='data',
        train='true',
        transform=ToTensor(),
        download=True,
    )
    
   
    
    # Split dataset to train,test and validation

    # train, test and validation seperation
    train_data, test_data, valid_data = torch.utils.data.random_split(data_full,
                                                                      [30000, 20000, 10000])
    # how to visualize the data
    plt.imshow(data_full.data[0])
    plt.imshow(data_full.data[0], cmap='gray')
    plt.imshow(data_full.data[44], cmap='gray')
    # access the data but you need export it first
    data1 = data_full.data[20]
    plt.imshow(data1, cmap='gray')
    # how to visualize multiple images
    figure = plt.figure(figsize=(10, 10))
    cols, row = 5, 5
    for i in range(1, cols * row + 1):
        idx = torch.randint(len(data_full), size=(1,)).item()

        img, label = data_full[idx]

        figure.add_subplot(row, cols, i)
        plt.title('Number: ' + str(label))
        plt.axis('off')
        plt.imshow(img.squeeze(), cmap='gray')
    plt.show()
    # let's put into a dict
    batch_size = 100
    loaders = {'train': torch.utils.data.DataLoader(train_data,
                                                    batch_size=batch_size,
                                                    shuffle=True),

               'test': torch.utils.data.DataLoader(test_data,
                                                   batch_size=batch_size,
                                                   shuffle=True),
               'valid': torch.utils.data.DataLoader(valid_data,
                                                    batch_size=batch_size,
                                                    shuffle=True),

               }
    
    # visualize the dict
    train_part = loaders.get('train')
    data2 = train_part.dataset
    element1 = data2[0][0].squeeze()
    plt.imshow(element1, cmap='gray')
    return train_part,loaders
    
    
# Convolution block with 3 X 3 kernel
def conv_block(in_channels, out_channels, pool=True, pool_no=2, padding=1):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU()
              ]
    if pool: layers.append(nn.MaxPool2d(pool_no))
    return nn.Sequential(*layers)


class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ResNet9, self).__init__()
        self.conv1 = conv_block(in_channels, 32, pool=True, pool_no=2)

        self.conv2 = conv_block(32, 64, pool=True, pool_no=2)
        self.res1 = nn.Sequential(conv_block(64, 64), conv_block(64, 64))

        self.conv3 = conv_block(64, 128, pool=True, pool_no=1)
        self.conv4 = conv_block(128, 256, pool=True, pool_no=1, padding=1)

        self.res2 = nn.Sequential(conv_block(256, 256), conv_block(256, 256))

        self.MP = nn.MaxPool2d(2)
        self.FlatFeats = nn.Flatten()
        self.classifier = nn.Linear(2304, num_classes)

    def forward(self, xb):
        # print("1111",xb.size())
        out = self.conv1(xb)

        out = self.conv2(out)
        """print("2222",out.size())"""
        out = self.res1(out) + out
        """print("3333",out.size())"""
        out = self.conv3(out)
        """ print("444",out.size())"""
        out = self.conv4(out)
        """print("555",out.size())"""

        """ new_data = F.pad(input=out, pad=(1, 1, 1, 1), mode='constant', value=0)"""

        """print("5151",out.size())"""
        out = self.res2(out) + out
        """print("666",out.size())"""
        out = self.MP(out)  # classifier(out_emb)
        out_emb = self.FlatFeats(out)
        out = self.classifier(out_emb)
        return out
    
    
def resnet9():
    return ResNet9(1,10)


def model_train_test_validate(model,num_epochs,train_part,loaders,device,optimizer,criterion,loss_list,loss_list_mean):
    iter = 0
    for epoch in range(num_epochs):

        print('Epoch: {}'.format(epoch))

        loss_buff = []

        for i, (images, labels) in enumerate(loaders['train']):

            # getting the images and labels from the training dataset
            images = images.requires_grad_().to(device)
            labels = labels.to(device)

            # clear the gradients w.r.t parameters
            optimizer.zero_grad()

            # now we can call the CNN and operate over the images
            outputs = model(images)

            # loss calculation
            loss = criterion(outputs, labels)
            loss_buff = np.append(loss_buff, loss.item())

            # backward for getting the gradients w.r.t. parameters
            loss.backward()

            loss_list = np.append(loss_list, (loss_buff))

            # update the parameters
            optimizer.step()

            iter += 1

            if iter % 10 == 0:
                """print('Iterations: {}'.format(iter))"""

           # Validation
            if iter % 100 == 0:

                # Accuracy
                correct = 0
                total = 0

                for images, labels in loaders['valid']:
                    images = images.requires_grad_().to(device)
                    labels = labels.to(device)

                    outputs = model(images)

                    # get the predictions from the maximum value
                    _, predicted = torch.max(outputs.data, 1)

                    # how many labels I have, also mean the size of the valid
                    total += labels.size(0)

                    correct += (predicted == labels).sum()

                accuracy = 100 * correct / total

                print('Iterations: {}. Loss: {}. Validation Accuracy: {}'.format(iter,
                                                                                 loss.item(), accuracy))
            loss_list_mean = np.append(loss_list_mean, (loss.item()))

    # visualize the loss
    plt.plot(loss_list)
    plt.plot(loss_list_mean)
    return model

    #Testing
    if iter % 100 == 0:

        # Accuracy
        correct = 0
        total = 0

        for images, labels in loaders['test']:
            images = images.requires_grad_().to(device)
            labels = labels.to(device)

            outputs = model(images)

            # get the predictions from the maximum value
            _, predicted = torch.max(outputs.data, 1)

            # how many labels I have, also mean the size of the valid
            total += labels.size(0)

            correct += (predicted == labels).sum()

        accuracy = 100 * correct / total

        print('Test Accuracy: {}'.format(accuracy))

    # saving the model
    torch.save(model.state_dict(), '/Users/shilpadeshpande/Downloads/resnet.pth')

    
    
    
def model_setup():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet9(1, 10)
    model.to(device)
    # define the loss
    criterion = nn.CrossEntropyLoss()
    # we also need to define an optimizer
    learning_rate = 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # let's define how many epoch we want
    num_epochs = 3
    # initialize some loss lists
    loss_list = []
    loss_list_mean = []
    return(device,model,criterion,learning_rate,optimizer,num_epochs,loss_list,loss_list_mean)


def preprocess_custom_image(img_path):
    # update this image path
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Cropping the image from center
    width = img.shape[1]
    height = img.shape[0]
    print(width)
    print(height)

    left = int(np.ceil((width - 280) / 2))
    right = width - int(np.floor((width - 180) / 2))

    top = int(np.ceil((height - 800) / 2))
    bottom = height - int(np.floor((height - 300) / 2))

    center_cropped_img = img[top:bottom, left:right]
    plt.imshow(center_cropped_img, cmap='gray')

    # resize the image to MNIST size 28 X 28

    # ret,thresh1 = cv2.threshold(center_cropped_img,127,255,cv2.THRESH_BINARY_INV)
    img_resized = cv2.resize(center_cropped_img, (28, 28))
    img_resized = cv2.bitwise_not(img_resized)

    ret, thresh1 = cv2.threshold(img_resized, 127, 255, cv2.INTER_CUBIC)
    # img = img.resize((28, 28), Image.ANTIALIAS)

    print(img_resized)

    plt.imshow(thresh1, cmap='gray')
    # img_resized = cv2.bitwise_not(img_resized)
    print(img_resized)

    plt.imshow(img_resized, cmap='gray')

    # converting the image to  tensor
    """ImgTensor=transform_img(image_enhanced)"""
    transform_img = transforms.ToTensor()
    ImgTensor = transform_img(img_resized)
    ImgTensor = ImgTensor.unsqueeze(0)
    print("Loaded image's tensor size", ImgTensor.size())
    return ImgTensor

def prediction_custom_image(ImgTensor,model):
    # Load custom image in to the model and predict the class
    output_img = model(ImgTensor)
    print(output_img)
    index = output_img.data.cpu().numpy().argmax()
    """pred = classes[index]"""
    print("predicted", index)
    ################################
     
    return index
    
    
    
    
    