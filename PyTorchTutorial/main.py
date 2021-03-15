#!/usr/bin/env python
# -*- coding:utf-8 -*-
# AUTHOR: Ti Bai
# EMAIL: tibaiw@gmail.com
# AFFILIATION: MAIA Lab | UT Southwestern Medical Center
# DATETIME: 9/10/2020 12:17 PM

import os
import cv2

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn


class FancyArchitecture(nn.Module):
    def __init__(self, num_classes=3):
        super(FancyArchitecture, self).__init__()
        initial_channels = 16
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=initial_channels, kernel_size=3, padding=1)
        self.pool1 = nn.AvgPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=initial_channels, out_channels=initial_channels * 2, kernel_size=3, padding=1)
        self.pool2 = nn.AvgPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(in_channels=initial_channels * 2, out_channels=initial_channels * 4, kernel_size=3, padding=1)
        self.pool3 = nn.AvgPool2d(kernel_size=2)

        self.conv4 = nn.Conv2d(in_channels=initial_channels * 4, out_channels=initial_channels * 8, kernel_size=3, padding=1)
        self.pool4 = nn.AvgPool2d(kernel_size=2)

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.relu = nn.ReLU()

        self.fc = nn.Linear(initial_channels * 8, num_classes)
        self.feature_dimension = initial_channels * 8

    def forward(self, x):
        x = self.relu(self.pool1(self.conv1(x)))
        x = self.relu(self.pool2(self.conv2(x)))
        x = self.relu(self.pool3(self.conv3(x)))
        x = self.relu(self.pool4(self.conv4(x)))
        x = self.gap(x)
        x = x.view(-1, self.feature_dimension)
        x = self.fc(x)
        return x


class GreatDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, data_list_path):
        super(GreatDataset, self).__init__()
        self.data_list = []
        with open(os.path.join(data_root, data_list_path), 'r') as f:
            for current_line in f.readlines():
                current_path, current_label = current_line.strip().split('\t')
                self.data_list.append((os.path.join(data_root, current_path), int(current_label)))

    def __getitem__(self, item):
        current_image_path, current_label = self.data_list[item]
        current_image = cv2.imread(current_image_path)[:, :, 0]
        current_image = torch.from_numpy(current_image).float().unsqueeze(0)
        return {'image': current_image,
                'label': current_label,
                'path': current_image_path}

    def __len__(self):
        return len(self.data_list)


if __name__ == '__main__':
    # this is a very simple tutorial for PyTorch
    # key steps:
    # 1: define the dataset (You need to implement the __getitem__ and the __len__ methods for customized dataset!)
    # 2: define the model (You need to implement the __forward__ method. This is your computation graph!)
    # 3: define the criterion and the optimizer (This part is very easy, just two lines of codes!)
    # 4: begin train the network
    # 5: In this tutorial, the only purpose is to show you how to use PyTorch.
    #    Therefore, I didn't write any un-necessary codes, such as data augmentation, performance validation,
    #    model save, performance analysis, etc.
    # 6: For your convenience, I provided here a simple sample that can be run even with your CPU.
    # 7: About the sample here:
    #    Supposed we are asked to recognize the modality (CT/MRI/PET) of any give medical image,
    #    I will collect some images for each modality (~600 images in total, ~2MB),
    #    then customized a GreatDataset class as the data provider,
    #    then customized a FancyModel as the classifier,
    #    then define the criterion and the optimizer,
    #    then begin the training.
    # 8: When you learn how to code, try to use debug it line by line, and watch the memory information in each step.
    # 9: After you are familiar with the PyTorch as well as deep learning,
    #    you can use my another opensourced codebase "https://github.com/baiti01/CodeBase",
    #    which might be much more powerful to be used in your real project.

    # params
    data_root = r'dataset'
    train_list = 'train.list'
    val_list = 'val.list'
    batch_size = 8
    lr = 0.001
    epoch_number = 100

    # define env
    is_gpu = True if torch.cuda.is_available() else False
    if is_gpu:
        cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.enabled = True

    # define dataset
    train_dataset = GreatDataset(data_root, train_list)
    val_dataset = GreatDataset(data_root, val_list)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0,
                                               pin_memory=is_gpu)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=0,
                                             pin_memory=is_gpu)

    # define model
    model = FancyArchitecture()
    model = model.cuda() if is_gpu else model

    # define loss
    criterion = torch.nn.CrossEntropyLoss()
    criterion = criterion.cuda() if is_gpu else criterion

    # define optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4, nesterov=True)

    # main loop
    for current_epoch in range(epoch_number):
        for idx, current_data in enumerate(train_loader):
            current_input = current_data['image']
            current_target = current_data['label']
            current_target.requires_grad = False
            if is_gpu:
                current_input = current_input.cuda()
                current_target = current_target.cuda()

            current_output = model(current_input)
            loss = criterion(current_output, current_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            msg = f'Epoch: [{current_epoch}|{epoch_number}]\t'\
                f'Iteration: [{idx}|{len(train_loader)}]\t'\
                f'Loss: {loss.item()}'
            print(msg)

    print('Congrats! May the force be with you ...')
