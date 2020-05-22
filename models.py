#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : DeepLearningProject1
# @Author       : Xiaoyu LIN
# @File         : models.py
# @Description  :

import torch
from torch import nn
from torch.nn import functional as F


class MultilayerPerceptron(nn.Module):
    """
    Build a MLP model.
    """
    def __init__(self, args):
        super(MultilayerPerceptron, self).__init__()
        self.args = args

        if not args.use_weight_sharing:
            # Whether weight sharing is used.
            self.fc1 = nn.Linear(392, args.hidden_unit)
        else:
            self.fc1 = nn.Linear(196, args.hidden_unit)

        if not args.use_dropout:
            # Whether dropput is used
            self.perceptron_blocks = nn.Sequential(
                *(nn.Sequential(nn.Linear(args.hidden_unit, args.hidden_unit), nn.ReLU())
                  for _ in range(args.block_num))
            )
        else:
            self.perceptron_blocks = nn.Sequential(
                *(nn.Sequential(nn.Linear(args.hidden_unit, args.hidden_unit), nn.ReLU(), nn.Dropout(args.dropout_rate))
                  for _ in range(args.block_num))
            )
            self.dropout = nn.Dropout(args.dropout_rate)

        if not args.use_weight_sharing:
            self.fc2 = nn.Linear(args.hidden_unit, 2)
        else:
            self.fc2 = nn.Linear(args.hidden_unit * 2, 2)

        if args.use_auxiliary_losses:
            # Whether auxiliary loss is used
            self.auxiliary1 = nn.Linear(args.hidden_unit, args.hidden_unit)
            self.auxiliary2 = nn.Linear(args.hidden_unit, 10)

    def forward(self, x):
        if not self.args.use_weight_sharing:
            y = x.view(x.size(0), -1)
            y = F.relu(self.fc1(y))
            if self.args.use_dropout:
                y = self.dropout(y)
            y = self.perceptron_blocks(y)
            y = self.fc2(y)
            return y
        else:
            x0, x1 = x[:, 0, :, :], x[:, 1, :, :]
            y0, y1 = x0.view(x0.size(0), -1), x1.view(x1.size(0), -1)
            y0, y1 = F.relu(self.fc1(y0)), F.relu(self.fc1(y1))
            if self.args.use_dropout:
                y0, y1 = self.dropout(y0), self.dropout(y1)
            y0, y1 = self.perceptron_blocks(y0), self.perceptron_blocks(y1)
            y = torch.cat((y0, y1), 1)
            y = y.view(y.size(0), -1)
            if self.args.use_auxiliary_losses:
                y0, y1 = F.relu(self.auxiliary1(y0)), F.relu(self.auxiliary1(y1))
                if self.args.use_dropout:
                    y0, y1 = self.dropout(y0), self.dropout(y1)
                y0, y1 = self.auxiliary2(y0), self.auxiliary2(y1)
        y = self.fc2(y)

        if not self.args.use_auxiliary_losses:
            return y
        else:
            return y, y0, y1


class ConvolutionalNeuralNetwork(nn.Module):
    """
    Build a CNN model
    """
    def __init__(self, args):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.args = args

        if not args.use_weight_sharing:
            # Whether weight sharing is used
            self.conv1 = nn.Conv2d(2, 16, kernel_size=3)
            self.conv2 = nn.Conv2d(16, 64, kernel_size=3)
        else:
            self.conv1 = nn.Conv2d(1, 8, kernel_size=3)
            self.conv2 = nn.Conv2d(8, 32, kernel_size=3)

        self.fc1 = nn.Linear(256, args.hidden_unit)
        self.fc2 = nn.Linear(args.hidden_unit, 2)

        if args.use_dropout:
            # Whether dropput is used
            self.dropout = nn.Dropout(args.dropout_rate)

        if args.use_auxiliary_losses:
            # Whether auxiliary loss is used
            self.auxiliary1 = nn.Linear(128, args.hidden_unit)
            self.auxiliary2 = nn.Linear(args.hidden_unit, 10)

    def forward(self, x):

        if not self.args.use_weight_sharing:
            y = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
            y = F.relu(F.max_pool2d(self.conv2(y), kernel_size=2, stride=2))
        else:
            x0, x1 = x[:, 0, :, :].unsqueeze(1), x[:, 1, :, :].unsqueeze(1)
            y0, y1 = F.relu(F.max_pool2d(self.conv1(x0), kernel_size=2, stride=2)), \
                     F.relu(F.max_pool2d(self.conv1(x1), kernel_size=2, stride=2))
            y0, y1 = F.relu(F.max_pool2d(self.conv2(y0), kernel_size=2, stride=2)), \
                     F.relu(F.max_pool2d(self.conv2(y1), kernel_size=2, stride=2))
            y = torch.cat((y0, y1), 1)
            if self.args.use_auxiliary_losses:
                y0, y1 = y0.view(y0.size(0), -1), y1.view(y1.size(0), -1)
                y0, y1 = F.relu(self.auxiliary1(y0)), F.relu(self.auxiliary1(y1))
                if self.args.use_dropout:
                    y0, y1 = self.dropout(y0), self.dropout(y1)
                y0, y1 = self.auxiliary2(y0), self.auxiliary2(y1)

        y = y.view(y.size(0), -1)
        y = F.relu(self.fc1(y))
        if self.args.use_dropout:
            y = self.dropout(y)
        y = self.fc2(y)

        if not self.args.use_auxiliary_losses:
            return y
        else:
            return y, y0, y1


class ResNet(nn.Module):
    """
    Build a ResNet model
    """
    def __init__(self, args):
        super(ResNet, self).__init__()
        self.args = args

        if not args.use_weight_sharing:
            # Whether weight sharing is used
            self.channel_num = args.channel_num
            self.conv = nn.Conv2d(2, self.channel_num, kernel_size=args.kernel_size)
        else:
            self.channel_num = int(args.channel_num/2)
            self.conv = nn.Conv2d(1, self.channel_num, kernel_size=args.kernel_size)
        self.bn = nn.BatchNorm2d(self.channel_num)

        self.resnet_blocks = nn.Sequential(
            *(ResNetBlock(self.channel_num, args.kernel_size, args.skip_connections, args.batch_normalization)
              for _ in range(args.block_num))
        )

        self.fc1 = nn.Linear(512, args.hidden_unit)
        self.fc2 = nn.Linear(args.hidden_unit, 2)

        if args.use_dropout:
            # Whether dropput is used
            self.dropout = nn.Dropout(args.dropout_rate)

        if args.use_auxiliary_losses:
            # Whether auxiliary loss is used
            self.auxiliary1 = nn.Linear(256, args.hidden_unit)
            self.auxiliary2 = nn.Linear(args.hidden_unit, 10)

    def forward(self, x):

        if not self.args.use_weight_sharing:
            y = F.relu(self.bn(self.conv(x)))
            y = self.resnet_blocks(y)
            y = F.avg_pool2d(y, kernel_size=3, stride=3)
        else:
            x0, x1 = x[:, 0, :, :].unsqueeze(1), x[:, 1, :, :].unsqueeze(1)
            y0, y1 = F.relu(self.bn(self.conv(x0))),  F.relu(self.bn(self.conv(x1)))
            y0, y1 = self.resnet_blocks(y0), self.resnet_blocks(y1)
            y0, y1 = F.avg_pool2d(y0, kernel_size=3, stride=3), F.avg_pool2d(y1, kernel_size=3, stride=3)
            y = torch.cat((y0, y1), 1)
            if self.args.use_auxiliary_losses:
                y0, y1 = y0.view(y0.size(0), -1), y1.view(y1.size(0), -1)
                y0, y1 = F.relu(self.auxiliary1(y0)), F.relu(self.auxiliary1(y1))
                if self.args.use_dropout:
                    y0, y1 = self.dropout(y0), self.dropout(y1)
                y0, y1 = self.auxiliary2(y0), self.auxiliary2(y1)

        y = y.view(y.size(0), -1)
        y = F.relu(self.fc1(y))
        if self.args.use_dropout:
            y = self.dropout(y)
        y = self.fc2(y)

        if not self.args.use_auxiliary_losses:
            return y
        else:
            return y, y0, y1


class ResNetBlock(nn.Module):
    """
    Construct a ResNet block
    """

    def __init__(self, nb_channels, kernel_size, skip_connections=True, batch_normalization=True):
        super(ResNetBlock, self).__init__()

        self.conv1 = nn.Conv2d(nb_channels, nb_channels,
                               kernel_size=kernel_size,
                               padding=(kernel_size - 1) // 2)

        self.bn1 = nn.BatchNorm2d(nb_channels)

        self.conv2 = nn.Conv2d(nb_channels, nb_channels,
                               kernel_size=kernel_size,
                               padding=(kernel_size - 1) // 2)

        self.bn2 = nn.BatchNorm2d(nb_channels)

        self.skip_connections = skip_connections
        self.batch_normalization = batch_normalization

    def forward(self, x):
        y = self.conv1(x)
        if self.batch_normalization:
            y = self.bn1(y)
        y = F.relu(y)
        y = self.conv2(y)
        if self.batch_normalization:
            y = self.bn2(y)
        if self.skip_connections:
            y = y + x
        y = F.relu(y)

        return y
