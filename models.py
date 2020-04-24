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

    def __init__(self, args):
        super(MultilayerPerceptron, self).__init__()
        self.args = args

        if not args.use_weight_sharing:
            self.fc1 = nn.Linear(392, args.hidden_unit)
        else:
            self.fc1 = nn.Linear(196, args.hidden_unit)

        if not args.use_dropout:
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
            self.fc2 = nn.Linear(args.hidden_unit*2, 2)

        if args.use_auxiliary_losses:
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

    def __init__(self, args):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.args = args

        if not args.use_weight_sharing:
            self.conv1 = nn.Conv2d(2, 16, kernel_size=3)
            self.conv2 = nn.Conv2d(16, 64, kernel_size=3)
        else:
            self.conv1 = nn.Conv2d(1, 8, kernel_size=3)
            self.conv2 = nn.Conv2d(8, 32, kernel_size=3)

        self.fc1 = nn.Linear(256, args.hidden_unit)
        self.fc2 = nn.Linear(args.hidden_unit, 2)

        if args.use_dropout:
            self.dropout = nn.Dropout(args.dropout_rate)

        if args.use_auxiliary_losses:
            self.auxiliary1 = nn.Linear(128, args.hidden_unit)
            self.auxiliary2 = nn.Linear(args.hidden_unit, 10)

    def forward(self, x):

        if not self.args.use_weight_sharing:
            y = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
            y = F.relu(F.max_pool2d(self.conv2(y), kernel_size=2, stride=2))
        else:
            x0, x1 = x[:, 0, :, :].unsqueeze(1), x[:, 1, :, :].unsqueeze(1)
            y0, y1 = F.relu(F.max_pool2d(self.conv1(x0), kernel_size=2, stride=2)),\
                     F.relu(F.max_pool2d(self.conv1(x1), kernel_size=2, stride=2))
            y0, y1 = F.relu(F.max_pool2d(self.conv2(y0), kernel_size=2, stride=2)),\
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