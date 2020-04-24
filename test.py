#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : DeepLearningProject1
# @Author       : Xiaoyu LIN
# @File         : test.py
# @Description  :

import argparse
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import dlc_practical_prologue as prologue
from models import *


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def train_model(model, data_input, data_target, data_classes, args, device, logs):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss().to(device)

    for epoch in range(1, args.num_epochs + 1):
        train_loss = 0
        for batch_input, batch_target, batch_classes in zip(data_input.split(args.batch_size),
                                                            data_target.split(args.batch_size),
                                                            data_classes.split(args.batch_size)):
            output = model(batch_input)
            if args.use_auxiliary_losses:
                output, dig1, dig2 = output
                original_loss = criterion(output, batch_target)
                auxiliary_loss = criterion(dig1, batch_classes[:, 0]) + criterion(dig2, batch_classes[:, 1])
                loss = original_loss + args.auxiliary_losses_rate * auxiliary_loss
            else:
                loss = criterion(output, batch_target)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if args.display_train and epoch % args.display_frequency == 0:
            error_rate = compute_error_rate(model, data_input, data_target, args, False)
            info = 'Train epoch %d: train loss: %.4f, train error rate: %.2f%%.' \
                   % (epoch, train_loss, 100 * error_rate)
            print(info)
            logs.write('\n%s' % info)


def compute_error_rate(model, data_input, data_target, args, display=False, logs=None):
    error_num = 0
    for batch_input, batch_target in zip(data_input.split(args.batch_size), data_target.split(args.batch_size)):
        if args.use_auxiliary_losses:
            output, _, _ = model(batch_input)
        else:
            output = model(batch_input)
        pred = output.argmax(-1)
        batch_error_num = torch.Tensor([1 if p != t else 0 for p, t in zip(pred, batch_target)]).sum().item()
        error_num += batch_error_num
    if display:
        info = 'Test error rate: %.2f%%.' % (100 * error_num / data_input.size(0))
        logs.write('\n%s' % info)
        print(info)
    return error_num / data_input.size(0)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=MultilayerPerceptron)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--hidden_unit', default=64, type=int)
    parser.add_argument('--block_num', default=3, type=int)

    parser.add_argument('--use_weight_sharing', default=True, type=str2bool)
    parser.add_argument('--use_auxiliary_losses', default=True, type=str2bool)
    parser.add_argument('--auxiliary_losses_rate', default=1e-3, type=float)

    parser.add_argument('--use_dropout', default=False, type=str2bool)
    parser.add_argument('--dropout_rate', default=0.5, type=float)

    parser.add_argument('--rounds_num', default=20, type=int)
    parser.add_argument('--data_size', default=1000, type=int)
    parser.add_argument('--num_epochs', default=25, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--SEED', default=666, type=int)

    parser.add_argument('--display_train', default=True, type=str2bool)
    parser.add_argument('--display_frequency', default=5, type=int)
    parser.add_argument('--display_test', default=True, type=str2bool)

    args = parser.parse_args()

    # For reproducibility
    np.random.seed(args.SEED)
    torch.manual_seed(args.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.SEED)
        torch.backends.cudnn.deterministic = True

    # Set device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Prepare data
    # load data
    train_input, train_target, train_classes, test_input, test_target, test_classes = \
        prologue.generate_pair_sets(args.data_size)
    # move to device
    train_input, train_target, train_classes = train_input.to(device), train_target.to(device), train_classes.to(device)
    test_input, test_target, test_classes = test_input.to(device), test_target.to(device), test_classes.to(device)
    # normalize input data
    mean, std = train_input.mean(), train_input.std()
    train_input = train_input.sub_(mean).div_(std)
    test_input = test_input.sub_(mean).div_(std)

    # Record results
    logs = open('logs.txt', mode='a')
    logs.write(' '.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))

    # Train model
    error_rates = []
    for _ in range(args.rounds_num):
        model = args.model(args)
        train_model(model, train_input, train_target, train_classes, args, device, logs)
        error_rate = compute_error_rate(model, test_input, test_target, args, args.display_test, logs)
        error_rates.append(error_rate)

    info = 'Average test error rate is %.2f%%, and standard deviation is %.4f.' \
           % (100*(np.array(error_rates).mean()), np.array(error_rates).std())
    print(info)
    logs.write('\n%s\nDone.\n\n' % info)

    print('Done.')

