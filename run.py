#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : DeepLearningProject1
# @Author       : Xiaoyu LIN
# @File         : run.py.py
# @Description  :

import subprocess

if __name__ == '__main__':

    common_cmd = 'python test.py '
    for model in ['MultilayerPerceptron', 'ConvolutionalNeuralNetwork', 'ResNet']:
        for use_dropout in ['False', 'True']:
            for use_weight_sharing, use_auxiliary_losses in [['False', 'False'],['True', 'False'],['True', 'True']]:
                cmd = common_cmd + '--model=%s --use_dropout=%s --use_weight_sharing=%s --use_auxiliary_losses=%s' % \
                      (model, use_dropout, use_weight_sharing, use_auxiliary_losses)
                p = subprocess.Popen(cmd, shell=True)
                p.wait()

    print('Done')

