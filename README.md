# Classification, weight sharing, auxiliary losses

**EPFL | [Deep Learning (EE-559)](https://fleuret.org/ee559/) (Spring 2020) | Mini-project 1**  

![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)
![Pytorch 1.13.1](https://img.shields.io/badge/pytorch-1.4.0-orange.svg)

## About
- This is our implementaion for the mini-project 1 in the Deep leaning course at EPFL.
  - **Team member**: Pengkang Guo, Xiaoyu Lin， Zhenyu Zhu
- [[report](report.pdf)]

## Project Discription
In this project, we compare the performance of different neural network structures, including multilayer perceptron (MLP), convolutional neural network (CNN) and residual network (ResNet), in the task of identifying which of the two handwritten digits is larger. The aim is to study the impact of auxiliary loss and weight sharing
on the task of image classification. All our data comes from the MNIST database. Both the training set and the test set are composed of 1000 pairs of 14 × 14 grayscale images of handwritten digits (1000×2×14×14 tensors).

## Requirements
Pytorch

## Run
From the root of the project: `python test.py`

## Description of the files
* module.py: the implementation of the modules
  * Includes `Linear`, `Relu`, `Tanh`, `LossMSE` and `Sequential`.
* test.py: the required basic Python script using our framework `module.py`.  <br>
  * Generates the dataset, initializes and trains the required model with three hidden layers of 25 units.
  * Generates an output file, `logs.out`, logging the loss and error rate of each epoch. Calculates and prints the average test error rate, average time and their standard deviations.
* test_plot.py: an upgraded version of `test.py`.   <br>
  * Has all the functions of `test.py`. <br>
  * Generates the images of the dataset, training error rate and test error rate.
* test_torch.py: an Pytorch version of `test_plot.py`.   <br>
  * has all the functions of `test_plot.py` but is implemented using Pytorch
