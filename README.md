# deep_learning_notebooks

This repository contains a collection of Juypter notebooks showcasing fundamental deep learning architectures trained and tested on textbook datasets (such as MNIST).

Popular numerical programming and deep learning libraries such as NumPy and PyTorch are used for model implementations.

The descriptions below provide a brief summary of each notebook.

## Linear Regression Model and a Binary Classifier

**'lin_regression_and_binary_classifier.ipynb'** contains two separate parts.

(1) Polynomial and ridge regression models trained on an artificially constructed dataset.

(2) A logistic regression model trained on the 'Pima Indians Diabetes Dataset' which contains different clinical parameters for multiple subjects along with the label (diabetic or not-diabetic) for binary classification.

## Multi-Category Multi-Layer Neural Network Classifier

**'multi-cat_classifier_NN.ipynb'** uses the popular MNIST dataset to train a multi-layer neural network for multi-class (digits 0-9) classification.

## Optimization Techniques (dropout, momentum and batch norm) for Training of a Multilayer Neural Network

Like the previous notebook, **'optimized_classifier_NN.ipynb'** shows the training of a multilayer neural network to classify the MNIST handwritten digits. 
However, this notebook showcases the use of optimization techniques like dropout, momentum, learning rate scheduling, and minibatch gradient descent.

## Convolutional Neural Network (CNN)

In **'CNN.ipynb'**, a CNN is built and trained on the FashionMNIST dataset for image classification. 

Implementation in PyTorch.

## Generative Adversarial Network (GAN)

To implement a GAN, we basically require 5 components:

- Real Dataset (real distribution)
- Low dimensional random noise that is input to the Generator to produce fake images
- Generator that generates fake images
- Discriminator that acts as an expert to distinguish real and fake images.
- Training loop where the competition occurs and models better themselves.

**'GAN.ipynb'** shows the implementation of each of these components to build a GAN to generate images that resemble the digits from the MNIST dataset.

Implementation in PyTorch.

## Domain Adaptation Model (DANN)

**'domain_adaptation_model.ipynb'** uses the PyTorch deep learning library to build and train an unsupervised domain adaptation model to transfer knowledge from the SVHN dataset to MNIST dataset using adversarial feature alignment. 
